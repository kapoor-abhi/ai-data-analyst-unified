# ingestion/agent.py
import os
import re
import json
import shutil
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt 
from core.state import MasterState
from core.sandbox import DockerREPL
from dotenv import load_dotenv
from langchain_core.runnables.config import var_child_runnable_config, RunnableConfig

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SANDBOX_DIR = "sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)

coder_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
repl_sandbox = DockerREPL(sandbox_dir="sandbox", image_name="python-data-sandbox:latest")

def get_dataframe_schema(working_files: dict) -> str:
    schema_info = ""
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            schema_info += f"\n--- File: {name} ---\n"
            schema_info += f"Columns: {list(df.columns)}\n"
            schema_info += f"Data Types: {list(df.dtypes)}\n"
            schema_info += f"Shape: {df.shape}\n"
        except Exception:
            pass
    return schema_info

def ingest_data_node(state: MasterState):
    file_paths = state.get('file_paths', [])
    working_files = state.get('working_files', {})
    
    try:
        for path in file_paths:
            filename = os.path.basename(path)
            if filename not in working_files:
                df = pd.read_csv(path)
                pickle_path = os.path.join(SANDBOX_DIR, f"{filename}.pkl")
                df.to_pickle(pickle_path)
                working_files[filename] = pickle_path
        return {"working_files": working_files, "error": None}
    except Exception as e:
        return {"error": str(e)}

def route_after_ingestion(state: MasterState):
    if state.get("error"): return "end"
    return "optimize_data"

def optimize_data_node(state: MasterState):
    working_files = state.get('working_files', {})
    is_large = False
    
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            if len(df) > 50000:
                is_large = True
                for col in df.select_dtypes(include=['object']).columns:
                    if len(df[col].unique()) / len(df) < 0.5:
                        df[col] = df[col].astype('category')
                df.to_pickle(path)
        except Exception:
            pass
    return {"is_large_dataset": is_large}

def column_selection_node(state: MasterState):
    working_files = state.get('working_files', {})
    schema = get_dataframe_schema(working_files)
    is_large = state.get('is_large_dataset', False)
    
    performance_instruction = "PERFORMANCE MODE ACTIVE: DO NOT use loops. Use vectorized Pandas operations only." if is_large else ""
    error_context = f"\nPREVIOUS ERROR: {state.get('error')}" if state.get("error") else ""
    
    raw_intent = state.get('user_feedback') or state.get('user_input', '')
    
    # Intercept system commands
    if str(raw_intent).strip().lower() in ['approve', 'yes', 'ok', '', 'load the data.', 'load and clean this data.']:
        current_intent = "Do not filter or drop any rows. Just load the data exactly as is."
    else:
        current_intent = raw_intent
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a High-Performance Python Data Expert. 
        
        CRITICAL EXECUTION CONTEXT: 
        I will provide a dictionary `working_files` mapping filenames to their temporary pickle paths.
        
        YOUR SCRIPT MUST:
        1. Import pandas and numpy.
        2. Define the `working_files` dictionary exactly as provided.
        3. Loop through the dictionary, use `pd.read_pickle(path)` to load the dataframe.
        4. Apply the requested transformations.
        5. Use `df.to_pickle(path)` to save the dataframe back to the EXACT SAME PATH.
        
        FATAL ERRORS TO AVOID:
        - DO NOT create dummy data. Use ONLY the files provided.
        - DO NOT use print() statements.
        - NEVER drop rows (no dropna(), no drop_duplicates()) during ingestion. Leave that for the cleaning phase.
        - NEVER ALTER DATA TYPES. Do NOT use pd.to_datetime(), pd.to_numeric(), or astype(). Leave all type casting for the cleaning phase.
        
        {performance_instruction}
        
        RETURN ONLY VALID PYTHON CODE inside ```python ... ``` blocks.
        """),
        ("user", "Working Files Paths:\n{files}\n\nSchemas:\n{schema}\n\nUser Query: {query}\n{error_context}")
    ])

    chain = prompt_template | coder_llm
    response = chain.invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema, 
        "query": current_intent, 
        "performance_instruction": performance_instruction,
        "error_context": error_context
    })

    raw = response.content
    match = re.search(r"```python(.*?)```", raw, re.DOTALL)
    return {"python_code": match.group(1).strip() if match else raw.strip(), "error": None}

def execute_code_node(state: MasterState):
    code = state.get('python_code', '')
    working_files = state.get('working_files', {})
    
    for name, path in working_files.items():
        if os.path.exists(path):
            shutil.copy(path, path + ".bak")
            
    result = repl_sandbox.run(code)
    
    if result["error"]:
        for name, path in working_files.items():
            if os.path.exists(path + ".bak"):
                shutil.move(path + ".bak", path)
        current_iter = state.get("iteration_count", 0) + 1
        return {"error": result["error"], "iteration_count": current_iter}
        
    for name, path in working_files.items():
        if os.path.exists(path + ".bak"):
            os.remove(path + ".bak")
            
    return {"error": None, "iteration_count": 0}

async def human_review_node(state: MasterState, config: RunnableConfig):
    # Manually patch the broken context
    var_child_runnable_config.set(config)
    
    feedback = interrupt("Execution complete. Is the data looking correct, or do you need changes?")
    if feedback: return {"user_feedback": feedback, "error": None}
    return {"error": None}

def router_logic(state: MasterState):
    if state.get("error"):
        if state.get("iteration_count", 0) >= 3: return "review" 
        return "retry_llm"
    return "review"

def route_after_review(state: MasterState):
    feedback = state.get("user_feedback")
    if feedback and str(feedback).strip().lower() not in ["yes", "y", "looks good", "approve", "ok", "proceed"]:
        return "reasoning"
    return "end"

def build_ingestion_graph():
    workflow = StateGraph(MasterState)
    workflow.add_node("ingest_data", ingest_data_node)
    workflow.add_node("optimize_data", optimize_data_node)
    workflow.add_node("reasoning", column_selection_node)
    workflow.add_node("execution", execute_code_node)
    workflow.add_node("human_review", human_review_node)
    workflow.set_entry_point("ingest_data")
    workflow.add_conditional_edges("ingest_data", route_after_ingestion, {"optimize_data": "optimize_data", "end": END})
    workflow.add_edge("optimize_data", "reasoning")
    workflow.add_edge("reasoning", "execution")
    workflow.add_conditional_edges("execution", router_logic, {"retry_llm": "reasoning", "review": "human_review"})
    workflow.add_conditional_edges("human_review", route_after_review, {"reasoning": "reasoning", "end": END})
    return workflow