# ingestion/agent.py
import os
import re
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from core.state import MasterState
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ingestion/agent.py
import os
import re
import io
import json
import traceback
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from core.state import MasterState

SANDBOX_DIR = "sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class PythonREPL:
    """A safe execution sandbox."""
    def run(self, code: str) -> dict:
        obs = io.StringIO()
        try:
            with redirect_stdout(obs):
                exec(code, {})
            return {"output": obs.getvalue(), "error": None}
        except Exception as e:
            error_trace = traceback.format_exc()
            return {"output": obs.getvalue(), "error": f"{str(e)}\n{error_trace}"}

repl_sandbox = PythonREPL()

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
    if state.get("error"):
        return "end"
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
    
    performance_instruction = ""
    if is_large:
        performance_instruction = "PERFORMANCE MODE ACTIVE: DO NOT use loops. Use vectorized Pandas operations only."
        
    error_context = f"\nPREVIOUS ERROR: {state.get('error')}" if state.get("error") else ""
    
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
        - DO NOT create dummy data. Use ONLY the files provided in the dictionary.
        - DO NOT use print() statements.
        
        {performance_instruction}
        
        RETURN ONLY VALID PYTHON CODE inside ```python ... ``` blocks.
        """),
        ("user", "Working Files Paths:\n{files}\n\nSchemas:\n{schema}\n\nUser Query: {query}\n{error_context}")
    ])

    chain = prompt_template | llm
    response = chain.invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema, 
        "query": state.get('user_input', ''),
        "performance_instruction": performance_instruction,
        "error_context": error_context
    })

    raw = response.content
    match = re.search(r"```python(.*?)```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()

    return {"python_code": code, "error": None}

def execute_code_node(state: MasterState):
    code = state.get('python_code', '')
    
    result = repl_sandbox.run(code)
    
    if result["error"]:
        return {"error": result["error"]}
        
    return {"error": None}

def human_review_node(state: MasterState):
    return {"error": None}

def router_logic(state: MasterState):
    if state.get("error"):
        return "retry_llm"
    return "review"

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
    workflow.add_edge("human_review", END)
    
    return workflow