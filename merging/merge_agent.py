# merge_agent.py
import os
import re
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt # Added interrupt import explicitly
from core.state import MasterState

# Import our new secure Docker sandbox
from core.sandbox import DockerREPL

from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SANDBOX_DIR = "sandbox"

# OPTIMIZATION: Fast model for analysis and simple reasoning
fast_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# OPTIMIZATION: Heavy 70B model strictly for complex Python pandas code generation
coder_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# SECURITY FIX: Initialize the Docker Sandbox
repl_sandbox = DockerREPL(sandbox_dir="sandbox", image_name="python-data-sandbox:latest")

def get_dataframe_schema(working_files: dict) -> str:
    schema_info = ""
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            schema_info += f"\n--- File: {name} ---\n"
            schema_info += f"Columns: {list(df.columns)}\n"
            schema_info += f"Shape: {df.shape}\n"
        except Exception:
            pass
    return schema_info

def analyze_merge_node(state: MasterState):
    working_files = state.get('working_files', {})
    
    if len(working_files) < 2:
        return {"suggestion": "Only one file provided. No merge needed.", "error": "Not enough files", "iteration_count": 0}

    schema = get_dataframe_schema(working_files)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Data Integration Expert.
        Analyze the schemas of the provided DataFrames.
        Identify the most likely column(s) to merge these datasets on.
        
        Rules:
        1. Look for columns with identical or similar names (e.g., 'ID' vs 'Client_ID').
        2. Output a SHORT suggestion string explaining the strategy. 
        """),
        ("user", "Schemas:\n{schema}")
    ])

    chain = prompt | fast_llm
    response = chain.invoke({"schema": schema})
    
    return {"suggestion": response.content, "error": None, "iteration_count": 0}

from langchain_core.runnables.config import var_child_runnable_config, RunnableConfig

# Notice we added `config: RunnableConfig` inside the parentheses here!
async def human_strategy_node(state: MasterState, config: RunnableConfig):
    # Manually patch the broken context
    var_child_runnable_config.set(config)
    
    if state.get("error") and state.get("iteration_count", 0) >= 2:
        msg = f"Code generation failed multiple times with error: {state.get('error')}. Please provide a simpler merge instruction or type 'skip' to bypass."
    else:
        msg = state.get('suggestion')
        
    feedback = interrupt(msg)
    
    if feedback and str(feedback).strip().lower() == 'skip':
        return {"user_feedback": "skip", "error": None, "iteration_count": 0}
        
    return {"user_feedback": feedback, "error": None, "iteration_count": 0}

def generate_merge_code_node(state: MasterState):
    if str(state.get('user_feedback', '')).strip().lower() == 'skip':
        return {"python_code": ""}
        
    working_files = state.get('working_files', {})
    schema = get_dataframe_schema(working_files)
    
    raw_instruction = state.get('user_feedback', state.get('suggestion'))
    error_context = f"\n\nPrevious Error to Fix: {state.get('error')}" if state.get("error") else ""

    # NEW: Intercept system commands so they aren't treated as Pandas filters
    if str(raw_instruction).strip().lower() in ['approve', 'yes', 'ok', '']:
        instruction = f"Proceed with your suggested strategy: {state.get('suggestion')}"
    else:
        instruction = raw_instruction

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Pandas Expert.
        
        CRITICAL EXECUTION CONTEXT: 
        I will provide a dictionary `working_files` mapping filenames to their temporary pickle file paths.
        
        YOUR SCRIPT MUST EXACTLY DO THIS:
        1. Import pandas.
        2. Define the `working_files` dictionary exactly as provided.
        3. Load the dataframes using `pd.read_pickle(path)`.
        4. Merge them according to the user instruction.
        5. Save the final merged dataframe EXACTLY to 'sandbox/merged_dataset.pkl' using `.to_pickle('sandbox/merged_dataset.pkl')`.
        
        Rules:
        1. Return ONLY valid Python code inside Markdown blocks (```python ... ```).
        2. Use `pd.merge()`.
        3. CRITICAL: Default to `how='outer'` to prevent accidental data loss, unless specifically told otherwise.
        """),
        ("user", "Working Files Paths:\n{files}\n\nSchemas:\n{schema}\n\nInstruction: {instruction}\n{error_context}")
    ])

    chain = prompt | coder_llm
    response = chain.invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema,
        "instruction": instruction,
        "error_context": error_context
    })

    raw_content = response.content
    match = re.search(r"```python(.*?)```", raw_content, re.DOTALL)
    return {"python_code": match.group(1).strip() if match else raw_content.strip()}

def execute_merge_node(state: MasterState):
    code = state.get('python_code', '')
    
    if not code:
        return {"error": None, "iteration_count": 0}
    
    result = repl_sandbox.run(code)
    
    if result["error"]:
        current_iter = state.get("iteration_count", 0) + 1
        return {"error": result["error"], "iteration_count": current_iter}
        
    merged_path = "sandbox/merged_dataset.pkl"
    
    # NEW: Verify the merge didn't decimate the dataset
    try:
        df = pd.read_pickle(merged_path)
        if len(df) == 0:
            current_iter = state.get("iteration_count", 0) + 1
            return {
                "error": "CRITICAL AUDIT FAILURE: The merge resulted in 0 rows. Please check your join keys or use how='outer'.", 
                "iteration_count": current_iter
            }
    except Exception as e:
        current_iter = state.get("iteration_count", 0) + 1
        return {"error": f"Failed to load merged dataset for audit: {e}", "iteration_count": current_iter}
        
    return {
        "working_files": {"merged_dataset.pkl": merged_path}, 
        "error": None,
        "iteration_count": 0
    }

def route_merge_retry(state: MasterState):
    if state.get("error"):
        if state.get("iteration_count", 0) >= 2:
            return "human_strategy" 
        return "retry"
    return "success"

def build_merge_graph():
    workflow = StateGraph(MasterState)
    
    workflow.add_node("analyze", analyze_merge_node)
    workflow.add_node("human_strategy", human_strategy_node)
    workflow.add_node("generate", generate_merge_code_node)
    workflow.add_node("execute", execute_merge_node)
    
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "human_strategy")
    workflow.add_edge("human_strategy", "generate")
    workflow.add_edge("generate", "execute")
    
    workflow.add_conditional_edges(
        "execute",
        route_merge_retry,
        {"retry": "generate", "human_strategy": "human_strategy", "success": END}
    )
    
    return workflow