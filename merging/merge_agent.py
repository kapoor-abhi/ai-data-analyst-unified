import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from core.state import MasterState
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SANDBOX_DIR = "sandbox"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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
        # FIX: Reset transient state variables just in case
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

    chain = prompt | llm
    response = chain.invoke({"schema": schema})
    
    # FIX: Initialize iteration count for the downstream retry loop
    return {"suggestion": response.content, "error": None, "iteration_count": 0}

def human_strategy_node(state: MasterState):
    # This node acts as an anchor for the LangGraph interrupt. 
    # The frontend will inject the 'user_feedback' here.
    if not state.get('user_feedback'):
        return {"user_feedback": state.get('suggestion')}
    return {"error": None}

def generate_merge_code_node(state: MasterState):
    schema = get_dataframe_schema(state.get('working_files', {}))
    instruction = state.get('user_feedback', state.get('suggestion'))
    error_context = f"\n\nPrevious Error to Fix: {state.get('error')}" if state.get("error") else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Pandas Expert.
        
        CRITICAL CONTEXT:
        1. You are running inside a function where a dictionary named `dfs` IS ALREADY DEFINED.
        2. `dfs` keys are filename strings (e.g. 'file1.csv').
        
        Task: 
        Generate Python code to merge dataframes from `dfs` into a single variable `merged_df`.
        
        Rules:
        1. Return ONLY valid Python code inside Markdown blocks (```python ... ```).
        2. Use `pd.merge()`.
        3. Assign the result to EXACTLY `merged_df`.
        """),
        ("user", "Schemas:\n{schema}\n\nUser Instruction: {instruction}\n{error_context}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "schema": schema,
        "instruction": instruction,
        "error_context": error_context
    })

    raw_content = response.content
    match = re.search(r"```python(.*?)```", raw_content, re.DOTALL)
    code = match.group(1).strip() if match else raw_content.strip()
    
    return {"python_code": code}

def execute_merge_node(state: MasterState):
    code = state.get('python_code', '')
    working_files = state.get('working_files', {})
    
    # 1. Load dataframes from disk into memory
    dfs = {name: pd.read_pickle(path) for name, path in working_files.items()}
    local_scope = {"dfs": dfs, "pd": pd}
    
    try:
        # 2. Execute LLM code
        exec(code, {}, local_scope)
        
        result_df = local_scope.get("merged_df")
        if result_df is None:
            raise ValueError("Code ran, but 'merged_df' variable was not created.")
            
        # 3. Save the merged dataframe as a new file, update registry
        merged_path = "sandbox/merged_dataset.pkl"
        result_df.to_pickle(merged_path)
        
        # FIX: Explicitly clear errors and reset iteration counters on success
        return {
            "working_files": {"merged_dataset.pkl": merged_path}, 
            "error": None,
            "iteration_count": 0
        }
    except Exception as e:
        # FIX: Increment the iteration counter specifically when code execution fails
        current_iter = state.get("iteration_count", 0) + 1
        return {"error": str(e), "iteration_count": current_iter}

def route_merge_retry(state: MasterState):
    if state.get("error"):
        # FIX: Implement a hard limit to prevent infinite loops (stops after 2 retries)
        if state.get("iteration_count", 0) >= 2:
            return "success" 
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
        {"retry": "generate", "success": END}
    )
    
    return workflow