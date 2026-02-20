# chat/chat_agent.py
import os
import re
import io
import uuid
import json
import traceback
from contextlib import redirect_stdout
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from core.state import MasterState

from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLMs
# Using 8B for fast, cheap routing, and 70B for heavy SQL/Python generation
router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
coder_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

class PythonREPL:
    """A safe execution sandbox that captures prints and errors."""
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

def get_database_schema(working_files: dict) -> tuple[str, dict]:
    """Returns the text schema for the LLM and a clean dictionary of DataFrames for DuckDB."""
    schema_info = ""
    clean_dfs = {}
    
    for filename, path in working_files.items():
        # Create a SQL-safe table name (e.g., 'messy_sales.pkl' -> 'messy_sales')
        table_name = re.sub(r'\W|^(?=\d)', '_', filename.replace('.pkl', '').replace('.csv', ''))
        
        try:
            df = pd.read_pickle(path)
            clean_dfs[table_name] = df
            
            schema_info += f"\nTable Name: `{table_name}`\n"
            schema_info += f"Columns & Types:\n"
            for col, dtype in df.dtypes.items():
                schema_info += f"  - {col} ({dtype})\n"
            schema_info += f"Sample Row: {df.head(1).to_dict(orient='records')}\n"
        except Exception as e:
            pass
            
    return schema_info, clean_dfs

def intent_router_node(state: MasterState):
    """Fast node using the 8B model to classify user intent."""
    working_files = state.get('working_files', {})
    schema_info, _ = get_database_schema(working_files)
    user_input = state.get('user_input', '')
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent router for a Data AI.
        Determine if the user's request requires calculating numbers/text (query) OR generating a chart/graph (visualize).
        
        Respond with EXACTLY ONE WORD:
        - "query" (for questions like "What is the average price?", "Show top 5 sales", etc.)
        - "visualize" (for questions like "Plot a bar chart", "Show the distribution", "Graph this", etc.)
        """),
        ("user", "Data Schema:\n{schema}\n\nUser Request: {request}")
    ])
    
    response = (prompt | router_llm).invoke({"schema": schema_info, "request": user_input})
    intent = response.content.lower().strip().replace("'", "").replace(".", "")
    
    if "visualize" in intent or "plot" in intent or "chart" in intent:
        intent = 'visualize'
    else:
        intent = 'query'
        
    return {"next_step": intent, "error": None}

def sql_query_node(state: MasterState):
    """Generates and executes DuckDB SQL against the Pandas DataFrames."""
    working_files = state.get('working_files', {})
    schema_info, clean_dfs = get_database_schema(working_files)
    user_input = state.get('user_input', '')
    
    # 1. Generate SQL
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert DuckDB SQL Developer.
        Write a SQL query to answer the user's question based on the provided tables.
        Return ONLY valid SQL code inside ```sql ... ``` blocks. Do not add explanations.
        """),
        ("user", "Schema:\n{schema}\n\nQuestion: {request}")
    ])
    
    sql_response = (sql_prompt | coder_llm).invoke({"schema": schema_info, "request": user_input})
    
    raw_sql = sql_response.content
    match = re.search(r"```sql(.*?)```", raw_sql, re.DOTALL | re.IGNORECASE)
    sql_query = match.group(1).strip() if match else raw_sql.strip()
    
    messages = list(state.get("messages", []))
    
    # 2. Execute SQL via DuckDB safely
    try:
        conn = duckdb.connect()
        for table_name, df in clean_dfs.items():
            conn.register(table_name, df)
            
        query_result_df = conn.execute(sql_query).df()
        query_result_str = query_result_df.head(20).to_string()
        conn.close()
    except Exception as e:
        # FIX: Append the error to messages so the UI doesn't crash!
        error_msg = f"SQL Execution Failed: {str(e)}\nAttempted Query: {sql_query}"
        messages.append(AIMessage(content=error_msg))
        return {"messages": messages, "error": error_msg, "next_step": "end"}

    # 3. Summarize the answer
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful Data Analyst. Summarize the answer to the user's question using the raw SQL results below in a clean, human-readable way."),
        ("user", "Question: {request}\nSQL Result:\n{result}")
    ])
    
    summary_response = (summary_prompt | coder_llm).invoke({"request": user_input, "result": query_result_str})
    messages.append(AIMessage(content=summary_response.content))
    
    return {"messages": messages, "error": None}

def visualizer_node(state: MasterState):
    """Generates Python code to plot data and saves it to the sandbox."""
    working_files = state.get('working_files', {})
    user_input = state.get('user_input', '')
    error_feedback = f"\n\nPrevious Error to Fix: {state.get('error')}" if state.get('error') else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Data Visualization Expert.
        Write a Python script to create the requested chart.
        
        CRITICAL EXECUTION CONTEXT: 
        I will provide a dictionary `working_files` mapping filenames to their temporary pickle paths.
        
        YOUR SCRIPT MUST:
        1. Import pandas, matplotlib.pyplot as plt, and seaborn as sns.
        2. Define the `working_files` dictionary exactly as provided.
        3. Load the relevant dataframe using `pd.read_pickle(path)`.
        4. Generate the plot.
        5. Generate a unique filename: `unique_name = f"sandbox/plot_{{__import__('uuid').uuid4().hex[:8]}}.png"`
        6. Save the plot: `plt.savefig(unique_name, bbox_inches='tight')`
        7. Print ONLY the file path using `print(f"SAVED_CHART:{{unique_name}}")`
        8. Clear the plot: `plt.close('all')`
        
        RETURN ONLY VALID PYTHON CODE inside ```python ... ``` blocks.{error_feedback}
        """),
        ("user", "Working Files Paths:\n{files}\n\nRequest: {request}")
    ])
    
    # FIX: Added "error_feedback" to the invoke dictionary here!
    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "request": user_input,
        "error_feedback": error_feedback 
    })
    
    raw_code = response.content
    match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
    code = match.group(1).strip() if match else raw_code.strip()
    
    # Execute safely in REPL Sandbox
    result = repl_sandbox.run(code)
    
    if result["error"]:
        return {"error": result["error"], "iteration_count": state.get("iteration_count", 0) + 1}
        
    # Extract the saved chart path from standard output
    output = result["output"]
    chart_match = re.search(r"SAVED_CHART:(.*\.png)", output)
    
    messages = list(state.get("messages", []))
    charts_generated = list(state.get("charts_generated", []))
    
    if chart_match:
        chart_path = chart_match.group(1).strip()
        charts_generated.append(chart_path)
        messages.append(AIMessage(content=f"I have generated the chart. You can view it here: {chart_path}"))
    else:
        messages.append(AIMessage(content="The code executed successfully, but no chart was saved."))
        
    return {"messages": messages, "charts_generated": charts_generated, "error": None}

def route_intent(state: MasterState):
    if state.get("next_step") == "visualize":
        return "visualizer"
    return "sql_query"

def route_visualizer_retry(state: MasterState):
    if state.get("error"):
        # Basic loop prevention
        if state.get("iteration_count", 0) > 2:
            return END
        return "visualizer"
    return END

def build_chat_graph():
    workflow = StateGraph(MasterState)
    
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("sql_query", sql_query_node)
    workflow.add_node("visualizer", visualizer_node)
    
    workflow.set_entry_point("intent_router")
    
    workflow.add_conditional_edges("intent_router", route_intent, {"sql_query": "sql_query", "visualizer": "visualizer"})
    
    workflow.add_edge("sql_query", END)
    workflow.add_conditional_edges("visualizer", route_visualizer_retry, {"visualizer": "visualizer", END: END})
    
    return workflow