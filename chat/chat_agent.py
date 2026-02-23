# chat_agent.py
import os
import re
import json
import pandas as pd
import duckdb
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from core.state import MasterState

# Import our new secure Docker sandbox
from core.sandbox import DockerREPL

from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLMs
router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
coder_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# Initialize the Docker Sandbox
repl_sandbox = DockerREPL(sandbox_dir="sandbox", image_name="python-data-sandbox:latest")

def get_database_schema(working_files: dict) -> tuple[str, dict]:
    """Returns the text schema for the LLM and a clean dictionary of DataFrames for DuckDB."""
    schema_info = ""
    clean_dfs = {}
    
    for filename, path in working_files.items():
        table_name = re.sub(r'\W|^(?=\d)', '_', filename.replace('.pkl', '').replace('.csv', ''))
        
        try:
            df = pd.read_pickle(path)
            clean_dfs[table_name] = df
            
            schema_info += f"\nTable Name: `{table_name}`\n"
            schema_info += f"Columns & Types:\n"
            for col, dtype in df.dtypes.items():
                schema_info += f"  - {col} ({dtype})\n"
            schema_info += f"Sample Row: {df.head(1).to_dict(orient='records')}\n"
        except Exception:
            pass
            
    return schema_info, clean_dfs

def intent_router_node(state: MasterState):
    working_files = state.get('working_files', {})
    schema_info, _ = get_database_schema(working_files)
    user_input = state.get('user_input', '')
    
    # MEMORY FIX: Get the last 5 messages, excluding the current query we just appended
    raw_msgs = state.get("messages", [])
    recent_msgs = raw_msgs[:-1][-5:] if len(raw_msgs) > 0 else []
    history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in recent_msgs]) if recent_msgs else "No prior conversation."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent router for a Data AI.
        Determine if the user's request requires calculating numbers/text (query) OR generating a chart/graph (visualize).
        
        Conversation Context (Last 5 Messages):
        {history}
        
        Respond with EXACTLY ONE WORD:
        - "query" (for questions like "What is the average price?", "Show top 5 sales", etc.)
        - "visualize" (for questions like "Plot a bar chart", "Show the distribution", "Graph this", etc.)
        """),
        ("user", "Data Schema:\n{schema}\n\nUser Request: {request}")
    ])
    
    response = (prompt | router_llm).invoke({"schema": schema_info, "history": history_str, "request": user_input})
    intent = response.content.lower().strip().replace("'", "").replace(".", "")
    
    if "visualize" in intent or "plot" in intent or "chart" in intent:
        intent = 'visualize'
    else:
        intent = 'query'
        
    return {
        "next_step": intent, 
        "error": None, 
        "iteration_count": 0 
    }

def sql_query_node(state: MasterState):
    working_files = state.get('working_files', {})
    schema_info, clean_dfs = get_database_schema(working_files)
    user_input = state.get('user_input', '')
    messages = list(state.get("messages", []))
    
    # MEMORY FIX
    recent_msgs = messages[:-1][-5:] if len(messages) > 0 else []
    history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in recent_msgs]) if recent_msgs else "No prior conversation."
    
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert DuckDB SQL Developer.
        Write a SQL query to answer the user's question based on the provided tables.
        
        Conversation Context (Last 5 Messages):
        {history}
        
        Return ONLY valid SQL code inside ```sql ... ``` blocks. Do not add explanations.
        """),
        ("user", "Schema:\n{schema}\n\nQuestion: {request}")
    ])
    
    sql_response = (sql_prompt | coder_llm).invoke({"schema": schema_info, "history": history_str, "request": user_input})
    
    raw_sql = sql_response.content
    match = re.search(r"```sql(.*?)```", raw_sql, re.DOTALL | re.IGNORECASE)
    sql_query = match.group(1).strip() if match else raw_sql.strip()
    
    try:
        conn = duckdb.connect()
        for table_name, df in clean_dfs.items():
            conn.register(table_name, df)
            
        query_result_df = conn.execute(sql_query).df()
        query_result_str = query_result_df.head(20).to_string()
        conn.close()
    except Exception as e:
        error_msg = f"SQL Execution Failed: {str(e)}\nAttempted Query: {sql_query}"
        messages.append(AIMessage(content=error_msg))
        return {"messages": messages, "error": error_msg, "next_step": "end", "iteration_count": 0}

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Data Analyst. Summarize the answer to the user's question using the raw SQL results below in a clean, human-readable way.
        
        Conversation Context:
        {history}
        """),
        ("user", "Question: {request}\nSQL Result:\n{result}")
    ])
    
    summary_response = (summary_prompt | coder_llm).invoke({"history": history_str, "request": user_input, "result": query_result_str})
    messages.append(AIMessage(content=summary_response.content))
    
    return {"messages": messages, "error": None, "iteration_count": 0}

def visualizer_node(state: MasterState):
    working_files = state.get('working_files', {})
    schema_info, _ = get_database_schema(working_files) 
    
    user_input = state.get('user_input', '')
    error_feedback = f"\n\nPrevious Error to Fix: {state.get('error')}" if state.get('error') else ""
    raw_msgs = state.get("messages", [])
    
    # MEMORY FIX
    recent_msgs = raw_msgs[:-1][-5:] if len(raw_msgs) > 0 else []
    history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in recent_msgs]) if recent_msgs else "No prior conversation."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Data Visualization Expert.
        Write a Python script to create the requested chart using the provided Data Schemas.
        
        Conversation Context (Last 5 Messages):
        {history}
        
        CRITICAL EXECUTION CONTEXT: 
        I will provide a dictionary `working_files` mapping filenames to their temporary pickle paths.
        
        YOUR SCRIPT MUST:
        1. Import pandas, matplotlib.pyplot as plt, seaborn as sns, and uuid.
        2. Define the `working_files` dictionary exactly as provided.
        3. Load the relevant dataframe using `pd.read_pickle(path)`.
        4. Generate the plot based on the actual columns in the schema.
        5. Generate a unique filename: `unique_name = f"sandbox/plot_{{uuid.uuid4().hex[:8]}}.png"`
        6. Save the plot: `plt.savefig(unique_name, bbox_inches='tight')`
        7. Print ONLY the file path using `print(f"SAVED_CHART:{{unique_name}}")`
        8. Clear the plot: `plt.close('all')`
        
        RETURN ONLY VALID PYTHON CODE inside ```python ... ``` blocks.{error_feedback}
        """),
        ("user", "Working Files Paths:\n{files}\n\nData Schemas:\n{schema}\n\nRequest: {request}")
    ])
    
    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema_info, 
        "history": history_str,
        "request": user_input,
        "error_feedback": error_feedback 
    })
    
    raw_code = response.content
    match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
    code = match.group(1).strip() if match else raw_code.strip()
    
    result = repl_sandbox.run(code)
    
    messages = list(state.get("messages", []))
    
    if result["error"]:
        current_iter = state.get("iteration_count", 0) + 1
        if current_iter > 2:
            messages.append(AIMessage(content=f"I failed to generate the chart after multiple attempts. The execution error was: {result['error']}"))
            return {"error": result["error"], "iteration_count": current_iter, "messages": messages}
        
        return {"error": result["error"], "iteration_count": current_iter}
        
    output = result["output"]
    chart_match = re.search(r"SAVED_CHART:(.*\.png)", output)
    
    charts_generated = list(state.get("charts_generated", []))
    
    if chart_match:
        chart_path = chart_match.group(1).strip()
        charts_generated.append(chart_path)
        messages.append(AIMessage(content=f"I have generated the chart. You can view it here: {chart_path}"))
    else:
        messages.append(AIMessage(content="The code executed successfully, but no chart was saved."))
        
    return {"messages": messages, "charts_generated": charts_generated, "error": None, "iteration_count": 0}

def route_intent(state: MasterState):
    if state.get("next_step") == "visualize":
        return "visualizer"
    return "sql_query"

def route_visualizer_retry(state: MasterState):
    if state.get("error"):
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