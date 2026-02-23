# chat/chat_agent.py
import os
import re
import json
import pandas as pd
import duckdb
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    user_input = state.get('user_input', '')
    
    # FIX: We NO LONGER pass the chat history to the 8B router. 
    # It only needs to evaluate the current user_input to prevent hallucinating.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent classification router for a Data AI.
        Evaluate the user's latest request and determine if it requires calculating numbers/text (query) OR generating a chart/graph (visualize).
        
        Respond with EXACTLY ONE WORD:
        - "query" (for questions like "What is the average price?", "Show top 5 sales", "Calculate revenue")
        - "visualize" (for questions like "Plot a bar chart", "Show the distribution", "Graph this")
        """),
        ("user", "{request}")
    ])
    
    response = (prompt | router_llm).invoke({"request": user_input})
    intent = response.content.lower().strip().replace("'", "").replace(".", "")
    
    # Fallback keyword logic to guarantee accuracy
    if any(word in intent for word in ["visualize", "plot", "chart", "graph"]):
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
    
    raw_msgs = state.get("messages", [])
    context_msgs = raw_msgs[-5:] if len(raw_msgs) > 0 else []
    
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert DuckDB SQL Developer.
        Write a SQL query to answer the user's question based on the provided tables.
        
        Schema:
        {schema}
        
        Return ONLY valid SQL code inside ```sql ... ``` blocks. Do not add explanations.
        """),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    sql_response = (sql_prompt | coder_llm).invoke({"schema": schema_info, "messages": context_msgs})
    
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
        return {"messages": [AIMessage(content=error_msg)], "error": error_msg, "next_step": "end", "iteration_count": 0}

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Data Analyst. Summarize the answer to the user's question using the raw SQL results below in a clean, human-readable way.
        
        SQL Result:
        {result}
        """),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    summary_response = (summary_prompt | coder_llm).invoke({"result": query_result_str, "messages": context_msgs})
    
    return {"messages": [AIMessage(content=summary_response.content)], "error": None, "iteration_count": 0}

def visualizer_node(state: MasterState):
    working_files = state.get('working_files', {})
    schema_info, _ = get_database_schema(working_files) 
    
    error_feedback = f"\n\nPREVIOUS EXECUTION ERROR TO FIX:\n{state.get('error')}" if state.get('error') else ""
    
    raw_msgs = state.get("messages", [])
    context_msgs = raw_msgs[-5:] if len(raw_msgs) > 0 else []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Data Visualization Expert.
        Write a Python script to create the requested chart.
        
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
        
        Data Schemas:
        {schema}
        
        Working Files Paths:
        {files}
        {error_feedback}
        """),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema_info, 
        "error_feedback": error_feedback,
        "messages": context_msgs
    })
    
    raw_code = response.content
    match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
    code = match.group(1).strip() if match else raw_code.strip()
    
    result = repl_sandbox.run(code)
    
    if result["error"]:
        current_iter = state.get("iteration_count", 0) + 1
        if current_iter > 2:
            return {"error": result["error"], "iteration_count": current_iter, "messages": [AIMessage(content=f"I failed to generate the chart after multiple attempts. The execution error was: {result['error']}")]}
        
        return {"error": result["error"], "iteration_count": current_iter}
        
    output = result["output"]
    chart_match = re.search(r"SAVED_CHART:(.*\.png)", output)
    
    charts_generated = list(state.get("charts_generated", []))
    
    if chart_match:
        chart_path = chart_match.group(1).strip()
        charts_generated.append(chart_path)
        return {"messages": [AIMessage(content=f"I have generated the chart. You can view it here: {chart_path}")], "charts_generated": charts_generated, "error": None, "iteration_count": 0}
    else:
        return {"messages": [AIMessage(content="The code executed successfully, but no chart was saved.")], "charts_generated": charts_generated, "error": None, "iteration_count": 0}

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