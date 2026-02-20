import os
import json
import re
import io
import traceback
import pandas as pd
import numpy as np
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from core.state import MasterState
from dotenv import load_dotenv
import io
import traceback
from contextlib import redirect_stdout

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 1. Flattened, Resilient Pydantic Schemas ---
# We use a flat list of actions instead of deep nesting to prevent LLM hallucinations.
class CleaningAction(BaseModel):
    target_file: str = Field(description="The exact filename, e.g., 'customers.csv'")
    target_column: str = Field(description="The exact column name to clean")
    action_type: Literal["fill_missing", "drop_outliers", "convert_type", "standardize_text", "custom"]
    code_instruction: str = Field(description="Plain English instruction for the Python engineer, e.g., 'Fill missing values with the median'")

class CleaningPlan(BaseModel):
    actions: List[CleaningAction]

# --- 2. Initialize Models & Sandbox ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class PythonREPL:
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if pd.isna(obj): return None
        return super(NpEncoder, self).default(obj)

def strip_markdown(text: str) -> str:
    """Removes ```json and ``` from the LLM output."""
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

# --- 3. Robust Profiler ---
def analyze_column(series: pd.Series):
    """Safely profiles a column with error handling and date recognition."""
    clean_series = series.dropna()
    total_count = len(series)
    non_null_count = len(clean_series)
    
    report = {
        "logical_type": "Unknown",
        "null_count": int(total_count - non_null_count),
        "pct_missing": round((1 - (non_null_count / total_count)) * 100, 2) if total_count > 0 else 0,
        "issues": []
    }
    
    if non_null_count == 0:
        report["logical_type"] = "Empty"
        return report

    # 1. Check for Datetime explicitly
    if pd.api.types.is_datetime64_any_dtype(series) or \
       (clean_series.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').sum() > len(clean_series) * 0.8):
        report["logical_type"] = "Datetime"
        return report

    # 2. Check for Numeric
    numeric_converted = pd.to_numeric(clean_series, errors='coerce')
    num_valid_count = numeric_converted.notna().sum()
    
    if num_valid_count / non_null_count > 0.7:
        report["logical_type"] = "Numeric"
        if num_valid_count < non_null_count:
            report["logical_type"] = "Numeric (Dirty)"
            report["issues"].append("Contains non-numeric characters")
            
        valid_nums = numeric_converted.dropna()
        if not valid_nums.empty:
            try:
                Q1, Q3 = valid_nums.quantile(0.25), valid_nums.quantile(0.75)
                IQR = Q3 - Q1
                outliers = valid_nums[(valid_nums < (Q1 - 1.5 * IQR)) | (valid_nums > (Q3 + 1.5 * IQR))]
                skewness = float(valid_nums.skew()) if len(valid_nums) > 2 else 0.0
                
                report["stats"] = {
                    "min": valid_nums.min(),
                    "max": valid_nums.max(),
                    "mean": round(valid_nums.mean(), 2),
                    "median": round(valid_nums.median(), 2),
                    "skewness": round(skewness, 2),
                    "outlier_count": int(len(outliers))
                }
                if len(outliers) > 0: report["issues"].append("Has Outliers")
                if abs(skewness) > 1: report["issues"].append("Highly Skewed Distribution")
            except Exception:
                pass # Fail silently on math errors, keep the pipeline moving
        return report

    # 3. Check for Categorical
    unique_count = clean_series.nunique()
    if unique_count < 50 or (unique_count / total_count < 0.2):
        report["logical_type"] = "Categorical"
        lower_unique = clean_series.astype(str).str.lower().nunique()
        if lower_unique < unique_count:
            report["issues"].append("Inconsistent Casing")
        return report

    report["logical_type"] = "Text/ID"
    return report

def profiler_node(state: MasterState):
    working_files = state.get('working_files', {})
    full_report = {}
    
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            # PERFORMANCE FIX: Sample massive datasets to prevent blocking the event loop
            sample_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df
            
            file_report = {"rows": len(df), "cols": len(df.columns), "columns": {}}
            for col in sample_df.columns:
                file_report["columns"][col] = analyze_column(sample_df[col])
            full_report[name] = file_report
        except Exception as e:
            print(f"Failed to profile {name}: {e}")
            
    profile_json = json.dumps(full_report, cls=NpEncoder, indent=2)
    return {"deep_profile_report": profile_json, "error": None, "iteration_count": 0}

# --- 4. The Self-Correcting Strategist ---
def strategist_node(state: MasterState):
    profile_json_str = state.get('deep_profile_report', '')
    error_context = state.get('error', '')
    
    schema_str = json.dumps(CleaningPlan.model_json_schema(), indent=2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Principal Data Engineer. Generate a JSON cleaning plan based on the data profile.
        
        CRITICAL INSTRUCTIONS:
        1. Return ONLY valid JSON matching the schema below.
        2. DO NOT wrap the JSON in markdown (```json).
        3. DO NOT include greetings or explanations outside the JSON.
        
        SCHEMA:
        {schema}
        
        LOGIC RULES:
        - If 'Highly Skewed' or 'Has Outliers' -> instruct to fill with median.
        - If Categorical 'Inconsistent Casing' -> instruct to apply str.lower.
        - If Datetime -> instruct to convert using pd.to_datetime(errors='coerce').
        """),
        ("user", "DATA PROFILE:\n{profile}\n\nPREVIOUS ERROR (If any):\n{error}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"schema": schema_str, "profile": profile_json_str, "error": error_context})
    
    raw_json = strip_markdown(response.content)
    
    try:
        # Validate manually so we can catch and route the exact error
        plan_obj = CleaningPlan.model_validate_json(raw_json)
        return {"cleaning_plan": plan_obj.model_dump_json(indent=2), "error": None}
    except ValidationError as e:
        error_msg = f"Pydantic Validation Error. Fix your JSON format. Details: {e.errors()}"
        return {"error": error_msg}

def route_strategist(state: MasterState):
    """If the LLM failed to write valid JSON, loop back immediately."""
    if state.get("error") and "Pydantic" in state.get("error", ""):
        return "strategist"
    return "human_review"

def human_review_plan_node(state: MasterState):
    return {"error": None}

# --- 5. Context-Aware Engineer ---
def engineer_node(state: MasterState):
    cleaning_plan_json = state.get('cleaning_plan', '')
    working_files = state.get('working_files', {})
    profile_json_str = state.get('deep_profile_report', '') # NOW INCLUDED!
    error_feedback = f"\n\nPrevious Code Error to Fix: {state.get('error')}" if state.get('error') else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Python Data Engineer.
        Write a Python script to clean the data based exactly on the provided JSON Plan.
        
        CRITICAL EXECUTION CONTEXT: 
        You have access to the Data Profile. Use it to ensure your Pandas syntax matches the actual column data types.
        I will provide a dictionary mapping filenames to their temporary pickle file paths.
        
        YOUR SCRIPT MUST EXACTLY DO THIS:
        1. Import pandas and numpy.
        2. Define the `working_files` dictionary exactly as provided.
        3. Loop through the dictionary, use `pd.read_pickle(path)` to load the dataframe.
        4. Apply the transformations specified in the plan.
        5. Use `df.to_pickle(path)` to save the dataframe back to the EXACT SAME PATH.
        
        RULES:
        - DO NOT create dummy data.
        - Use modern pandas. DO NOT use inplace=True.
        - If filling NaNs in a Category column, add the category first using `.cat.add_categories()`.
        - RETURN ONLY VALID PYTHON CODE inside ```python ... ``` blocks.{error_feedback}
        """),
        ("user", "Working Files Paths:\n{files}\n\nData Profile (Context):\n{profile}\n\nCleaning Plan (Actions):\n{plan}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "files": json.dumps(working_files, indent=2),
        "profile": profile_json_str,
        "plan": cleaning_plan_json,
        "error_feedback": error_feedback  # <--- This was missing!
    })
    
    raw = response.content
    match = re.search(r"```python(.*?)```", raw, re.DOTALL)
    code = match.group(1).strip() if match else raw.strip()
    
    return {"python_code": code}

def execute_clean_node(state: MasterState):
    code = state.get('python_code', '')
    result = repl_sandbox.run(code)
    
    if result["error"]:
        # Increment iteration count to trigger two-tier retry
        current_iter = state.get("iteration_count", 0) + 1
        return {"error": result["error"], "iteration_count": current_iter}
        
    return {"error": None, "iteration_count": 0}

# --- 6. Two-Tier Retry Routing ---
def route_clean_retry(state: MasterState):
    if state.get("error"):
        if state.get("iteration_count", 0) >= 2:
            # Fatal error: The engineer tried twice and failed. The plan is likely logically flawed.
            # Send the error ALL the way back to the strategist.
            return "strategist"
        # Syntax error: Send back to the engineer to fix its code.
        return "engineer"
    return END

# --- 7. Graph Assembly ---
def build_cleaning_graph():
    workflow = StateGraph(MasterState)
    
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("human_review", human_review_plan_node)
    workflow.add_node("engineer", engineer_node)
    workflow.add_node("execute", execute_clean_node)
    
    workflow.set_entry_point("profiler")
    workflow.add_edge("profiler", "strategist")
    
    # Internal Pydantic Validation loop
    workflow.add_conditional_edges("strategist", route_strategist, {"strategist": "strategist", "human_review": "human_review"})
    
    workflow.add_edge("human_review", "engineer")
    workflow.add_edge("engineer", "execute")
    
    # Two-Tier Execution loop
    workflow.add_conditional_edges("execute", route_clean_retry, {
        "engineer": "engineer", 
        "strategist": "strategist", 
        END: END
    })
    
    return workflow