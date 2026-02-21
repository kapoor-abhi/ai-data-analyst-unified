# preprocessing/clean_agent.py
import os
import json
import re
import shutil
import pandas as pd
import numpy as np
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from core.state import MasterState
from langgraph.types import interrupt
from core.sandbox import DockerREPL
from dotenv import load_dotenv

# Added config import for the interrupt state preservation
from langchain_core.runnables.config import var_child_runnable_config, RunnableConfig

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class CleaningAction(BaseModel):
    target_file: str = Field(description="The exact filename, e.g., 'merged_dataset.pkl'")
    target_column: Optional[str] = Field(None, description="The exact column name. MUST NOT BE NULL unless the action is 'deduplicate_records' or 'drop_redundant_columns'.")
    groupby_column: Optional[str] = Field(None, description="MUST be provided for 'fill_missing_median'.")
    action_type: Literal[
        "replace_fake_nulls", "strip_whitespace", "extract_numeric_regex",
        "parse_dates_resiliently", "standardize_categories", "fill_missing_median", 
        "fill_missing_unknown", "cap_outliers", "convert_type", "deduplicate_records", "drop_redundant_columns", "custom"
    ] = Field(description="The specific cleaning operation.")
    code_instruction: str = Field(description="Plain English instruction.")

class CleaningPlan(BaseModel):
    actions: List[CleaningAction]
    # NEW: Force the LLM to explicitly acknowledge and handle these enterprise rules
    has_dropped_redundant_columns: bool = Field(description="Must be true. You must include a drop_redundant_columns action if merge keys overlap (e.g., client_id and user_id).")
    has_deduplicated: bool = Field(description="Must be true. You must include a deduplicate_records action to remove duplicate rows.")
    has_filled_unknowns: bool = Field(description="Must be true. Categorical columns with NaNs MUST get a fill_missing_unknown action.")

fast_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
coder_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
repl_sandbox = DockerREPL(sandbox_dir="sandbox", image_name="python-data-sandbox:latest")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if pd.isna(obj): return None
        return super(NpEncoder, self).default(obj)

def strip_markdown(text: str) -> str:
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return text.strip()

def analyze_column(series: pd.Series):
    clean_series = series.dropna()
    total_count = len(series)
    non_null_count = len(clean_series)
    
    report = {"logical_type": "Unknown", "null_count": int(total_count - non_null_count), "pct_missing": round((1 - (non_null_count / total_count)) * 100, 2) if total_count > 0 else 0, "issues": []}
    
    if non_null_count == 0:
        report["logical_type"] = "Empty"
        return report

    fake_null_placeholders = ['?', 'N/A', 'n/a', 'NA', 'None', 'null', 'nan', 'NaN', '', ' ', '-999', '-9999', '9999']
    fake_nulls_found = clean_series.astype(str).str.strip().isin(fake_null_placeholders).sum()
    if fake_nulls_found > 0:
        report["issues"].append(f"Contains {fake_nulls_found} Fake Nulls/Magic Numbers")

    col_name_lower = str(series.name).lower()
    if any(keyword in col_name_lower for keyword in ['id', 'code', 'link', 'uuid', 'key']):
        report["logical_type"] = "Text/ID"
        return report

    if pd.api.types.is_datetime64_any_dtype(series) or (clean_series.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').sum() > len(clean_series) * 0.5) or (clean_series.astype(str).str.match(r'^\d{2}/\d{2}/\d{4}').sum() > len(clean_series) * 0.5):
        report["logical_type"] = "Datetime"
        report["issues"].append("Requires resilient multi-format date parsing")
        return report

    numeric_converted = pd.to_numeric(clean_series, errors='coerce')
    num_valid_count = numeric_converted.notna().sum()
    
    if num_valid_count / non_null_count > 0.7:
        report["logical_type"] = "Numeric"
        if num_valid_count < non_null_count: report["issues"].append("Contains non-numeric characters")
        return report

    str_series = clean_series.astype(str)
    has_digits = str_series.str.contains(r'\d', regex=True)
    has_letters_or_currency = str_series.str.contains(r'[a-zA-Z$€£¥,]', regex=True)
    if (has_digits & has_letters_or_currency).sum() / non_null_count > 0.5:
        report["logical_type"] = "Mixed Units/Currency"
        report["issues"].append("Requires Regex Extraction to become numeric")
        return report

    unique_count = clean_series.nunique()
    if unique_count < 50 or (unique_count / total_count < 0.2):
        report["logical_type"] = "Categorical"
        if str_series.str.contains(r'^\s|\s$', regex=True).any(): report["issues"].append("Contains Leading/Trailing Whitespace")
        return report

    report["logical_type"] = "Text/ID"
    return report

def profiler_node(state: MasterState):
    working_files = state.get('working_files', {})
    full_report = {}
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            sample_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df
            file_report = {"rows": len(df), "cols": len(df.columns), "columns": {}}
            for col in sample_df.columns: file_report["columns"][col] = analyze_column(sample_df[col])
            full_report[name] = file_report
        except Exception as e: pass
    return {"deep_profile_report": json.dumps(full_report, cls=NpEncoder, indent=2), "error": None, "iteration_count": 0}

def strategist_node(state: MasterState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Principal Enterprise Data Engineer. Generate a JSON cleaning plan based on the data profile.
        
        CRITICAL ENTERPRISE RULES:
        1. target_column MUST NOT BE NULL unless the action is deduplicating records.
        2. NEVER suggest dropping rows (`dropna`). You must use imputation or fill actions.
        3. IMPUTATION RULE: If using `fill_missing_median`, the `groupby_column` MUST be a low-cardinality category (e.g., 'priority', 'status', 'warehouse'). NEVER use a unique identifier or Primary Key (like 'client_id' or 'user_id') because the median of a single NaN is NaN.
        4. If a column is "Mixed Units/Currency", instruct to `extract_numeric_regex`.
        5. CRITICAL: Look for redundant ID columns generated by merges (e.g., 'client_id' and 'user_id' containing the same data). Instruct to drop one of them using action 'drop_redundant_columns'. Set has_dropped_redundant_columns to true.
        6. CRITICAL: If a column's profile issues mention "Fake Nulls/Magic Numbers", you MUST include a 'replace_fake_nulls' action for that specific column.
        7. CRITICAL: For categorical columns that have nulls, you MUST include a 'fill_missing_unknown' action. Set has_filled_unknowns to true.
        8. CRITICAL: You MUST include a 'deduplicate_records' action for the dataset. Set has_deduplicated to true.
        9. Return ONLY valid JSON matching the schema below.
        
        SCHEMA:
        {schema}
        """),
        ("user", "DATA PROFILE:\n{profile}\n\nPREVIOUS ERROR:\n{error}")
    ])
    chain = prompt | fast_llm
    response = chain.invoke({"schema": json.dumps(CleaningPlan.model_json_schema(), indent=2), "profile": state.get('deep_profile_report', ''), "error": state.get('error', '')})
    
    try:
        plan_obj = CleaningPlan.model_validate_json(strip_markdown(response.content))
        return {"cleaning_plan": plan_obj.model_dump_json(indent=2), "error": None, "iteration_count": 0}
    except ValidationError as e:
        # FIX: Track validation iterations to prevent infinite loops
        current_iter = state.get("iteration_count", 0) + 1
        return {"error": f"Pydantic Validation Error: {e.errors()}", "iteration_count": current_iter}

def route_strategist(state: MasterState):
    if state.get("error") and "Pydantic" in state.get("error", ""): 
        # FIX: Break out of infinite JSON validation loops and ask the human
        if state.get("iteration_count", 0) >= 3:
            return "human_review"
        return "strategist"
    return "human_review"

async def human_review_plan_node(state: MasterState, config: RunnableConfig):
    var_child_runnable_config.set(config)
    
    plan = state.get('cleaning_plan', '')
    error = state.get('error')
    msg = f"Execution failed after multiple attempts:\n{error}\n\nPlease revise instructions or approve plan:\n{plan}" if error else plan
    
    feedback = interrupt(msg)
    return {"user_feedback": feedback, "error": None, "iteration_count": 0}

def engineer_node(state: MasterState):
    # FIX: Null safety fallback using `or ''` 
    human_instruction = state.get('user_feedback') or ''
    if human_instruction.strip().lower() in ['approve', 'enter', 'yes', '']:
        human_instruction = "Proceed with the JSON plan exactly as written. No overrides."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Python Data Engineer. Write a Python script to clean the data.
        
        CRITICAL EXECUTION CONTEXT & RULES: 
        1. I will provide a dictionary mapping filenames to their temporary paths (`working_files`).
        2. Loop through the dictionary, use `pd.read_pickle(path)` to load the dataframe.
        3. Apply transformations according to the human instructions and plan.
        4. **CRITICAL: You MUST use `df.to_pickle(path)` to save the dataframe back to the EXACT SAME PATH when finished.** Do not use `.to_csv()`.
        5. DO NOT WRITE HELPER FUNCTIONS OR WRAPPER FUNCTIONS. Write flat, procedural code directly inside the file loop.
        6. NEVER drop rows using dropna().
        
        BULLETPROOF CODE SNIPPETS (USE EXACTLY AS WRITTEN):
        - `replace_fake_nulls`: `df[col] = df[col].replace(['?', 'N/A', 'n/a', 'NA', 'None', 'null', 'nan', 'NaN', '', ' ', '-999', '-9999', '9999', -999, -9999, 9999], np.nan)` (DO NOT use regex stripping before running this).
        - `standardize_categories`: `df[col] = df[col].astype(str).str.lower().str.strip().replace('nan', np.nan)` (Do NOT write your own regex filters for this).
        - `extract_numeric_regex`: `df[col] = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True); df[col] = pd.to_numeric(df[col], errors='coerce')`
        - `parse_dates_resiliently`: `df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')`
        - `groupby imputation`: `df[col] = df[col].fillna(df.groupby(groupby_col, dropna=False)[col].transform('median'))`
        - `drop_redundant_columns`: `df = df.drop(columns=[col])`
        - `deduplicate_records`: `df = df.drop_duplicates()`
        - `fill_missing_unknown`: `df[col] = df[col].fillna('Unknown')`
        
        HUMAN OVERRIDE INSTRUCTIONS:
        {human}
        
        RETURN ONLY VALID PYTHON CODE inside ```python ... ``` blocks.{error_feedback}
        """),
        ("user", "Files:\n{files}\n\nProfile:\n{profile}\n\nPlan:\n{plan}")
    ])
    chain = prompt | coder_llm
    response = chain.invoke({
        "human": human_instruction, "files": json.dumps(state.get('working_files', {}), indent=2),
        "profile": state.get('deep_profile_report') or '', "plan": state.get('cleaning_plan') or '',
        "error_feedback": f"\n\nPrevious Code Error to Fix: {state.get('error')}" if state.get('error') else ""
    })
    
    raw = response.content
    match = re.search(r"```python(.*?)```", raw, re.DOTALL)
    return {"python_code": match.group(1).strip() if match else raw.strip()}

def execute_clean_node(state: MasterState):
    working_files = state.get('working_files', {})
    for name, path in working_files.items():
        if os.path.exists(path): shutil.copy(path, path + ".bak")

    result = repl_sandbox.run(state.get('python_code', ''))
    
    if result["error"]: 
        for name, path in working_files.items():
            if os.path.exists(path + ".bak"): shutil.move(path + ".bak", path)
        return {"error": result["error"], "iteration_count": state.get("iteration_count", 0) + 1}
        
    return {"error": None}

def verify_cleaning_node(state: MasterState):
    # FIX: Return empty dict instead of state. Returning the full state causes the `add_messages` reducer to duplicate the entire chat history.
    if state.get("error"): return {} 
    
    working_files = state.get('working_files', {})
    
    # FIX: Null safety fallback for json.loads to prevent TypeError crashes
    original_profile = json.loads(state.get('deep_profile_report') or '{}')
    
    # FIX: Null safety fallback to prevent "in None" TypeError
    plan_str = state.get('cleaning_plan') or ''
    is_dedup_intended = "deduplicate_records" in plan_str
    
    error_msgs = []
    
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            orig_rows = original_profile.get(name, {}).get("rows", len(df))
            
            if len(df) < orig_rows and not is_dedup_intended:
                error_msgs.append(f"CRITICAL AUDIT FAILURE: Your code dropped {orig_rows - len(df)} rows. You are strictly forbidden from dropping rows. Fix the code to impute or ignore, but do not drop.")
        except Exception as e: error_msgs.append(f"Audit failed: {e}")
            
    if error_msgs: 
        for name, path in working_files.items():
            if os.path.exists(path + ".bak"): shutil.move(path + ".bak", path)
        return {"error": " | ".join(error_msgs), "iteration_count": state.get("iteration_count", 0) + 1}
        
    for name, path in working_files.items():
        if os.path.exists(path + ".bak"): os.remove(path + ".bak")
            
    return {"error": None, "iteration_count": 0}

async def post_clean_review_node(state: MasterState, config: RunnableConfig):
    var_child_runnable_config.set(config)
    
    msg = "Iteration executed successfully. Review the updated statistics. Type 'approve' to finalize, or provide further cleaning instructions."
    feedback = interrupt(msg)
    
    return {"user_feedback": feedback, "error": None, "iteration_count": 0}

def route_clean_retry(state: MasterState):
    if state.get("error"):
        if state.get("iteration_count", 0) >= 3: return "human_review"
        return "engineer"
    return "post_clean_review"

def route_post_clean(state: MasterState):
    # Null safety check added here as well
    feedback = str(state.get("user_feedback") or "").strip().lower()
    if feedback in ['approve', 'yes', 'ok', 'done', 'finish', '']:
        return END
    return "engineer"

def build_cleaning_graph():
    workflow = StateGraph(MasterState)
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("human_review", human_review_plan_node)
    workflow.add_node("engineer", engineer_node)
    workflow.add_node("execute", execute_clean_node)
    workflow.add_node("verify", verify_cleaning_node) 
    
    workflow.add_node("post_clean_review", post_clean_review_node)
    
    workflow.set_entry_point("profiler")
    workflow.add_edge("profiler", "strategist")
    workflow.add_conditional_edges("strategist", route_strategist, {"strategist": "strategist", "human_review": "human_review"})
    workflow.add_edge("human_review", "engineer")
    workflow.add_edge("engineer", "execute")
    workflow.add_edge("execute", "verify")
    
    workflow.add_conditional_edges("verify", route_clean_retry, {
        "engineer": "engineer", 
        "human_review": "human_review", 
        "post_clean_review": "post_clean_review"
    })
    
    workflow.add_conditional_edges("post_clean_review", route_post_clean, {
        "engineer": "engineer", 
        END: END
    })
    
    return workflow