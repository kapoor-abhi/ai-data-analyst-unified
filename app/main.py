# app/main.py
import os
import uuid
import shutil
import logging
import hashlib
import pandas as pd
import numpy as np 
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
import math
from fastapi.responses import Response

# Enterprise ML Imports
import psycopg
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from redis import Redis
from langgraph.types import Command

# Langfuse Observability Imports
from langfuse import observe
from langfuse.langchain import CallbackHandler

# Import our Super-Graph
from core.super_agent import build_super_graph

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URI = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/postgres")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# 1. Initialize Langfuse Observability
langfuse_handler = CallbackHandler()

# 2. CACHE FIX: Initialize Exact-Match Redis Cache (Not Semantic!)
try:
    redis_client = Redis.from_url(REDIS_URL)
    set_llm_cache(RedisCache(redis_=redis_client))
    logger.info("Exact-Match Redis Cache activated.")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}. Running without cache.")

SANDBOX_DIR = "sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)

# Simple local registry for file hashes to prevent redundant processing
HASH_REGISTRY = {}

def get_file_hash(filepath: str) -> str:
    """Generates a SHA-256 hash of a file to check if we've processed it before."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles PostgreSQL DB migrations and connection pooling for LangGraph Memory."""
    try:
        async with await psycopg.AsyncConnection.connect(DB_URI, autocommit=True) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
        logger.info("PostgreSQL memory tables ready.")
    except Exception as e:
        logger.error(f"PostgreSQL failed: {e}")
        
    async with AsyncConnectionPool(conninfo=DB_URI, max_size=20) as pool:
        app.state.pool = pool
        yield
        
    # Flush Langfuse telemetry on shutdown to prevent dropped traces
    langfuse_handler.flush()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

super_graph = build_super_graph()

def extract_deepest_state_and_interrupt(state_snapshot):
    """Recursively extracts the interrupt message and merges state from nested sub-graphs."""
    msg = ""
    values = getattr(state_snapshot, 'values', {})
    
    if hasattr(state_snapshot, 'tasks') and state_snapshot.tasks:
        for task in state_snapshot.tasks:
            
            # 1. DIVE DEEP FIRST: Always recurse into sub-graphs to get the freshest data
            if hasattr(task, 'state') and task.state:
                sub_msg, sub_values = extract_deepest_state_and_interrupt(task.state)
                # Merge states: sub-graph data (like cleaning_plan) updates the parent data
                values = {**values, **sub_values}
                if sub_msg:
                    msg = sub_msg
            
            # 2. CHECK INTERRUPTS: Grab the interrupt if we haven't found one deeper down
            if not msg and hasattr(task, 'interrupts') and task.interrupts:
                msg = str(task.interrupts[0].value)
                
    return msg, values

@app.post("/upload")
async def upload_file(
    thread_id: str = Form(...), 
    user_input: str = Form(""), 
    files: list[UploadFile] = File(...)
):
    """Uploads files, checks hashes, and triggers the ETL pipeline until the first interrupt."""
    try:
        file_paths = []
        for file in files:
            temp_id = str(uuid.uuid4())[:8]
            temp_name = f"temp_{temp_id}_{file.filename}"
            full_path = os.path.join(SANDBOX_DIR, temp_name)
            
            with open(full_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            file_hash = get_file_hash(full_path)
            
            if file_hash in HASH_REGISTRY:
                logger.info(f"File {file.filename} already processed. Utilizing cache.")
                os.remove(full_path) 
                file_paths.append(HASH_REGISTRY[file_hash])
            else:
                final_path = os.path.join(SANDBOX_DIR, f"{temp_id}_{file.filename}")
                os.rename(full_path, final_path)
                HASH_REGISTRY[file_hash] = final_path
                file_paths.append(final_path)
            
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            app_compiled = super_graph.compile(checkpointer=checkpointer)
            
            # ATTACH LANGFUSE CALLBACK TO CONFIG
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [langfuse_handler]
            }
            inputs = {"file_paths": file_paths, "messages": [], "user_input": user_input} 
            
            output = await app_compiled.ainvoke(inputs, config)
            
            # Fetch the state snapshot and explicitly ask LangGraph to include sub-graphs
            state_snapshot = await app_compiled.aget_state(config, subgraphs=True)
            
            if state_snapshot.next:
                # Use our recursive function to pull the nested state and interrupt
                msg, combined_values = extract_deepest_state_and_interrupt(state_snapshot)
                return {
                    "status": "paused", 
                    "interrupt_msg": msg,
                    "pending_state": combined_values
                }
            
            return {"status": "success", "message": "Pipeline completed without interrupts.", "working_files": output.get("working_files")}
            
    except Exception as e:
        logger.error("Upload error encountered:", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/resume")
async def resume_pipeline(
    thread_id: str = Form(...), 
    user_feedback: str = Form(None)
):
    """HITL ENDPOINT: Injects human feedback into the state and unpauses the graph."""
    try:
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            app_compiled = super_graph.compile(checkpointer=checkpointer)
            
            # ATTACH LANGFUSE CALLBACK TO CONFIG
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [langfuse_handler]
            }
            
            # Include subgraphs=True here to accurately verify if the nested graph is paused
            state_snapshot = await app_compiled.aget_state(config, subgraphs=True)
            if not state_snapshot.next:
                raise HTTPException(status_code=400, detail="Graph is not currently paused.")
                
            output = await app_compiled.ainvoke(Command(resume=user_feedback), config)
            
            # Include subgraphs=True here to fetch the newly paused state
            new_snapshot = await app_compiled.aget_state(config, subgraphs=True)
            
            if new_snapshot.next:
                # Use our recursive function
                msg, combined_values = extract_deepest_state_and_interrupt(new_snapshot)
                return {
                    "status": "paused", 
                    "interrupt_msg": msg,
                    "pending_state": combined_values
                }
                
            return {"status": "success", "message": "ETL Pipeline complete. Ready for Chat.", "working_files": output.get("working_files")}
            
    except Exception as e:
        logger.error(f"Resume error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat(message: str = Form(...), thread_id: str = Form(...)):
    """Handles user questions and routes to DuckDB or Visualization."""
    try:
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            app_compiled = super_graph.compile(checkpointer=checkpointer)
            
            # ATTACH LANGFUSE CALLBACK TO CONFIG
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [langfuse_handler]
            }
            
            # ✅ THE MEMORY FIX: Append to LangGraph's message history AND pass user_input
            inputs = {
                "messages": [("user", message)], 
                "user_input": message
            }
            
            output = await app_compiled.ainvoke(inputs, config)
            last_msg = output["messages"][-1].content
            charts = output.get("charts_generated", [])
            latest_chart = charts[-1] if charts else None

            return {"response": last_msg, "plot_url": f"/{latest_chart}" if latest_chart else None}
            
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
            
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/sandbox/{plot_name}")
async def get_plot(plot_name: str):
    full_path = os.path.join(SANDBOX_DIR, plot_name)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    return JSONResponse(status_code=404, content={"error": "Plot not found"})

@app.get("/download")
async def download_cleaned_data(filename: str = "merged_dataset.pkl"):
    file_path = os.path.join(SANDBOX_DIR, filename)
    if not os.path.exists(file_path):
        pkl_files = [f for f in os.listdir(SANDBOX_DIR) if f.endswith('.pkl')]
        if not pkl_files:
            return JSONResponse(status_code=404, content={"error": "No cleaned dataset found in sandbox."})
        file_path = os.path.join(SANDBOX_DIR, pkl_files[0])

    try:
        df = pd.read_pickle(file_path)
        csv_filename = file_path.replace(".pkl", ".csv")
        df.to_csv(csv_filename, index=False)
        return FileResponse(path=csv_filename, filename="cleaned_data_output.csv", media_type="text/csv")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate CSV: {str(e)}"})

@app.get("/statistics")
async def get_data_statistics(filename: str = None):
    """Aggregates statistics for ALL active datasets in the sandbox."""
    pkl_files = [f for f in os.listdir(SANDBOX_DIR) if f.endswith('.pkl')]
    
    if not pkl_files:
        return JSONResponse(status_code=404, content={"error": "No data found in sandbox."})
        
    # CRITICAL: If the merging agent has unified the data, prioritize that single file!
    if "merged_dataset.pkl" in pkl_files:
        pkl_files = ["merged_dataset.pkl"]

    total_rows = 0
    total_cols = 0
    all_columns_info = {}
    raw_sample_data = []

    try:
        for file in pkl_files:
            file_path = os.path.join(SANDBOX_DIR, file)
            df = pd.read_pickle(file_path)
            
            total_rows += len(df)
            total_cols += len(df.columns)
            
            stats_df = df.describe(include='all')
            stats_dict = stats_df.to_dict()
            
            # 1. Prefix columns with filename if there are multiple files
            is_multi_file = len(pkl_files) > 1
            
            # 2. Extract sample rows and append the filename to the keys
            head_records = df.head(5).to_dict(orient="records")
            for record in head_records:
                if is_multi_file:
                    raw_sample_data.append({f"[{file}] {k}": v for k, v in record.items()})
                else:
                    raw_sample_data.append(record)

            # 3. Aggregate column profiling stats
            for col in df.columns:
                display_col = f"[{file}] {col}" if is_multi_file else col
                
                all_columns_info[display_col] = {
                    "dtype": str(df[col].dtype),
                    "missing_values": int(df[col].isna().sum()),
                    "unique_values": int(df[col].nunique())
                }
                if col in stats_dict:
                    all_columns_info[display_col].update(stats_dict[col])

        # 4. Pad the sample data with nulls so the frontend table doesn't break when columns mismatch
        final_sample_data = []
        if len(pkl_files) > 1 and raw_sample_data:
            all_keys = list(all_columns_info.keys())
            for record in raw_sample_data:
                padded_record = {}
                for k in all_keys:
                    padded_record[k] = record.get(k, None)
                final_sample_data.append(padded_record)
        else:
            final_sample_data = raw_sample_data

        payload = {
            "total_rows": total_rows, 
            "total_columns": total_cols, 
            "columns": all_columns_info, 
            "sample_data": final_sample_data[:15] # Send top 15 rows total
        }

        def clean_value(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
            if pd.api.types.is_scalar(obj) and pd.isna(obj): return None
            if isinstance(obj, (pd.Timestamp, pd.Series, pd.Index)): return str(obj)
            return obj

        def deep_clean(d):
            if isinstance(d, dict): return {k: deep_clean(v) for k, v in d.items()}
            elif isinstance(d, list): return [deep_clean(v) for v in d]
            else: return clean_value(d)

        return Response(content=json.dumps(deep_clean(payload)), media_type="application/json")
    except Exception as e:
        logger.error(f"Critical Stats Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    

from ydata_profiling import ProfileReport

@app.get("/generate-eda")
async def generate_eda_report():
    """Generates a deep univariate/bivariate EDA report on the cleaned dataset."""
    filename = "merged_dataset.pkl"
    file_path = os.path.join(SANDBOX_DIR, filename)
    report_path = os.path.join(SANDBOX_DIR, "eda_report.html")
    
    # If a previous report already exists, serve it instantly
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="text/html")
        
    # If no merged dataset exists yet, fallback to looking for any .pkl
    if not os.path.exists(file_path):
        pkl_files = [f for f in os.listdir(SANDBOX_DIR) if f.endswith('.pkl')]
        if not pkl_files:
            return JSONResponse(status_code=404, content={"error": "No cleaned data found to profile."})
        file_path = os.path.join(SANDBOX_DIR, pkl_files[0])

    try:
        # Load the cleaned data
        df = pd.read_pickle(file_path)
        
        # Generate the report (minimal=True speeds it up for large datasets)
        profile = ProfileReport(df, title="Deep Data Profiling Report", minimal=False, explorative=True)
        profile.to_file(report_path)
        
        return FileResponse(report_path, media_type="text/html")
        
    except Exception as e:
        logger.error(f"EDA Generation Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to generate EDA: {str(e)}"})