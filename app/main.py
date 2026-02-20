# app/main.py
import os
import uuid
import shutil
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
# Enterprise ML Imports
import psycopg
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langfuse.langchain import CallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Import our Super-Graph
from core.super_agent import build_super_graph

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URI = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/postgres")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# 1. Initialize Langfuse Observability
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# 2. Initialize Redis Semantic Cache (Open Source Embeddings)
try:
    set_llm_cache(RedisSemanticCache(
        redis_url=REDIS_URL,
        embedding=FastEmbedEmbeddings()
    ))
    logger.info("Redis Semantic Cache activated.")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}. Running without cache.")

SANDBOX_DIR = "sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)

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

app = FastAPI(lifespan=lifespan)
super_graph = build_super_graph()

@app.post("/upload")
async def upload_file(thread_id: str = Form(...), files: list[UploadFile] = File(...)):
    """Uploads files and triggers the Ingestion -> Merge -> Clean pipeline."""
    try:
        file_paths = []
        for file in files:
            file_id = str(uuid.uuid4())[:8]
            file_name = f"{file_id}_{file.filename}"
            full_path = os.path.join(SANDBOX_DIR, file_name)
            
            with open(full_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(full_path)
            
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            app_compiled = super_graph.compile(checkpointer=checkpointer)
            
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [langfuse_handler] # Injects Observability
            }
            
            inputs = {"file_paths": file_paths, "messages": []}
            
            # Run the graph. It will process ingestion/cleaning and pause.
            output = await app_compiled.ainvoke(inputs, config)
            
            return {"status": "success", "message": "Files ingested and cleaned.", "working_files": output.get("working_files")}
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat(message: str = Form(...), thread_id: str = Form(...)):
    """Handles user questions and routes to DuckDB or Visualization."""
    try:
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            app_compiled = super_graph.compile(checkpointer=checkpointer)
            
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [langfuse_handler]
            }
            
            inputs = {"user_input": message}
            
            # The graph will automatically start at 'chat' because working_files already exist in the PostgreSQL state
            output = await app_compiled.ainvoke(inputs, config)
            
            last_msg = output["messages"][-1].content
            
            # Check if the visualization node saved a chart
            charts = output.get("charts_generated", [])
            latest_chart = charts[-1] if charts else None

            return {
                "response": last_msg, 
                "plot_url": f"/{latest_chart}" if latest_chart else None
            }
            
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/sandbox/{plot_name}")
async def get_plot(plot_name: str):
    full_path = os.path.join(SANDBOX_DIR, plot_name)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    return JSONResponse(status_code=404, content={"error": "Plot not found"})