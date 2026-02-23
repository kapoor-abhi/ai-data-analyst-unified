# Autonomous AI Data Analyst 

## System Overview

This project implements an enterprise-grade, autonomous Data Analytics and ETL (Extract, Transform, Load) engine powered by Large Language Models (LLMs) and structured as a Directed Acyclic Graph (DAG). The system ingests raw datasets, autonomously profiles the data, proposes and executes complex cleaning and merging strategies, and provides a conversational interface for SQL-based querying and Python-based visualization.

The architecture emphasizes execution security, observability, and robust state management to support Human-in-the-Loop (HITL) interactions for critical data pipeline approvals.

## Architecture & System Design

The system relies on a multi-layered architecture orchestrating specialized AI agents, secure execution environments, and stateful memory check-pointing.

High-Level Architecture Diagram
<p align="center">
  <img src="screenshots/architecture.png" width="900"/>
</p>

### 1. Agentic Workflow (Super-Graph Architecture)

The core logic is orchestrated using LangGraph, designed as a "Super-Graph" containing several independently compiled sub-graphs. This encapsulates state and logic for distinct phases of the pipeline:

* **Ingestion Agent**: Loads datasets, detects dataset scale, applies vectorized Pandas optimization for large datasets, and parses initial schemas.
* **Merging Agent**: Analyzes overlapping schemas across multiple files and intelligently generates `pandas.merge` code (defaulting to outer joins to prevent data loss) based on schema similarity.
* **Cleaning Agent**: Features a multi-node pipeline consisting of a Profiler, Strategist, and Engineer. It utilizes Pydantic structured outputs to generate deterministic cleaning plans (e.g., handling fake nulls, deduplication, regex extraction, and imputation) and writes procedural Python code to apply these transformations.
* **Chat Agent**: Utilizes a semantic router to classify user intents. It routes queries to an expert DuckDB SQL developer node for numerical analysis or a Python visualization node for generating charts via Matplotlib/Seaborn.

### 2. State Management & Checkpointing

To support Human-in-the-Loop (HITL) workflows, the pipeline must pause execution to await human approval or modification of the generated ETL logic.

* **PostgreSQL Checkpointing**: The system uses `AsyncPostgresSaver` to persist the entire graph state at every node transition.
* **Nested State Extraction**: Due to the nested sub-graph architecture, interrupt signals and deeply nested state variables (like the generated JSON cleaning plan) are recursively extracted and flattened, allowing the backend to seamlessly resume the DAG from the exact point of interruption across stateless HTTP requests.

### 3. Secure Execution Sandbox (Docker REPL)

Executing LLM-generated code natively introduces severe security and stability risks.

* **Isolated Containerization**: Generated Python code is never executed in the main application thread. Instead, it is routed to a custom `DockerREPL` interface utilizing native subprocesses with strict timeout limits to prevent infinite loops.
* **File System Isolation**: The sandbox operates within a dedicated volume, ensuring that only temporary data artifacts are manipulated, preserving the integrity of the host system.

### 4. Caching & Cost Optimization

* **Exact-Match Redis Cache**: An upstream Redis cache intercepts identical prompts to the LLM (common during iterative retry loops for code execution failures), bypassing the API call entirely to reduce latency and API costs.

## Technical Complexity & Key Solutions

* **Sliding Window Memory Management**: To prevent context window overflow and infinite LLM recursion during the chat phase, the system implements a strict slicing mechanism. It injects only the last 5 conversational turns (as native `HumanMessage` and `AIMessage` objects) into the prompt template using `MessagesPlaceholder`, maintaining context without token bloat.
* **Automated Error Recovery**: Code execution nodes are wrapped in validation loops. If the Docker sandbox throws an exception (e.g., a Pandas `TypeError`), the error traceback is truncated to save context space and fed back to the LLM for self-correction. The loop has a hard limit, routing to human intervention after consecutive failures.
* **Pandas Type-Casting Resiliency**: The system proactively forces standardization on categorical columns prior to logic execution, preventing silent failures caused by Pandas string manipulation quirks (e.g., `.astype(str)` converting `np.nan` to the literal string `"nan"`).

## Performance & Latency Metrics

The system is heavily optimized for speed, utilizing Groq's LPU inference engine combined with asynchronous backend processing. Distributed tracing via Langfuse demonstrates highly competitive latencies for a multi-agent system:

* **Overall Graph Execution (LangGraph Trace)**:
* p50 Latency: 2.478s
* p90 Latency: 3.405s


* **LLM Generation Latency (ChatGroq)**:
* p50 Latency: 1.765s
* p90 Latency: 3.152s


* **Specialized Node Execution**:
* Intent Routing (Llama-3.1-8b): ~0.473s (p50)
* SQL Query Generation: ~1.375s (p50)
* Code Generation & Execution (Engineer Node): ~3.099s (p50)



## Tech Stack

* **Backend Framework**: FastAPI (Asynchronous)
* **AI/Orchestration**: LangChain, LangGraph, Groq API (Llama-3.3-70b-versatile, Llama-3.1-8b-instant)
* **Data Processing**: Pandas, NumPy, DuckDB (In-memory SQL analytics)
* **Infrastructure**: Docker, Docker Compose
* **Databases**: PostgreSQL (Graph State), Redis (LLM Caching)
* **Observability**: Langfuse

Langgraph Graph
<p align="center">
  <img src="screenshots/langgraph_graph.png" width="200"/>
</p>