# core/super_agent.py

from langgraph.graph import StateGraph, START, END
from core.state import MasterState

from ingestion.agent import build_ingestion_graph
from merging.merge_agent import build_merge_graph
from preprocessing.clean_agent import build_cleaning_graph
from chat.chat_agent import build_chat_graph


# =====================================================
# ROUTER
# =====================================================

def entry_router(state: MasterState):

    working_files = state.get("working_files", {})
    user_input = state.get("user_input", "")

    # Chat if data already prepared
    if user_input and working_files and state.get("deep_profile_report"):
        return "chat"

    # New upload
    if state.get("file_paths") and not working_files:
        return "ingestion"

    # Data exists but not cleaned
    if working_files and not state.get("deep_profile_report"):

        if len(working_files) > 1:
            return "merging"

        return "cleaning"

    return "chat"


# =====================================================
# POST INGEST ROUTER
# =====================================================

def route_after_ingestion(state: MasterState):

    if state.get("error"):
        return END

    working_files = state.get("working_files", {})

    if len(working_files) > 1:
        return "merging"

    return "cleaning"


# =====================================================
# BUILD GRAPH
# =====================================================

def build_super_graph():

    workflow = StateGraph(MasterState)

    # Build and compile subgraphs (required in your version)
    ingestion_graph = build_ingestion_graph().compile()
    merge_graph = build_merge_graph().compile()
    cleaning_graph = build_cleaning_graph().compile()
    chat_graph = build_chat_graph().compile()

    # Add compiled subgraphs
    workflow.add_node("ingestion", ingestion_graph)
    workflow.add_node("merging", merge_graph)
    workflow.add_node("cleaning", cleaning_graph)
    workflow.add_node("chat", chat_graph)

    # Entry routing
    workflow.add_conditional_edges(
        START,
        entry_router,
        {
            "ingestion": "ingestion",
            "merging": "merging",
            "cleaning": "cleaning",
            "chat": "chat"
        }
    )

    # After ingestion
    workflow.add_conditional_edges(
        "ingestion",
        route_after_ingestion,
        {
            "merging": "merging",
            "cleaning": "cleaning",
            END: END
        }
    )

    # Linear ETL
    workflow.add_edge("merging", "cleaning")

    workflow.add_edge("cleaning", END)
    workflow.add_edge("chat", END)

    # IMPORTANT: Do NOT compile here
    return workflow