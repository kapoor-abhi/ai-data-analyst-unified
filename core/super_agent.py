# core/super_agent.py
from langgraph.graph import StateGraph, START, END
from core.state import MasterState

from ingestion.agent import build_ingestion_graph
from merging.merge_agent import build_merge_graph
from preprocessing.clean_agent import build_cleaning_graph
from chat.chat_agent import build_chat_graph

def entry_router(state: MasterState):
    working_files = state.get("working_files", {})
    user_input = state.get("user_input", "")
    
    # 1. If we have a user question/input and data is already cleaned, jump to chat
    if user_input and working_files and state.get("deep_profile_report"):
        return "chat"
    
    # 2. New Upload -> Start the ETL Pipeline
    if state.get("file_paths") and not working_files:
        return "ingestion"
    
    # 3. Data exists but not cleaned -> Resume the ETL
    if working_files and not state.get("deep_profile_report"):
        if len(working_files) > 1:
            return "merging"
        return "cleaning"
        
    return "chat"

def route_after_ingestion(state: MasterState):
    if state.get("error"): return END
    working_files = state.get("working_files", {})
    if len(working_files) > 1: return "merging"
    return "cleaning"

def build_super_graph():
    workflow = StateGraph(MasterState)
    
    # Compile WITHOUT interrupts to make it 100% Autonomous
    workflow.add_node("ingestion", build_ingestion_graph().compile())
    workflow.add_node("merging", build_merge_graph().compile())
    workflow.add_node("cleaning", build_cleaning_graph().compile())
    workflow.add_node("chat", build_chat_graph().compile())
    
    workflow.add_conditional_edges(START, entry_router, {
        "ingestion": "ingestion",
        "merging": "merging",
        "cleaning": "cleaning",
        "chat": "chat"
    })
    
    workflow.add_conditional_edges("ingestion", route_after_ingestion, {
        "merging": "merging",
        "cleaning": "cleaning",
        END: END
    })
    
    workflow.add_edge("merging", "cleaning")
    # HALT THE ETL PIPELINE HERE. Wait for the user to chat.
    workflow.add_edge("cleaning", END) 
    workflow.add_edge("chat", END)
    
    return workflow