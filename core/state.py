# core/state.py
from typing import Annotated, TypedDict, Sequence, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MasterState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    file_paths: List[str]
    working_files: Dict[str, str]  # filename -> path to pickle file
    user_input: str
    error: Optional[str]
    
    is_large_dataset: bool
    python_code: Optional[str]
    
    suggestion: Optional[str]
    user_feedback: Optional[str]
    
    deep_profile_report: Optional[str]
    cleaning_plan: Optional[str]
    
    df_info: Optional[str]
    analysis_plan: Optional[str]
    charts_generated: List[str]
    iteration_count: int
    next_step: Optional[str]