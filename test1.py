# test_phase1.py
import pandas as pd
from ingestion.agent import build_ingestion_graph

if __name__ == "__main__":
    df = pd.DataFrame({'HNUM': ['A1', 'B2'], 'RESULT': [10, 20]})
    df.to_csv("dummy.csv", index=False)
    
    initial_state = {
        "file_paths": ["dummy.csv"],
        "user_input": "Rename the column 'HNUM' to 'ID'.",
        "working_files": {}, # <--- Changed dfs to working_files
        "messages": []
    }
    
    graph = build_ingestion_graph()
    
    print("--- RUNNING PHASE 1 TEST ---")
    from langgraph.checkpoint.memory import MemorySaver
    
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory, interrupt_before=["human_review"])
    
    config = {"configurable": {"thread_id": "test_1"}}
    
    for event in app.stream(initial_state, config=config):
        for node, state_update in event.items():
            print(f"Finished node: {node}")
            if "python_code" in state_update:
                print(f"Code Generated:\n{state_update['python_code']}")
                
    print("\nGraph execution paused naturally for human review.")