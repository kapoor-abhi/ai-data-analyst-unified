# test_phase2.py
import os
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from merging.merge_agent import build_merge_graph

if __name__ == "__main__":
    print("\n--- 1. SETTING UP DUMMY DATA ---")
    os.makedirs("sandbox", exist_ok=True)
    
    # Create two different datasets that share an 'ID' column
    df1 = pd.DataFrame({'ID': [101, 102, 103], 'Name': ['Alice', 'Bob', 'Charlie']})
    df2 = pd.DataFrame({'ID': [101, 102, 103], 'Department': ['HR', 'Engineering', 'Sales']})
    
    # Simulate Phase 1: Saving them as high-speed Pickles
    df1.to_pickle("sandbox/employees.pkl")
    df2.to_pickle("sandbox/departments.pkl")
    
    initial_state = {
        "working_files": {
            "employees.csv": "sandbox/employees.pkl", 
            "departments.csv": "sandbox/departments.pkl"
        },
        "messages": []
    }
    
    print("\n--- 2. RUNNING MERGE GRAPH ---")
    graph = build_merge_graph()
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory, interrupt_before=["human_strategy"])
    config = {"configurable": {"thread_id": "test_merge_1"}}
    
    # Run the graph until it hits the interrupt (Analyze Node)
    for event in app.stream(initial_state, config=config):
        for node, state_update in event.items():
            print(f"Finished node: {node}")
            if "suggestion" in state_update:
                print(f"LLM Strategy Suggestion: {state_update['suggestion']}")

    print("\nGraph paused naturally for human strategy confirmation.")
    print("Simulating User clicking 'Accept Suggestion'...\n")
    
    # Resume the graph by passing None (this tells LangGraph to continue with existing state)
    for event in app.stream(None, config=config):
        for node, state_update in event.items():
            print(f"Finished node: {node}")
            if "python_code" in state_update:
                print(f"Merge Code Generated:\n{state_update['python_code']}")

    print("\n--- 3. EXPORTING FINAL CSV ---")
    # Fetch the final state of the graph
    final_state = app.get_state(config).values
    working_files = final_state.get("working_files", {})
    
    merged_pkl_path = working_files.get("merged_dataset.pkl")
    
    if merged_pkl_path and os.path.exists(merged_pkl_path):
        # Read the merged pickle file and export it as a CSV
        final_df = pd.read_pickle(merged_pkl_path)
        csv_output_path = "sandbox/FINAL_MERGED_OUTPUT.csv"
        final_df.to_csv(csv_output_path, index=False)
        
        print(f"SUCCESS! The datasets were merged.")
        print(f"You can view your file here: {csv_output_path}")
        print("\nPreview of FINAL_MERGED_OUTPUT.csv:")
        print(final_df)
    else:
        print("Error: Merged dataset was not found in the sandbox.")