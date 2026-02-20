# test_phase3.py
import os
import pandas as pd
import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from preprocessing.clean_agent import build_cleaning_graph

if __name__ == "__main__":
    print("\n--- 1. GENERATING MESSY DATA ---")
    os.makedirs("sandbox", exist_ok=True)
    
    # Generate messy data with missing values, bad casing, and crazy outliers
    df_messy = pd.DataFrame({
        "category": ['Electronics', 'electronics', 'Toys', 'HOME', None],
        "price": [100.0, 5000000.0, np.nan, 45.0, 50.0], # Massive outlier included
    })
    
    print("\nOriginal Messy Data:")
    print(df_messy)
    
    df_messy.to_pickle("sandbox/messy_sales.pkl")
    
    initial_state = {
        "working_files": {"messy_sales.pkl": "sandbox/messy_sales.pkl"},
        "messages": []
    }
    
    print("\n--- 2. RUNNING ADVANCED CLEANING GRAPH ---")
    graph = build_cleaning_graph()
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory, interrupt_before=["human_review"])
    config = {"configurable": {"thread_id": "test_clean_1"}}
    
    for event in app.stream(initial_state, config=config):
        for node, state_update in event.items():
            print(f"Finished node: {node}")
            if "deep_profile_report" in state_update:
                print("Deep Profile Generated successfully.")
            if "cleaning_plan" in state_update:
                print("\nAI Structured Cleaning Plan:")
                print(state_update["cleaning_plan"])

    print("\nGraph paused for Human Review of the Cleaning Plan.")
    print("Simulating User Approval...\n")
    
    for event in app.stream(None, config=config):
        for node, state_update in event.items():
            print(f"Finished node: {node}")
            if "python_code" in state_update:
                print("\nGenerated Pandas Code:")
                print(state_update["python_code"])

    print("\n--- 3. VERIFYING CLEAN DATA ---")
    df_clean = pd.read_pickle("sandbox/messy_sales.pkl")
    print("\nCleaned Data Output:")
    print(df_clean)