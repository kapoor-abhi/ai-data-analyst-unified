# test_phase4.py
import os
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from chat.chat_agent import build_chat_graph

if __name__ == "__main__":
    os.makedirs("sandbox", exist_ok=True)
    
    # 1. Create a dummy cleaned dataset in case Phase 3 data isn't there
    df_clean = pd.DataFrame({
        "category": ['electronics', 'electronics', 'toys', 'home', 'unknown'],
        "price": [100.0, 5000000.0, 75.0, 45.0, 50.0],
    })
    df_clean.to_pickle("sandbox/clean_sales.pkl")
    
    print("\n=============================================")
    print("TEST 1: DUCKDB SQL QUERY")
    print("=============================================")
    
    state_sql = {
        "working_files": {"clean_sales.pkl": "sandbox/clean_sales.pkl"},
        "user_input": "What is the average price of electronics?",
        "messages": []
    }
    
    graph = build_chat_graph()
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config_sql = {"configurable": {"thread_id": "test_chat_sql"}}
    
    for event in app.stream(state_sql, config=config_sql):
        for node, state_update in event.items():
            print(f"Finished node: {node}")
            if "next_step" in state_update:
                print(f"-> Router decided to: {state_update['next_step']}")
                
    final_state = app.get_state(config_sql).values
    print("\nAI Response:")
    print(final_state["messages"][-1].content)


    print("\n=============================================")
    print("TEST 2: PYTHON VISUALIZATION")
    print("=============================================")
    
    state_viz = {
        "working_files": {"clean_sales.pkl": "sandbox/clean_sales.pkl"},
        "user_input": "Plot a bar chart showing the count of each category. I want it to look beautiful.",
        "messages": [],
        "charts_generated": [],
        "iteration_count": 0
    }
    
    config_viz = {"configurable": {"thread_id": "test_chat_viz"}}
    
    for event in app.stream(state_viz, config=config_viz):
        for node, state_update in event.items():
            print(f"Finished node: {node}")
            if "next_step" in state_update:
                print(f"-> Router decided to: {state_update['next_step']}")
                
    final_state_viz = app.get_state(config_viz).values
    print("\nAI Response:")
    print(final_state_viz["messages"][-1].content)
    
    if final_state_viz.get("charts_generated"):
        print(f"\nSUCCESS! Chart was actually saved to: {final_state_viz['charts_generated'][-1]}")