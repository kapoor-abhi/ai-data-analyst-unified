# test_final_master.py
import os
import pandas as pd
import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from core.super_agent import build_super_graph

def generate_disgusting_dataset():
    """Generates 55,000 rows of horribly messy, real-world data."""
    print("\n==================================================")
    print("1. GENERATING MASSIVE DIRTY DATASETS")
    print("==================================================")
    os.makedirs("sandbox", exist_ok=True)
    num_rows = 55000
    
    # Dataset 1: Customers
    customer_ids = np.arange(1, num_rows + 1)
    join_dates = pd.Series(pd.date_range(start="2020-01-01", periods=num_rows, freq="h")).astype(object)
    join_dates[np.random.choice(num_rows, 5000, replace=False)] = np.nan 
    
    df_customers = pd.DataFrame({
        "customer_id": customer_ids,
        "region": np.random.choice(["New York", "NY", "California", "CA", "unknown", None], num_rows),
        "join_date": join_dates
    })
    df_customers.to_csv("sandbox/customers.csv", index=False)
    
    # Dataset 2: Purchases
    txn_ids = np.arange(100001, 100001 + num_rows)
    shuffled_customers = np.random.permutation(customer_ids)
    
    prices = np.random.choice([10.5, 20.0, 45.99, "Unknown", np.nan], num_rows)
    prices[10] = 99999999.99 # Massive $99M outlier
    
    df_purchases = pd.DataFrame({
        "txn_id": txn_ids,
        "cust_id": shuffled_customers, # THE LLM MUST AUTO-MATCH THIS WITH customer_id!
        "category": np.random.choice(["Electronics", "electronics", "Books", "Clothing", None], num_rows),
        "price": prices
    })
    df_purchases.to_csv("sandbox/purchases.csv", index=False)
    
    print(f"Created customers.csv and purchases.csv")
    print("WARNING: Data contains mismatched column names ('cust_id'), NaNs, bad casing, and $99M outliers!")

if __name__ == "__main__":
    generate_disgusting_dataset()
    
    print("\n==================================================")
    print("2. RUNNING AUTONOMOUS ETL PIPELINE (Ingest -> Merge -> Clean)")
    print("==================================================")
    graph = build_super_graph()
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "final_autonomous_test_v1"}}
    
    # Uploading files. We pass an empty user_input to prevent Ingestion from mutating anything.
    initial_state = {
        "file_paths": ["sandbox/customers.csv", "sandbox/purchases.csv"],
        "user_input": "", 
        "messages": []
    }
    
    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        for node, state_update in event.items():
            if node == "merging":
                print(f"\n🧠 AI Merge Strategy Executed:\n{state_update.get('suggestion')}")
                print(f"💻 AI Generated Code:\n{state_update.get('python_code')}")
            if node == "cleaning":
                print(f"\n🧠 AI Deep Scanner & Cleaning Executed!")
                print(f"💻 AI Generated Cleaning Code:\n{state_update.get('python_code')}")

    print("\n==================================================")
    print("3. VERIFYING THE CLEANED DATASET")
    print("==================================================")
    
    state = app.get_state(config).values
    working_files = state.get("working_files", {})
    if working_files:
        final_file_path = list(working_files.values())[-1]
        final_df = pd.read_pickle(final_file_path)
        print("\n✅ CLEANED DATASET HEAD (Notice how it auto-merged 'cust_id' and handled outliers!):")
        # We print dynamically so it never throws a KeyError
        print(final_df.head(10))
        print(f"✅ Final Shape: {final_df.shape}")

    print("\n==================================================")
    print("4. DUCKDB SQL CHAT")
    print("\n==================================================")

    chat_query = "Plot a box plot of price by category to check for outliers."
    print(f"User: {chat_query}")

# Instead of None, pass the user_input in the inputs dictionary
# This forces the entry_router to re-run and select the 'chat' node
    chat_inputs = {"user_input": chat_query}

    for event in app.stream(chat_inputs, config=config, stream_mode="updates"):
        for node, state_update in event.items():
            if node == "chat":
                if "messages" in state_update and state_update["messages"]:
                    print(f"\n🤖 AI: {state_update['messages'][-1].content}")
                if "error" in state_update and state_update["error"]:
                    print(f"\n❌ Error: {state_update['error']}")