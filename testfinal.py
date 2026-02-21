import os
import random
import datetime
import pandas as pd
import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Import your graph directly
from core.super_agent import build_super_graph

def generate_advanced_datasets(n_rows=1500):
    print("\n==================================================")
    print("1. GENERATING ADVANCED ENTERPRISE DATASETS")
    print("==================================================")
    os.makedirs("sandbox", exist_ok=True)
    np.random.seed(42)

    client_ids = [f"C{i}" for i in range(1, n_rows + 1)]
    
    # Generate varied random dates
    base_date = datetime.date(2026, 1, 1)
    dates = []
    for _ in range(n_rows):
        random_days = random.randint(0, 45) # Spread across 45 days
        d = base_date + datetime.timedelta(days=random_days)
        # Mix ISO format and slash format
        dates.append(d.strftime("%Y-%m-%d") if random.random() > 0.3 else d.strftime("%d/%m/%Y"))
        
    # Inject fake nulls
    for i in random.sample(range(n_rows), int(n_rows * 0.05)):
        dates[i] = "?" 

    statuses = np.random.choice([" Active ", "inactive", "ACTIVE", " pending", "?", "-999"], n_rows)

    df_cust = pd.DataFrame({"client_id": client_ids, "signup_date": dates, "account_status": statuses})
    
    # Inject exact duplicates to trigger the deduplication logic
    df_cust = pd.concat([df_cust, df_cust.sample(50)]).sample(frac=1).reset_index(drop=True)
    df_cust.to_csv("sandbox/adv_customers.csv", index=False)

    revenues = []
    for _ in range(n_rows):
        if random.random() < 0.05:
            revenues.append("NaN") # String fake null
        else:
            # Mixed currency formats
            val = round(random.uniform(10.0, 5000.0), 2)
            revenues.append(f"${val:,.2f}" if random.random() > 0.5 else str(val))

    df_orders = pd.DataFrame({
        "order_id": range(10001, 10001 + n_rows),
        "user_id": client_ids, 
        "revenue": revenues,
        "priority": np.random.choice(["High", "Low", "Medium"], n_rows)
    })
    
    df_orders.to_csv("sandbox/adv_orders.csv", index=False)
    
    print(f"✅ Created 'adv_customers.csv' ({len(df_cust)} rows) and 'adv_orders.csv' ({len(df_orders)} rows).")
    print("⚠️  Injected Issues: Mismatched keys, Fake Nulls, Mixed Currency, Messy Dates, Duplicates.")
    return ["sandbox/adv_customers.csv", "sandbox/adv_orders.csv"]

def print_current_data(state_values, sample_size=3):
    """Helper function to print the actual data inside the working files."""
    working_files = state_values.get("working_files", {})
    if not working_files:
        print("   [No working files tracked in state yet]")
        return
        
    for filename, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            print(f"\n   📄 {filename} | Shape: {df.shape}")
            print(df.head(sample_size).to_string(index=False)) 
        except Exception as e:
            print(f"   [!] Could not read data for {filename}: {e}")

def run_advanced_test(file_paths):
    print("\n==================================================")
    print("2. RUNNING AUTONOMOUS ETL PIPELINE (WITH AUTO-APPROVE)")
    print("==================================================")
    
    graph = build_super_graph()
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "advanced_test_run"}}
    
    initial_state = {
        "file_paths": file_paths, 
        "user_input": "Load and clean this data.", 
        "messages": []
    }
    
    print("[🚀] Starting Graph Execution...")
    app.invoke(initial_state, config=config)
    
    approval_count = 0
    max_approvals = 15 # Circuit breaker to prevent infinite loops if AI gets stuck
    
    while True:
        state = app.get_state(config)
        if not state.next: 
            print("\n✅ ETL Pipeline Completed! No pending nodes.")
            break
            
        pending_node = state.next[0]
        state_values = state.values
        
        print(f"\n{'='*80}\n⏸️  PAUSED FOR HUMAN AT NODE: {pending_node.upper()}\n{'='*80}")
        
        # 1. Diagnose why we are paused
        if state_values.get("error"):
            print(f"⚠️  AI HIT AN ERROR (Attempt {state_values.get('iteration_count', 0)}):")
            print(state_values["error"])
        
        if pending_node == "human_strategy" and state_values.get("suggestion"):
            print("\n🧠 AI MERGE STRATEGY:")
            print(state_values["suggestion"])
            
        elif pending_node == "human_review" and state_values.get("cleaning_plan"):
            print("\n🧠 AI CLEANING PLAN:")
            print(state_values["cleaning_plan"])
        
        # 2. X-Ray Current Data State
        print("\n--- 🔍 LIVE DATA X-RAY ---")
        print_current_data(state_values)
        print("--------------------------")
        
        # 3. Auto-Approve Mechanism
        approval_count += 1
        if approval_count > max_approvals:
            print("\n❌ MAX APPROVALS REACHED. The AI is caught in a loop. Exiting test.")
            break
            
        human_feedback = "approve"
        print(f"\n[🤖 Auto-HITL] Injecting feedback: '{human_feedback}' (Action {approval_count}/{max_approvals})")
            
        # 4. Resume Graph
        print(f"▶️ Resuming Execution...")
        for event in app.stream(Command(resume=human_feedback), config=config, stream_mode="updates"):
            for node, update in event.items():
                if "python_code" in update:
                    print(f"\n[⚙️  {node.upper()} GENERATED CODE]\n{update['python_code']}\n")
                if "error" in update and update["error"]:
                    print(f"\n[⚠️  {node.upper()} EXECUTION FAILED]\n{update['error']}\n")

    print("\n==================================================")
    print("3. FINAL DATA VERIFICATION & AUTO-CHAT TEST")
    print("==================================================")
    final_state = app.get_state(config).values
    working_files = final_state.get("working_files", {})
    
    merged_path = working_files.get("merged_dataset.pkl")
    if merged_path and os.path.exists(merged_path):
        df = pd.read_pickle(merged_path)
        
        # Save a CSV copy so you can physically inspect the result
        csv_path = "sandbox/FINAL_ADVANCED_CLEANED.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Final Dataset Output: {csv_path}")
        print(f"📊 Final Rows: {len(df)} | Columns: {len(df.columns)}")
        print("\n🔍 Cleaned Data Sample:")
        print(df.head(5).to_string(index=False))
        
        # -----------------------------------------
        # Automated Chat Tests
        # -----------------------------------------
        print("\n🤖 Running Automated Chat Validations...")
        
        test_queries = [
            "What is the total revenue?",
            "Plot a bar chart of account status distribution."
        ]
        
        for query in test_queries:
            print(f"\n🧪 Testing Query: '{query}'")
            output = app.invoke({"user_input": query}, config)
            if output.get("messages"): 
                print(f"   🤖 AI Reply: {output['messages'][-1].content}")
            if output.get("charts_generated"):
                print(f"   📊 Chart Generated: {output['charts_generated'][-1]}")
        
        # -----------------------------------------
        # Interactive Chat
        # -----------------------------------------
        print("\n--- 💬 Interactive Chat Mode Ready (type 'exit' to quit) ---")
        while True:
            try:
                user_msg = input("\nYou: ")
                if user_msg.lower() in ['exit', 'quit']: break
                
                print("[...] Thinking...")
                output = app.invoke({"user_input": user_msg}, config)
                
                if output.get("messages"): 
                    print(f"\nAI: {output['messages'][-1].content}")
                if output.get("charts_generated"):
                    print(f"\n[Chart Saved: {output['charts_generated'][-1]}]")
                    
            except KeyboardInterrupt:
                break
    else:
        print("❌ Error: Merged file not found. The pipeline failed to reach the end.")

if __name__ == "__main__":
    paths = generate_advanced_datasets()
    run_advanced_test(paths)