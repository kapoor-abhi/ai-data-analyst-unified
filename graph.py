# graph.py
from core.super_agent import build_super_graph

if __name__ == "__main__":
    print("Building Super-Graph...")
    # FIX: Add .compile() here so LangGraph can render it!
    graph = build_super_graph().compile() 
    
    print("Generating PNG image...")
    # Now it is a CompiledStateGraph and has the get_graph() method
    png_bytes = graph.get_graph().draw_mermaid_png()
    
    with open("super_agent_architecture.png", "wb") as f:
        f.write(png_bytes)
        
    print("SUCCESS! Open 'super_agent_architecture.png' to see your AI architecture.")