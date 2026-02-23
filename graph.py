import os
import sys

# Import your uncompiled super_graph object
try:
    from app.main import super_graph
except ImportError:
    print("Could not import super_graph from app.main. Please update the import path in this script.")
    sys.exit(1)

def generate_detailed_architecture():
    print("Initializing LangGraph X-Ray compiler...")
    
    try:
        # THE FIX: We must compile() the StateGraph before calling get_graph()
        compiled_graph = super_graph.compile()
        detailed_graph = compiled_graph.get_graph(xray=1)
        
        print("Rendering Mermaid PNG (this requires an active internet connection)...")
        png_bytes = detailed_graph.draw_mermaid_png()
        
        output_filename = "full_agent_architecture.png"
        with open(output_filename, "wb") as f:
            f.write(png_bytes)
            
        print(f"Success! Detailed architecture saved to: {os.path.abspath(output_filename)}")
        
    except Exception as e:
        print(f"Failed to generate graph: {e}")

if __name__ == "__main__":
    generate_detailed_architecture()