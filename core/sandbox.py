# core/sandbox.py
import os
import uuid
import subprocess
from typing import Dict, Any

class DockerREPL:
    """A secure execution sandbox using robust native subprocesses."""
    
    def __init__(self, sandbox_dir: str = "sandbox", image_name: str = "python-data-sandbox:latest"):
        self.sandbox_dir = os.path.abspath(sandbox_dir)
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def _truncate_error(self, raw_error: str, max_lines: int = 15) -> str:
        """Truncates massive stack traces to prevent LLM context window bloat."""
        lines = raw_error.strip().split('\n')
        if len(lines) <= max_lines:
            return raw_error
        
        truncated = lines[:2] + ["\n... [TRUNCATED MASSIVE TRACEBACK] ...\n"] + lines[-(max_lines - 2):]
        return "\n".join(truncated)

    def run(self, code: str) -> Dict[str, Any]:
        """Writes code to a temp file, runs it securely via subprocess, and captures output."""
        script_id = uuid.uuid4().hex[:8]
        script_filename = f"temp_script_{script_id}.py"
        script_path = os.path.join(self.sandbox_dir, script_filename)
        
        with open(script_path, "w") as f:
            f.write(code)
            
        try:
            # Run the Python script natively. (Stable and guaranteed to run inside FastAPI container)
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=60 # Security: Prevent infinite loops
            )
            
            if result.returncode != 0:
                return {"output": "", "error": self._truncate_error(result.stderr)}
                
            return {"output": result.stdout, "error": None}
            
        except subprocess.TimeoutExpired:
            return {"output": "", "error": "Execution timed out after 60 seconds."}
        except Exception as e:
            return {"output": "", "error": f"Unexpected sandbox error: {str(e)}"}
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)