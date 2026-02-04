from typing import Any, Dict, List
from .base import BaseAgent
from ..llm.client import LLMClient
from ..tools.base import ToolRegistry

class ExecutorAgent(BaseAgent):
    def __init__(self, llm: LLMClient, tool_registry: ToolRegistry):
        super().__init__(llm)
        self.tool_registry = tool_registry

    def _resolve_args(self, args: Dict[str, Any], context: Dict[int, Any]) -> Dict[str, Any]:
        """
        Recursively resolve arguments containing {{step_N...}} placeholders.
        """
        import re
        
        resolved = {}
        for k, v in args.items():
            if isinstance(v, str):
                # Regex to match {{step_N.path}} or {step_N.path}
                match = re.search(r"\{+step_(\d+)(.*?)\}+", v)
                if match:
                    step_id = int(match.group(1))
                    raw_path = match.group(2).strip(".").split(".")
                    
                    # 1. Get step result
                    val = context.get(step_id)
                    
                    # 2. Traverse path
                    try:
                        for part in raw_path:
                            if not part: continue
                            
                            # Handle array access like items[0]
                            array_match = re.match(r"(\w+)\[(\d+)\]", part)
                            if array_match:
                                key_name = array_match.group(1)
                                index = int(array_match.group(2))
                                
                                if isinstance(val, dict):
                                    val = val.get(key_name)
                                if isinstance(val, list) and 0 <= index < len(val):
                                    val = val[index]
                                else:
                                    val = None
                                    break
                            # Handle standard keys "0" (as index) or "key"
                            elif isinstance(val, list) and part.isdigit():
                                idx = int(part)
                                if 0 <= idx < len(val):
                                    val = val[idx]
                                else:
                                    val = None
                                    break
                            elif isinstance(val, dict):
                                val = val.get(part)
                            else:
                                val = None
                                break
                        
                        resolved[k] = val if val is not None else v
                    except Exception as e:
                        print(f"Warning: Resolution failed for {v}: {e}")
                        resolved[k] = v
                else:
                    resolved[k] = v
            else:
                resolved[k] = v
        return resolved

    def run(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        context = {} # Map step_id -> output
        steps = plan.get("steps", [])
        
        print("\n--- Executor Starting ---")
        for step in steps:
            step_id = step.get("step_id")
            description = step.get("description")
            tool_name = step.get("tool_name")
            raw_args = step.get("tool_args", {})

            # Resolve arguments using context
            tool_args = self._resolve_args(raw_args, context)
            
            print(f"Step {step_id}: {description}")
            if tool_args != raw_args:
                print(f"  Resolved Args: {tool_args}")
            
            if tool_name == "none":
                result = "No tool execution needed."
            else:
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    result = f"Error: Tool '{tool_name}' not found."
                else:
                    try:
                        # We pass context from previous steps if needed (?)
                        # For now, simplistic execution.
                        print(f"  Calling {tool_name} with {tool_args}")
                        result = tool.run(**tool_args)
                    except Exception as e:
                        result = f"Error executing tool: {e}"
            
            print(f"  Result: {str(result)[:100]}...") # Truncate for log
            
            results.append({
                "step_id": step_id,
                "description": description,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "output": result
            })
            
            # Update context for future steps
            context[step_id] = result
            
        print("--- Executor Finished ---\n")
        return results
