import json
from typing import Any, Dict, List, Optional
from .base import BaseAgent

VERIFIER_PROMPT = """You are a Verifier Agent.
Your job is to review the results of an executed plan and determine if the user's original request was satisfied.

User Request: {query}

Execution Results:
{results_json}

Task:
1. Identify the user's required deliverables from the User Request.
2. Synthesize the execution results into a helpful final answer that addresses the request.
3. Treat tool errors as non-fatal if you can still satisfy the user's request from other successful steps.
   - Example: If the user only asked for weather, and weather was fetched successfully, mark success even if unrelated GitHub steps failed.
4. If the request is NOT satisfied, explain what is missing.
5. If missing info can be obtained by calling available tools, propose a small retry plan.

Output JSON with keys:
- "status": "success" or "failure"
- "final_answer": string (always non-empty; include partial answer if needed)
- "missing_info": string (if failure, what is missing. If success, leave empty string)
- "retry_plan": optional object with key "steps" (same shape as Planner plan) to fetch missing info.
  - Only include retry_plan if tool calls can resolve the missing info.
  - Keep it minimal (1-3 steps). Use placeholders like "{{step_1[0].name}}".
  - IMPORTANT: If a previous step returned a list of repos with keys "name/description/stars/url", do NOT use "full_name".

IMPORTANT: Output ONLY valid JSON. Do not use Markdown code blocks (```json ... ```). Just the JSON object.
"""

class VerifierAgent(BaseAgent):
    def run(self, query: str, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        results_json = json.dumps(execution_results, indent=2, default=str)

        messages = [
            {"role": "user", "content": VERIFIER_PROMPT.format(query=query, results_json=results_json)}
        ]
        
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "failure"]},
                "final_answer": {"type": "string"},
                "missing_info": {"type": "string"},
                "retry_plan": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step_id": {"type": "integer"},
                                    "description": {"type": "string"},
                                    "tool_name": {"type": "string"},
                                    "tool_args": {"type": "object"},
                                },
                                "required": ["step_id", "description", "tool_name", "tool_args"],
                            },
                        }
                    },
                    "required": ["steps"],
                },
            },
            "required": ["status", "final_answer", "missing_info"]
        }

        return self.llm.structured_output(messages, schema)
