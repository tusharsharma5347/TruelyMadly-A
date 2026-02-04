import json
from typing import Any, Dict, List
from .base import BaseAgent
from ..llm.client import LLMClient
from ..tools.base import ToolRegistry

PLANNER_PROMPT = """You are a Planner Agent.
Your goal is to break down a user's natural language request into a minimal sequence of steps that can be executed using the available tools.

Available Tools:
{tools}

Planning Rules:
- ONLY add steps that are necessary to satisfy the user's request.
- Prefer the smallest number of steps possible.
- If the user only asks for weather, use only the "get_weather" tool.
- If the user asks for GitHub information only, use only the GitHub tools.

Output Format:
You must output a valid JSON object with a "steps" key.
"steps" is a list of objects, where each object has:
- "step_id": integer, 1-indexed
- "description": string, what to do in this step
- "tool_name": string, the name of the tool to use
- "tool_args": object, specific arguments.

Data Dependencies:
If a step needs data from a previous step, reference earlier outputs with placeholders in this format:
- "{{step_1}}" to refer to the full output of step 1.
- "{{step_1[0].name}}" to refer to a field inside a list/dict output (list index + dict key).

IMPORTANT:
- Use EXACTLY double braces like "{{...}}".
- Use keys that match tool outputs. For GitHub search results, use "name", "description", "stars", "url".

Example Plan:
{{
  "steps": [
    {{
      "step_id": 1,
      "description": "Find a weather library",
      "tool_name": "github_search",
      "tool_args": {{ "query": "weather", "limit": 1 }}
    }},
    {{
      "step_id": 2,
      "description": "Get content of the repo",
      "tool_name": "github_content",
      "tool_args": {{ "repo_name": "{{step_1[0].name}}", "path": "README.md" }}
    }}
  ]
}}

User Request: {query}
"""

class PlannerAgent(BaseAgent):
    def __init__(self, llm: LLMClient, tool_registry: ToolRegistry):
        super().__init__(llm)
        self.tool_registry = tool_registry

    def run(self, query: str) -> Dict[str, Any]:
        tools_schema = self.tool_registry.get_tools_schema()
        tools_str = json.dumps(tools_schema, indent=2)
        
        system_content = PLANNER_PROMPT.format(tools=tools_str, query=query)
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Create a plan for this request."}
        ]

        # Define the schema for the planner's output
        tool_names = [
            t.get("function", {}).get("name")
            for t in tools_schema
            if t.get("function", {}).get("name")
        ]
        tool_name_enum = sorted(set(tool_names + ["none"]))

        plan_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {"type": "integer"},
                            "description": {"type": "string"},
                            "tool_name": {"type": "string", "enum": tool_name_enum},
                            "tool_args": {"type": "object"}
                        },
                        "required": ["step_id", "description", "tool_name", "tool_args"]
                    }
                }
            },
            "required": ["steps"]
        }

        return self.llm.structured_output(messages, plan_schema)
