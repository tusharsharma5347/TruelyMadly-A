from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from pydantic import BaseModel

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool."""
        pass

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema for LLM consumption."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            }
        }

class ToolRegistry:
    """Registry to manage available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        return [tool.to_json_schema() for tool in self._tools.values()]
