from .base import ToolRegistry
from .github_tool import GitHubSearchTool, GitHubContentTool
from .weather_tool import WeatherTool

def load_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GitHubSearchTool())
    registry.register(GitHubContentTool())
    registry.register(WeatherTool())
    return registry
