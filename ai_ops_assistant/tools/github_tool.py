import requests
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .base import BaseTool

class GitHubSearchArgs(BaseModel):
    query: str = Field(..., description="The search query (e.g., 'python agents')")
    limit: int = Field(5, description="Number of results to return")

class GitHubSearchTool(BaseTool):
    name: str = "github_search"
    description: str = "Search for repositories on GitHub."
    args_schema: Any = GitHubSearchArgs

    def run(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "per_page": limit, "sort": "stars"}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "name": item["full_name"],
                    "description": item["description"],
                    "stars": item["stargazers_count"],
                    "url": item["html_url"]
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]

class GitHubContentArgs(BaseModel):
    repo_name: str = Field(..., description="Full repository name (e.g., 'owner/repo')")
    path: str = Field("", description="File path to fetch (optional)")

class GitHubContentTool(BaseTool):
    name: str = "github_content"
    description: str = "Get details or content of a GitHub repository."
    args_schema: Any = GitHubContentArgs

    def run(self, repo_name: str, path: str = "") -> Dict[str, Any]:
        if not repo_name:
            return {"error": "repo_name cannot be empty"}

        url = f"https://api.github.com/repos/{repo_name}"
        if path:
            url += f"/contents/{path}"
            
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
