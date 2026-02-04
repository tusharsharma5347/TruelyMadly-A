from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..llm.client import LLMClient

class BaseAgent(ABC):
    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass
