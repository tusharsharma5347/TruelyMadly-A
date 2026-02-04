import os
import json
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Wrapper for OpenAI-compatible LLM APIs."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "local")
        self.api_key = os.getenv("LLM_API_KEY", "ollama")
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = os.getenv("LLM_MODEL", "llama3")
        
        # Configure client
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        else:
            # Local or other compatible provider
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0
    ) -> Any:
        """
        Send a chat completion request to the LLM.
        """
        # Prepare arguments
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if tools:
            kwargs["tools"] = tools
            
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
            
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise e

    def structured_output(self, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Force the LLM to output valid JSON conforming to a schema.
        Note: Checks if the provider supports 'json_schema' or 'json_object'.
        
        For simplicity with generic models, we'll prompt engineering + json mode if available.
        """
        def _extract_first_json_object(text: str) -> str:
            """
            Extract the first top-level JSON object from a string.
            Helps when models accidentally add leading/trailing text.
            """
            if not text:
                return text
            start = text.find("{")
            if start == -1:
                return text
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start : i + 1]
            return text[start:]

        # Ensure we have a system message; don't mutate caller list in-place.
        msgs = list(messages)
        if not msgs or msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": "You are a helpful assistant. Output only valid JSON."})

        # Keep schema instruction concise to reduce prompt bloat.
        schema_hint = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
        msgs.append(
            {
                "role": "system",
                "content": (
                    "Return a single JSON object that matches this JSON Schema exactly. "
                    "Do not include markdown, prose, or code fences.\n"
                    f"SCHEMA:{schema_hint}"
                ),
            }
        )

        kwargs = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = _extract_first_json_object(content)
                return json.loads(extracted)
        except Exception as e:
            print(f"Error getting structured output: {e}")
            raise e
