import requests
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from .base import BaseTool

DEFAULT_TIMEOUT_SECS = 15

class WeatherToolArgs(BaseModel):
    city: str = Field(..., description="Name of the city to get weather for")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather for a specific city."
    args_schema: Any = WeatherToolArgs

    def _get_coordinates(self, city: str) -> Optional[Dict[str, float]]:
        """Geocoding to get lat/lon."""
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "count": 1, "language": "en", "format": "json"}
        
        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECS)
            data = response.json()
            if not data.get("results"):
                return None
            return {
                "latitude": data["results"][0]["latitude"],
                "longitude": data["results"][0]["longitude"],
                "name": data["results"][0]["name"]
            }
        except requests.RequestException:
            return None

    def run(self, city: str) -> Dict[str, Any]:
        coords = self._get_coordinates(city)
        if not coords:
            return {"error": f"Could not find coordinates for {city}", "source": "get_weather"}
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
        }
        
        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECS)
            data = response.json()
            current = data.get("current", {})
            
            return {
                "city": coords["name"],
                "temperature": current.get("temperature_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "unit": data.get("current_units", {}).get("temperature_2m", "°C")
            }
        except requests.RequestException as e:
            return {"error": str(e), "source": "get_weather"}
