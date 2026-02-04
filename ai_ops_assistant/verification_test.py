import unittest
from unittest.mock import MagicMock
from ai_ops_assistant.llm.client import LLMClient
from ai_ops_assistant.tools import load_tools
from ai_ops_assistant.agents import PlannerAgent, ExecutorAgent, VerifierAgent

class TestAIOpsAssistant(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=LLMClient)
        self.mock_llm.provider = "mock"
        self.mock_llm.model = "mock-gpt"
        self.registry = load_tools()

    def test_full_flow(self):
        # 1. Setup Planner Mock
        # when planner.run is called, it calls llm.structured_output
        self.mock_llm.structured_output.side_effect = [
            # Planner response
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Check weather in London",
                        "tool_name": "get_weather",
                        "tool_args": {"city": "London"}
                    }
                ]
            },
            # Verifier response
            {
                "status": "success",
                "final_answer": "The weather in London is 15°C with wind speed 10km/h.",
                "missing_info": "",
                "retry_plan": {"steps": []}
            }
        ]

        # 2. Run Agents
        planner = PlannerAgent(self.mock_llm, self.registry)
        executor = ExecutorAgent(self.mock_llm, self.registry)
        verifier = VerifierAgent(self.mock_llm)

        query = "What is the weather in London?"
        
        # Planner
        plan = planner.run(query)
        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["tool_name"], "get_weather")

        # Executor
        # We need to mock the actual tool execution inside executor or let it fail gracefully if no internet
        # But here we are using real tools. WeatherTool uses requests. 
        # Let's mock the tool registry for executor to avoid network calls during test
        mock_tool = MagicMock()
        mock_tool.run.return_value = {"temperature": 15, "wind_speed": 10}
        self.registry._tools["get_weather"] = mock_tool

        results = executor.run(plan)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["output"], {"temperature": 15, "wind_speed": 10})

        # Verifier
        verification = verifier.run(query, results)
        self.assertEqual(verification["status"], "success")
        self.assertIn("missing_info", verification)

if __name__ == "__main__":
    unittest.main()
