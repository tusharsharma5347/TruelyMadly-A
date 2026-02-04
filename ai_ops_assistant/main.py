import sys
import json
import os
import argparse
from dotenv import load_dotenv

from ai_ops_assistant.llm.client import LLMClient
from ai_ops_assistant.tools import load_tools
from ai_ops_assistant.agents import PlannerAgent, ExecutorAgent, VerifierAgent

MAX_VERIFIER_RETRIES = int(os.getenv("MAX_VERIFIER_RETRIES", "2"))

def run_once(user_input: str, planner: PlannerAgent, executor: ExecutorAgent, verifier: VerifierAgent) -> None:
    # 2. Plan
    print("\n[Planner] Generating plan...")
    plan = planner.run(user_input)
    print(f"[Planner] Plan: {json.dumps(plan, indent=2)}")

    # 3. Execute
    print("\n[Executor] Executing plan...")
    execution_results = executor.run(plan)

    # 4. Verify
    print("\n[Verifier] Verifying results...")
    verification = verifier.run(user_input, execution_results)

    # 5. Retry loop (Verifier-driven)
    retries = 0
    while (
        verification.get("status") != "success"
        and retries < MAX_VERIFIER_RETRIES
        and isinstance(verification.get("retry_plan"), dict)
        and verification["retry_plan"].get("steps")
    ):
        retries += 1
        retry_plan = verification["retry_plan"]
        print(f"\n[Verifier] Retry plan proposed (attempt {retries}/{MAX_VERIFIER_RETRIES}).")
        print(f"[Verifier] Retry Plan: {json.dumps(retry_plan, indent=2)}")

        print("\n[Executor] Executing retry plan...")
        retry_results = executor.run(retry_plan)
        execution_results.extend(retry_results)

        print("\n[Verifier] Re-verifying results...")
        verification = verifier.run(user_input, execution_results)

    print("\n=== Final Response ===")
    if verification.get("status") == "success":
        answer = verification.get("final_answer")
        if isinstance(answer, (dict, list)):
            print(json.dumps(answer, indent=2))
        else:
            print(answer)
    else:
        print("Task Failed or Incomplete.")
        print(f"Reason: {verification.get('missing_info')}")
        print(f"Partial Answer: {verification.get('final_answer')}")
    print("========================")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI Operations Assistant")
    parser.add_argument("--task", type=str, help="Run a single task non-interactively and exit.")
    args = parser.parse_args()
    
    # 1. Initialize
    print("Initializing AI Operations Assistant...")
    try:
        llm = LLMClient()
        tools_registry = load_tools()
        
        planner = PlannerAgent(llm, tools_registry)
        executor = ExecutorAgent(llm, tools_registry)
        verifier = VerifierAgent(llm)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    print("Ready! (Type 'quit' to exit)")
    print(f"Using Provider: {llm.provider}, Model: {llm.model}")

    # Non-interactive mode: explicit --task or piped stdin
    if args.task:
        run_once(args.task.strip(), planner, executor, verifier)
        return
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            run_once(piped, planner, executor, verifier)
        return

    while True:
        try:
            user_input = input("\n> User Request: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            run_once(user_input, planner, executor, verifier)

        except EOFError:
            # e.g. Ctrl-D or closed stdin
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
