"""
Self-improving agent using Pydantic AI + Monty sandbox.
Uses structured outputs, tools, and built-in retries for better reliability.
"""

import tomllib
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from ollama_model import OllamaModel
from loguru import logger
import pydantic_monty as monty
import logfire

try:
    import logfire
    LOGFIRE_ENABLED = True
except ImportError:
    LOGFIRE_ENABLED = False

if LOGFIRE_ENABLED:
    logfire.configure(service_name="monty-pydantic-ai-agent")


class GeneratedCode(BaseModel):
    """Structured LLM output — only pure code allowed."""
    code: str = Field(..., description="Raw Monty-compatible Python code. No markdown, no explanations, no fences.")


class AgentConfig(BaseModel):
    """Parsed TOML config."""
    model: str = "codellama:34b"
    max_attempts: int = 12
    log_file: str = "agent.log"
    log_level: str = "INFO"
    prompts: Dict[str, str] = {}


def load_config(config_path: str = "config/agent_config.toml") -> AgentConfig:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_file, "rb") as f:
        raw = tomllib.load(f)
    return AgentConfig(**raw)


def create_monty_agent(config: AgentConfig) -> Agent:
    """Create Pydantic AI agent with custom Ollama model wrapper."""
    model = OllamaModel(model_name=config.model)

    system_prompt = (
        f"{config.prompts.get('universal_rules', '')}\n\n"
        f"{config.prompts.get('monty_constraints', '')}\n\n"
        "Generate code for the task using provided inputs. "
        "Always assign final value to 'result'. Output structured response."
    )

    agent = Agent(
        model=model,
        output_type=GeneratedCode,
        instructions=system_prompt,
        retries=config.max_attempts - 1,
    )

    @agent.tool
    async def execute_in_monty(ctx: RunContext, generated_code: str, expected_output: Optional[Any] = None) -> Dict[str, Any]:
        try:
            m = monty.Monty(generated_code, inputs=list(ctx.inputs.keys()))
            output = m.run(inputs=ctx.inputs)

            result = getattr(output, 'result', None)
            stdout = getattr(output, 'stdout', '').strip()

            if expected_output is not None and result != expected_output:
                return {"success": False, "error": f"Output {result} != expected {expected_output}"}

            return {"success": True, "result": result, "stdout": stdout}
        except monty.MontyError as e:
            return {"success": False, "error": str(e)}

    return agent

class SelfImprovingMontyAgent:
    """
    Self-improving agent using Pydantic AI + Monty sandbox.
    Generates code, executes via tool, refines on failure or test mismatch.
    """

    def __init__(self, config_path: str = "config/agent_config.toml") -> None:
        self.config = load_config(config_path)
        self.agent = create_monty_agent(self.config)

        logger.add(self.config.log_file, level=self.config.log_level)

    async def run(
        self,
        task: str,
        inputs: Dict[str, Any],
        expected_output: Optional[Any] = None,
    ) -> Any:
        deps = {"inputs": inputs, "expected_output": expected_output}

        # Await the agent
        result = await self.agent.run(task, deps=deps)

        final_code = result.output.code
        execution_result = result.tool_outputs[-1] if result.tool_outputs else {}

        logger.info(f"Agent completed task '{task}'")
        logger.info(f"Final code:\n{final_code}")
        logger.info(f"Execution result: {execution_result}")

        return execution_result
    
# ────────────────────────────────────────────────
# Main – Quick test harness for the Pydantic AI agent
# ────────────────────────────────────────────────

import asyncio

async def main() -> None:
    try:
        config = load_config("config/agent_config.toml")
        print(f"Loaded agent config – model: {config.model}, max_attempts: {config.max_attempts}")
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    agent = SelfImprovingMontyAgent(config_path="config/agent_config.toml")

    print("\n" + "="*60)
    print("Testing Pydantic AI + Monty agent")
    print("="*60 + "\n")

    # ────────────────
    # Test cases
    # ────────────────

    test_cases = [
        {
            "name": "Simple addition ×2",
            "task": "Add two numbers x and y, then multiply by 2",
            "inputs": {"x": 5, "y": 7},
            "expected": 24,
        },
        {
            "name": "List median (odd length)",
            "task": "Compute the median of a list of numbers called data",
            "inputs": {"data": [7, 3, 1, 9, 5]},
            "expected": 5.0,
        },
        {
            "name": "List median (empty list)",
            "task": "Compute the median of a list of numbers called data (return None for empty)",
            "inputs": {"data": []},
            "expected": None,
        },
        {
            "name": "Count vowels",
            "task": "Count vowels (a,e,i,o,u — case insensitive) in string text",
            "inputs": {"text": "The quick brown fox jumps over the lazy dog"},
            "expected": 11,
        },
    ]

    for case in test_cases:
        print(f"\n[TEST] {case['name']}")
        print(f"Task: {case['task']}")
        print(f"Inputs: {case['inputs']}")
        print(f"Expected: {case['expected']}")

        try:
            result = await agent.run(
                task=case["task"],
                inputs=case["inputs"],
                expected_output=case["expected"],
            )

            print("\nResult:")
            pprint(result)

            if "success" in result and result["success"]:
                print("→ PASSED")
            else:
                print("→ FAILED (see error or mismatch above)")
        except Exception as e:
            print(f"→ EXCEPTION during run: {e}")

        print("-" * 60)

    print("\nAll tests complete.")

if __name__ == "__main__":
    asyncio.run(main())
