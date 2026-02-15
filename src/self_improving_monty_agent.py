import tomllib
from pathlib import Path
import ollama
import pydantic_monty as monty
from loguru import logger
from typing import Any, Dict, List, Optional


class SelfImprovingMontyAgent:
    """
    A self-improving agent that generates and refines Monty-compatible Python code using a local LLM (Ollama).
    Executes code in the Monty sandbox and refines on errors. Supports early exit when code passes all provided tests.
    
    Config-driven via TOML (model, attempts, prompts, logging).
    """

    def __init__(self, config_path: str = "config/agent_config.toml") -> None:
        """
        Initialize agent from TOML configuration.

        Args:
            config_path: Path to agent configuration file.

        Raises:
            FileNotFoundError, ValueError: on config issues.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.universal_rules: str = self.config.get("prompts", {}).get("universal_rules", "")
        self.monty_constraints: str = self.config.get("prompts", {}).get("monty_constraints", "")
        self.refine_prompt_template: str = self.config.get("refine_prompt_template", (
            "{base_prompt}\nPrevious attempt failed with: '{error}'. "
            "Previous code was:\n{last_attempt}\nFix it and output ONLY the corrected code."
        ))
        self.last_attempt: Optional[str] = None

        logger.add(
            self.config.get("log_file", "agent.log"),
            level=self.config.get("log_level", "INFO")
        )

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate TOML configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        required = ["model", "max_attempts"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing keys: {', '.join(missing)}")

        logger.info(f"Loaded config from {config_path}")
        return config

    @staticmethod
    def clean_code(raw: str) -> str:
        """
        Clean LLM-generated code: remove markdown, leading junk, comments, etc.
        """
        text = raw.strip()
        lines = text.splitlines()
        code_lines: List[str] = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if low.startswith(('```', 'python', 'py', 'assistant:', 'here is')):
                continue
            if stripped and not stripped.startswith('#'):
                in_code = True
            if in_code:
                code_lines.append(line.rstrip())

        cleaned = '\n'.join(code_lines).strip()
        if '```' in cleaned:
            cleaned = cleaned.split('```')[0].strip()

        return cleaned

    def llm_generate(self, prompt: str) -> str:
        """Generate code from Ollama."""
        logger.debug(f"Sending prompt (first 100 chars): {prompt[:100]}...")
        try:
            response = ollama.chat(
                model=self.config["model"],
                messages=[{'role': 'user', 'content': prompt}]
            )
            generated = response['message']['content'].strip()
            logger.debug(f"LLM response (first 100 chars):\n{generated[:100]}...")
            return generated
        except ollama.ResponseError as e:
            logger.error(f"Ollama error: {e}")
            raise

    def _generate_task_guidance(self, task: str, input_keys: List[str]) -> str:
        """Generate task-specific instructions."""
        var_list = ", ".join(f"`{k}`" for k in sorted(input_keys))
        return f"""
Task: {task}

Available input variables: {var_list}
- Use ONLY these variables — do NOT hard-code values
- Do NOT use input(), eval(), or any I/O except print()
- Assign final answer to variable 'result'

Output ONLY the code.
"""

    def run(self, task: str, inputs: Dict[str, Any], tests: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Run agent loop to generate/refine code until success or max attempts reached.
        Supports optional list of tests for early exit when all pass.

        Args:
            task: Task description
            inputs: Primary input variables
            tests: Optional list of {"inputs": dict, "expected": Any} for multi-case validation

        Returns:
            Monty execution output on success

        Raises:
            ValueError: if max attempts reached without success
        """
        task_guidance = self._generate_task_guidance(task, list(inputs.keys()))
        base_prompt = f"{self.universal_rules}\n\n{self.monty_constraints}\n\n{task_guidance}"
        prompt = base_prompt
        max_attempts = self.config["max_attempts"]
        tests = tests or []  # default to empty

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Attempt {attempt}/{max_attempts}")

            generated_code = self.llm_generate(prompt)
            generated_code = self.clean_code(generated_code)
            self.last_attempt = generated_code

            logger.debug(f"Generated code:\n{generated_code}\n")

            try:
                m = monty.Monty(generated_code, inputs=list(inputs.keys()))
                output = m.run(inputs=inputs)

                logger.success(f"Success on attempt {attempt}!")

                # Check primary case (for backward compatibility)
                primary_result = getattr(output, 'result', None) or getattr(output, 'stdout', '').strip()

                # Check all additional tests
                all_tests_passed = True
                for test_case in tests:
                    test_inputs = test_case["inputs"]
                    test_expected = test_case["expected"]

                    test_m = monty.Monty(generated_code, inputs=list(test_inputs.keys()))
                    test_output = test_m.run(inputs=test_inputs)
                    test_result = getattr(test_output, 'result', None) or getattr(test_output, 'stdout', '').strip()

                    if test_result != test_expected:
                        all_tests_passed = False
                        logger.warning(f"Test failed: inputs={test_inputs}, expected={test_expected}, got={test_result}")

                if all_tests_passed:
                    logger.success("All tests passed — early exit")
                    return output
                else:
                    logger.warning("Code ran but failed some tests — refining...")
                    prompt = self.refine_prompt_template.format(
                        base_prompt=base_prompt,
                        error="Failed one or more test cases",
                        last_attempt=self.last_attempt
                    )

            except monty.MontyError as e:
                logger.warning(f"Monty error on attempt {attempt}: {e}")
                prompt = self.refine_prompt_template.format(
                    base_prompt=base_prompt,
                    error=str(e),
                    last_attempt=self.last_attempt
                )

        logger.error("Max attempts reached without success")
        raise ValueError("Max attempts reached without success")

# Example usage
if __name__ == "__main__":
    agent = SelfImprovingMontyAgent("config/agent_config.toml")
    task = "Add two numbers x and y, then multiply by 2"
    inputs = {"x": 5, "y": 7}

    # Optional: multi-test validation
    tests = [
        {"inputs": {"x": 5, "y": 7}, "expected": 24},
        {"inputs": {"x": 0, "y": 0}, "expected": 0},
        {"inputs": {"x": -1, "y": 1}, "expected": 0}
    ]

    result = agent.run(task, inputs, tests=tests)
    print("Final output:", result)