import tomllib
from pathlib import Path
import ollama
import pydantic_monty as monty
from loguru import logger
from typing import Any

class SelfImprovingAgent:
    """
    A class-based self-improving AI agent that uses a local LLM (via Ollama) to generate and refine
    Monty-compatible code snippets iteratively. It handles errors from Monty execution and feeds
    them back to the LLM for refinement.

    This agent is config-driven, allowing customization of the LLM model, max attempts, prompt
    templates, and logging settings via a TOML config file.

    Attributes:
        config (dict): Configuration loaded from a TOML file or provided as a dict.
        base_prompt_template (str): Template for the initial LLM prompt.
        refine_prompt_template (str): Template for refinement prompts after errors.
    """

    def __init__(self, config_path: str = "config/agent_config.toml"):
        """
        Initializes the SelfImprovingAgent with configuration from a TOML file.

        Args:
            config_path (str, optional): Path to the TOML config file. Defaults to "agent_config.toml".

        Raises:
            FileNotFoundError: If the config file is not found.
            ValueError: If the config is invalid or missing required keys.
        """
        self.config = self._load_config(config_path)
        # Extract prompt pieces
        self.universal_rules = self.config.get("prompts", {}).get("universal_rules", "")
        self.monty_constraints = self.config.get("prompts", {}).get("monty_constraints", "")

        self.refine_prompt_template = self.config.get("refine_prompt_template", (
            "{base_prompt}\nPrevious attempt failed with: '{error}'. Fix it and output ONLY the corrected code."
        ))
        logger.add(self.config.get("log_file", "agent.log"), level=self.config.get("log_level", "INFO"))

    def _generate_task_guidance(self, task: str, input_keys: list[str]) -> str:
        var_list = ", ".join(f"`{k}`" for k in sorted(input_keys))
        return f"""
Task: {task}

Available input variables: {var_list}
- Use ONLY these variables — do NOT hard-code any values
- Do NOT call input(), eval(), or any I/O functions
- Compute the answer and assign it to variable 'result'

Output ONLY the code.
"""

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """
        Loads the configuration from a TOML file.

        Args:
            config_path (str): Path to the TOML config file.

        Returns:
            dict: The loaded configuration.

        Raises:
            FileNotFoundError: If the file does not exist.
            tomllib.TOMLDecodeError: If the file is not valid TOML.
            ValueError: If required keys are missing.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, "rb") as f:  # TOML requires binary mode
            config = tomllib.load(f)
        
        required_keys = ["model", "max_attempts"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")
        
        logger.info(f"Loaded config from {config_path}")
        return config
    
    @staticmethod
    def clean_code(raw: str) -> str:
        text = raw.strip()

        # Remove everything before the first real code line
        lines = text.splitlines()
        code_lines = []

        in_code = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue  # skip empty

            # Skip common junk prefixes
            if stripped.lower().startswith(('```', 'python', 'py', 'assistant:', 'here is')):
                continue

            # Once we see real code, start collecting
            if stripped and not stripped.startswith('#'):
                in_code = True

            if in_code:
                code_lines.append(line.rstrip())  # preserve indentation

        cleaned = '\n'.join(code_lines).strip()

        # Final safety: remove any trailing junk after last statement
        if '```' in cleaned:
            cleaned = cleaned.split('```')[0].strip()

        return cleaned


    def llm_generate(self, prompt: str) -> str:
        """
        Generates a response from the local LLM using Ollama.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The generated text (code snippet) from the LLM.

        Raises:
            ollama.ResponseError: If there's an issue with the Ollama API call.
        """
        logger.debug(f"Sending prompt to LLM: {prompt[:100]}...")  # Truncate for log brevity
        try:
            response = ollama.chat(
                model=self.config["model"],
                messages=[{'role': 'user', 'content': prompt}]
            )
            generated = response['message']['content'].strip()
            logger.debug(f"LLM response: \n{generated[:100]}...")
            return generated
        except ollama.ResponseError as e:
            logger.error(f"Ollama error: {e}")
            raise




    def run(self, task: str, inputs: dict) -> Any:
        task_guidance = self._generate_task_guidance(task, list(inputs.keys()))
        base_prompt = f"{self.universal_rules}\n\n{self.monty_constraints}\n\n{task_guidance}"
        prompt = base_prompt
        max_attempts = self.config["max_attempts"]

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Starting attempt {attempt}/{max_attempts}")
            
            generated_code = self.llm_generate(prompt)
            generated_code = self.clean_code(generated_code)
            
            print(f"Attempt {attempt}:\n{generated_code}\n")   # debug
            logger.info(f"Generated code:\n{generated_code}")
            
            try:
                m = monty.Monty(generated_code, inputs=list(inputs.keys()))
                output = m.run(inputs=inputs)
                
                logger.success(f"Success on attempt {attempt}!")
                
                if hasattr(output, 'result') and output.result is not None:
                    logger.info(f"Result: {output.result}")
                else:
                    logger.info("No 'result' set — see stdout")
                
                if hasattr(output, 'stdout') and output.stdout.strip():
                    logger.info(f"Stdout: {output.stdout.strip()}")
                else:
                    logger.info("No stdout captured")
                
                return output   # ← now valid, because we're inside a method
                
            except monty.MontyError as e:
                logger.warning(f"Monty error on attempt {attempt}: {e}")
                prompt = self.refine_prompt_template.format(
                    base_prompt=base_prompt, 
                    error=str(e)
                )

        # If loop completes without success
        logger.error("Max attempts reached without success")
        raise ValueError("Max attempts reached")

# Example usage
if __name__ == "__main__":
    # Create a sample config file if it doesn't exist (for demo purposes)
    config_path = "config/agent_config.toml"
    if not Path(config_path).exists():
        sample_config = """
model = "llama3"
max_attempts = 5
log_file = "agent.log"
log_level = "DEBUG"

base_prompt_template = "Write a simple, Monty-compatible Python snippet (no classes, no imports, basic functions/loops/math only) for this task: {task}. Output ONLY the code, no explanations."
refine_prompt_template = "{base_prompt}\\nPrevious attempt failed with: '{error}'. Fix it and output ONLY the corrected code."
"""
        with open(config_path, "w") as f:
            f.write(sample_config)
        logger.info(f"Created sample config at {config_path}")

    agent = SelfImprovingAgent(config_path=config_path)
    task = "Find the roots of the equation a*x^2 + b*x + c"
    inputs = {"a": 5, "b": 7, "c": 10}
    agent.run(task, inputs)