from pathlib import Path
from loguru import logger


def main() -> None:
    config_path = Path("config/agent_config.toml")
    if config_path.exists():
        logger.info("Config already exists: %s", config_path)
        return

    sample_config = """model = "llama3"
max_attempts = 5
log_file = "agent.log"
log_level = "DEBUG"

base_prompt_template = "Write a simple, Monty-compatible Python snippet (no classes, no imports, basic functions/loops/math only) for this task: {task}. Output ONLY the code, no explanations."
refine_prompt_template = "{base_prompt}\\nPrevious attempt failed with: '{error}'. Fix it and output ONLY the corrected code."
"""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(sample_config)
    logger.info("Created sample config at %s", config_path)


if __name__ == "__main__":
    main()
