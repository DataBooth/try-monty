# try-monty

A small example project demonstrating a self-improving agent that uses a local Ollama LLM to generate and refine Monty-compatible Python snippets and execute them via [`pydantic-monty`](https://github.com/pydantic/monty).

## Quick start

- Prerequisites: Python 3.12+, Ollama running locally and accessible to the `ollama` Python client.
- Install dependencies (editable install recommended for development):

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

- Create a sample config (creates `config/agent_config.toml`):

```bash
python examples/create_sample_config.py
```

Edit `config/agent_config.toml` to set `model`, `max_attempts`, and logging settings.

## Usage

Run the example agent from the project root (after creating a config):

```bash
python -m src.monty_ollama_agent
```

Or import and instantiate the agent in your own script:

```python
from src.monty_ollama_agent import SelfImprovingAgent

agent = SelfImprovingAgent(config_path="config/agent_config.toml")
result = agent.run("your task description", {"a": 1, "b": 2})      # This is where the problem to be solve and inputs defined 
```

## Tests

Run the test suite with `pytest`. If you installed the dev extras above you can run directly; otherwise set `PYTHONPATH` so tests can import `src`:

```bash
# If pytest is installed in the venv
pytest -q

# Or, with PYTHONPATH set (works if pytest is in the venv but package not installed)
PYTHONPATH=$(pwd) .venv/bin/pytest -q
```

## Files of interest

- `src/monty_ollama_agent.py` — main agent implementation
- `examples/create_sample_config.py` — utility to create a starter `config/agent_config.toml`
- `config/agent_config.toml` — configuration for model, prompts, and logging
- `tests/` — pytest unit tests (run with `pytest`)

## Notes & recommendations

- The project requires Python >=3.12 (see `pyproject.toml`).
- The module no longer auto-creates a config on import — use `examples/create_sample_config.py` to create a starter file.
- The agent executes generated code via `pydantic-monty`; treat generated code as untrusted and run it in a controlled environment.

## License

See the repository `LICENSE` file.