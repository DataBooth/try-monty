import tomllib
from pathlib import Path

from src.monty_ollama_agent import SelfImprovingAgent


def test_clean_code_basic():
    raw = """
Here is some code:
```python
# comment
result = 1 + 2
print(result)
```
"""
    cleaned = SelfImprovingAgent.clean_code(raw)
    assert "result = 1 + 2" in cleaned
    assert "print(result)" in cleaned


def test_load_config_valid(tmp_path: Path):
    cfg = tmp_path / "agent_config.toml"
    cfg.write_bytes(b'model = "test-model"\nmax_attempts = 3\n')

    loaded = SelfImprovingAgent._load_config(str(cfg))
    assert loaded["model"] == "test-model"
    assert loaded["max_attempts"] == 3
