import pytest
from pathlib import Path

from src.monty_ollama_agent import SelfImprovingAgent


def test_clean_code_strip_assistant_and_fences():
    raw = """
Assistant: Here is the code you asked for
```py
# a comment
def foo():
    return 42
```
Some trailing text that should be removed
"""

    cleaned = SelfImprovingAgent.clean_code(raw)
    assert "def foo()" in cleaned
    assert "return 42" in cleaned
    assert "Some trailing text" not in cleaned
    assert "```" not in cleaned


def test_clean_code_preserve_indentation():
    raw = "   def indented():\n       return 1\n"
    cleaned = SelfImprovingAgent.clean_code(raw)
    # indentation should be preserved on the code line
    assert "   def indented()" in cleaned or "def indented()" in cleaned


def test_clean_code_remove_trailing_junk_after_codeblock():
    raw = "```python\nx = 1\n```\nUNWANTED_JUNK"
    cleaned = SelfImprovingAgent.clean_code(raw)
    assert "UNWANTED_JUNK" not in cleaned


def test_load_config_missing_keys(tmp_path: Path):
    cfg = tmp_path / "agent_config.toml"
    cfg.write_bytes(b'max_attempts = 2\n')

    with pytest.raises(ValueError):
        SelfImprovingAgent._load_config(str(cfg))


def test_load_config_file_not_found(tmp_path: Path):
    missing = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError):
        SelfImprovingAgent._load_config(str(missing))
