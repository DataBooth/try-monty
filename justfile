# justfile — Ollama helper recipes
# Run with: just <recipe-name>
# Requires: just (brew install just), ollama (brew install ollama)

set shell := ["bash", "-c"]

# ──────────────────────────────────────────────────────────────────────────────
# Basic commands
# ──────────────────────────────────────────────────────────────────────────────

default:
    @just --list

# Start Ollama as a launch daemon (recommended)
start:
    brew services start ollama

# Stop Ollama service
stop:
    brew services stop ollama

# Restart Ollama service
restart: stop start

# Check if Ollama is running + version
status:
    ollama --version
    ps aux | grep -i ollama | grep -v grep || echo "Ollama not running"

# Show currently running models + memory usage
ps:
    ollama ps

# List all downloaded models
list:
    ollama list

# ──────────────────────────────────────────────────────────────────────────────
# Model management
# ──────────────────────────────────────────────────────────────────────────────

# Pull a small/fast model to test (phi3 / ~2.3 GB)
pull-small:
    ollama pull phi3

# Pull good general-purpose model (~4.7 GB)
pull-llama3:
    ollama pull llama3

# Pull strong code model (~20 GB — good for your 32 GB Mac)
pull-codellama34b:
    ollama pull codellama:34b

# Pull very strong general model (70B Q4 — ~40 GB download, ~25–30 GB RAM)
pull-llama370b:
    ollama pull llama3:70b

# Remove a model to free space (example: remove phi3)
remove model:
    ollama rm {{model}}

# ──────────────────────────────────────────────────────────────────────────────
# Quick interactive chat sessions
# ──────────────────────────────────────────────────────────────────────────────

# Chat with phi3 (fast & tiny)
chat-small:
    ollama run phi3

# Chat with default llama3
chat:
    ollama run llama3

# Chat with codellama 34B (great for code/Monty tasks)
chat-code:
    ollama run codellama:34b

# Chat with llama3 70B (high quality, slower)
chat-large:
    ollama run llama3:70b

# ──────────────────────────────────────────────────────────────────────────────
# Utility / monitoring
# ──────────────────────────────────────────────────────────────────────────────

# Show Ollama logs (useful when debugging)
logs:
    tail -f ~/Library/Logs/ollama.log

# Watch memory & GPU usage while running a model (open in another terminal)
watch-memory:
    watch -n 2 "ps aux | grep ollama | grep -v grep || echo 'Not running'"

# Quick RAM + GPU summary (macOS)
sysinfo:
    system_profiler SPDisplaysDataType | grep -A 10 "Chipset Model"
    vm_stat | grep "Pages"

# Kill any stuck Ollama process (emergency only)
kill:
    pkill -f ollama || echo "No ollama process found"

# ──────────────────────────────────────────────────────────────────────────────
# Aliases / shortcuts you might like
# ──────────────────────────────────────────────────────────────────────────────

up: start
down: stop
ls: list
pull-test: pull-small
test: pull-small chat-small