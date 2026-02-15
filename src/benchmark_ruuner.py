"""
benchmark_runner.py

Runs benchmark tasks across multiple agent configurations defined in TOML.
Supports parallel execution, token/cost estimation, Logfire tracing, and multi-test validation.
"""

import tomllib
from pathlib import Path
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optional Logfire integration
try:
    import logfire
    LOGFIRE_ENABLED = True
except ImportError:
    LOGFIRE_ENABLED = False
    logger.warning("Logfire not installed — run 'pip install logfire' for tracing")

if LOGFIRE_ENABLED:
    logfire.configure(service_name="monty-benchmark-runner")

# Thread-safe logging lock
log_lock = threading.Lock()


@dataclass
class BenchmarkTask:
    """
    Definition of a single benchmark task with one or more test cases.
    """
    name: str
    description: str
    tests: List[Dict[str, Any]]  # each: {"inputs": dict, "expected": Any}


@dataclass
class AgentConfig:
    """
    Configuration for one agent variant to benchmark.
    """
    name: str
    model: str
    max_attempts: int
    provider: str = "ollama"


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across agent configurations and tasks defined in TOML.
    Features parallel task runs, token/cost estimation, Logfire tracing, and detailed metrics.
    """

    def __init__(self, config_path: str = "config/benchmark_config.toml") -> None:
        """
        Initializes the BenchmarkRunner by loading configuration from TOML.

        Args:
            config_path: Path to the benchmark TOML config file.

        Raises:
            FileNotFoundError: If the config file is missing.
        """
        self.config_path = Path(config_path)
        self._load_config()
        self.results: List[Dict[str, Any]] = []

        if LOGFIRE_ENABLED:
            logfire.info("BenchmarkRunner initialized",
                         config_path=str(self.config_path))

    def _load_config(self) -> None:
        """Loads runner settings, agent configurations, and tasks from TOML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Benchmark config not found: {self.config_path}")

        with open(self.config_path, "rb") as f:
            raw = tomllib.load(f)

        runner = raw.get("runner", {})
        self.timeout = runner.get("timeout_per_task", 300.0)
        self.verbose = runner.get("verbose", True)
        self.output_csv_template = runner.get("output_csv", "benchmark_results_{date}.csv")
        self.max_workers = runner.get("max_workers", 4)

        self.configs: List[AgentConfig] = [
            AgentConfig(**cfg) for cfg in raw.get("configs", [])
        ]

        self.tasks: List[BenchmarkTask] = []
        for t in raw.get("tasks", []):
            tests_raw = t.pop("tests", [])
            tests = []
            for test_case in tests_raw:
                inputs = test_case.pop("inputs", {})
                expected = test_case.pop("expected", None)
                tests.append({"inputs": inputs, "expected": expected})
            self.tasks.append(BenchmarkTask(
                name=t["name"],
                description=t["description"],
                tests=tests
            ))

        if LOGFIRE_ENABLED:
            logfire.info("Loaded benchmark config",
                         num_configs=len(self.configs),
                         num_tasks=len(self.tasks))

    def create_agent(self, cfg: AgentConfig) -> Any:
        """
        Factory method to instantiate an agent for the given configuration.

        Args:
            cfg: Agent configuration.

        Returns:
            Instantiated agent object.
        """
        if LOGFIRE_ENABLED:
            with logfire.span("create_agent", config_name=cfg.name, model=cfg.model):
                from self_improving_monty_agent import SelfImprovingMontyAgent  # Adjust import
                agent = SelfImprovingMontyAgent("config/agent_config.toml")
                agent.config["model"] = cfg.model
                agent.config["max_attempts"] = cfg.max_attempts
                return agent
        else:
            from self_improving_monty_agent import SelfImprovingMontyAgent
            agent = SelfImprovingMontyAgent("config/agent_config.toml")
            agent.config["model"] = cfg.model
            agent.config["max_attempts"] = cfg.max_attempts
            return agent

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation: tokens ≈ words * 1.3.

        Args:
            text: Input text (prompt or code).

        Returns:
            Estimated token count.
        """
        return int(len(text.split()) * 1.3) if text else 0

    def run_single(self, config: AgentConfig, task: BenchmarkTask) -> Dict[str, Any]:
        """
        Executes one task with one agent configuration and returns metrics.

        Args:
            config: Agent configuration.
            task: Task definition (with test cases).

        Returns:
            Dictionary of metrics.
        """
        if LOGFIRE_ENABLED:
            with logfire.span("run_single_task",
                              config_name=config.name,
                              task_name=task.name,
                              description=task.description[:100]):
                return self._run_single_impl(config, task)
        else:
            return self._run_single_impl(config, task)

    def _run_single_impl(self, config: AgentConfig, task: BenchmarkTask) -> Dict[str, Any]:
        """Internal implementation of run_single (without span)."""
        agent = self.create_agent(config)

        start = time.time()
        success = True
        attempts = 0
        all_results = []
        failed_tests_count = 0
        failed_details = []
        error = None

        try:
            for test_case in task.tests:
                prompt = f"{task.description}\nUse inputs: {test_case['inputs']}"  # For token estimation
                test_output = agent.run(
                    task=task.description,
                    inputs=test_case["inputs"],
                    expected_output=test_case["expected"]
                )

                test_result = getattr(test_output, 'result', None) or getattr(test_output, 'stdout', '').strip()
                all_results.append(test_result)

                if test_result != test_case["expected"]:
                    success = False
                    failed_tests_count += 1
                    failed_details.append({
                        "inputs": test_case["inputs"],
                        "expected": test_case["expected"],
                        "got": test_result
                    })

                attempts = max(attempts, getattr(test_output, 'attempts', 1) if hasattr(test_output, 'attempts') else 1)

            duration = time.time() - start

        except Exception as exc:
            duration = time.time() - start
            error = str(exc)
            success = False
            attempts = config.max_attempts

        row = {
            "config_name": config.name,
            "model": config.model,
            "task_name": task.name,
            "success": success,
            "attempts": attempts,
            "time_s": round(duration, 2),
            "failed_tests_count": failed_tests_count,
            "failed_tests": failed_details,
            "error": error or "",
            "prompt_tokens_est": self._estimate_tokens(prompt if 'prompt' in locals() else ""),
            "est_cost_usd": 0.0
        }

        if config.provider != "ollama":
            row["est_cost_usd"] = round((row["prompt_tokens_est"] / 1000) * 0.01, 4)  # Example rate

        return row

    def run_all(self) -> pd.DataFrame:
        """
        Executes all tasks across all configurations in parallel (per config) and returns results.

        Returns:
            pd.DataFrame: Benchmark results with summary stats logged.
        """
        if LOGFIRE_ENABLED:
            with logfire.span("run_all_benchmarks",
                              num_configs=len(self.configs),
                              num_tasks=len(self.tasks)):
                return self._run_all_impl()
        else:
            return self._run_all_impl()

    def _run_all_impl(self) -> pd.DataFrame:
        """Internal parallel implementation."""
        results: List[Dict[str, Any]] = []
        total = len(self.configs) * len(self.tasks)
        count = 0

        for cfg in self.configs:
            logger.info(f"Starting parallel execution for config: {cfg.name}")
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(self.run_single, cfg, task): task
                    for task in self.tasks
                }

                for future in as_completed(future_to_task):
                    count += 1
                    row = future.result()
                    results.append(row)
                    if self.verbose:
                        status = "✓" if row["success"] else f"✗ ({row['failed_tests_count']} failed)"
                        logger.info(f"[{count}/{total}] {cfg.name} → {future_to_task[future].name} {status}")

        df = pd.DataFrame(results)

        summary = df.groupby("config_name").agg(
            success_rate=("success", "mean"),
            avg_attempts=("attempts", "mean"),
            avg_time_s=("time_s", "mean"),
            avg_failed_tests=("failed_tests_count", "mean"),
            avg_prompt_tokens=("prompt_tokens_est", "mean"),
            total_est_cost_usd=("est_cost_usd", "sum"),
            tasks=("success", "count")
        ).round(3)

        logger.info("\nBenchmark Summary:\n" + summary.to_string())

        date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        csv_path = self.output_csv_template.format(date=date_str)
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")

        return df


if __name__ == "__main__":
    runner = BenchmarkRunner("config/benchmark_config.toml")
    results_df = runner.run_all()
    print(results_df.head())