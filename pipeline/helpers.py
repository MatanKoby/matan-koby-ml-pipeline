"""Helper functions for the evaluate_agent DAG.

Design notes:
- This module imports ONLY the standard library at top level, so Airflow can import the
  DAG regardless of environment. The heavy work runs in the project's Docker image via
  DockerOperator (agent, eval, summarize), where the venv is on PATH.
- The command builders take a `use_uv` flag: on a host with the project venv we prefix
  `uv run`; inside the project image the venv is already on PATH so no prefix is needed.
- RUNS_DIR is overridable via the RUNS_DIR env var so a shared volume (mounted at the same
  path in the Airflow containers and the task containers) can back the run folders.
- The agent config is the builtin packaged one (`--config swebench.yaml`), so the
  pipeline does not depend on the cloned upstream repos (the README states those are
  reference material, not for the final pipeline).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("RUNS_DIR") or (PROJECT_ROOT / "runs"))

# Defaults mirror the provided sample/scripts for convenience. They are only fallbacks:
# every value is overridable via Airflow params, so no experiment value is hard-coded
# into the task bodies.
DEFAULTS = {
    "split": "test",
    "subset": "verified",
    "workers": 1,
    "model": "nebius/moonshotai/Kimi-K2.6",
    "task_slice": "0:3",
    "cost_limit": 3.0,
}

# subset -> SWE-bench dataset name expected by the evaluation harness (--dataset_name).
# Unknown subsets are passed through unchanged (allows a custom dataset path).
EVAL_DATASET_MAPPING = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    "multimodal": "princeton-nlp/SWE-bench_Multimodal",
}


def _dotenv_values(path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE per line). Used to pass NEBIUS_API_KEY to the
    agent subprocess without depending on python-dotenv in the Airflow environment."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def build_run_config(params: dict) -> dict:
    """Turn Airflow params into a fully-resolved run config. Generates a run_id when
    none is provided. No experiment value is hard-coded: params win, DEFAULTS fill gaps."""

    def pick(key):
        val = params.get(key, None)
        if val is None or val == "":
            return DEFAULTS[key]
        return val

    run_id = params.get("run_id") or ""
    run_id = str(run_id).strip()
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    subset = pick("subset")
    config = {
        "run_id": run_id,
        "split": pick("split"),
        "subset": subset,
        "workers": int(pick("workers")),
        "model": pick("model"),
        "task_slice": pick("task_slice"),
        "cost_limit": float(pick("cost_limit")),
        "dataset_name": EVAL_DATASET_MAPPING.get(subset, subset),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return config


def prepare_run_dir(run_config: dict) -> Path:
    """Create runs/<run-id>/ with run-agent/ and run-eval/ subdirs and write config.json."""
    run_dir = RUNS_DIR / run_config["run_id"]
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    return run_dir


# --- Pure command/env builders (extracted so the argv/env can be unit-tested offline,
#     without Docker/API and without mocking subprocess) -------------------------------

def build_agent_command(run_config: dict, run_dir: Path, use_uv: bool = True) -> list[str]:
    """Build the `mini-extra swebench ...` argv for the agent batch run.

    use_uv=True prefixes `uv run` (host with the project venv); use_uv=False assumes the
    venv is already on PATH (inside the project Docker image)."""
    agent_out = Path(run_dir) / "run-agent"
    prefix = ["uv", "run"] if use_uv else []
    return [
        *prefix, "mini-extra", "swebench",
        "--subset", str(run_config["subset"]),
        "--split", str(run_config["split"]),
        "--model", str(run_config["model"]),
        "--slice", str(run_config["task_slice"]),
        "--workers", str(run_config["workers"]),
        # Builtin packaged config (resolved by filename), then cost_limit override.
        "--config", "swebench.yaml",
        "--config", f"agent.cost_limit={run_config['cost_limit']}",
        "-o", str(agent_out),
    ]


def build_agent_env(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Build the environment for the agent run: current env + .env values + cost tracking flag."""
    return {
        **os.environ,
        **_dotenv_values(Path(project_root) / ".env"),
        "MSWEA_COST_TRACKING": "ignore_errors",
    }


def build_eval_command(run_config: dict, preds_path: Path, use_uv: bool = True) -> list[str]:
    """Build the `python -m swebench.harness.run_evaluation ...` argv.

    use_uv=True prefixes `uv run` (host); use_uv=False runs the venv `python` on PATH
    (inside the project Docker image)."""
    prefix = ["uv", "run"] if use_uv else []
    return [
        *prefix, "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", str(run_config["dataset_name"]),
        "--predictions_path", str(Path(preds_path).resolve()),
        "--max_workers", str(run_config["workers"]),
        "--run_id", str(run_config["run_id"]),
    ]


def container_agent_command(run_config: dict, run_dir: Path) -> str:
    """Agent command as a single shell string for the DockerOperator (no `uv run`).
    Output goes to an absolute -o path, so no cwd change is needed."""
    return " ".join(build_agent_command(run_config, run_dir, use_uv=False))


def container_eval_command(run_config: dict, run_dir: Path) -> str:
    """Eval command as a single shell string for the DockerOperator (no `uv run`).

    The SWE-bench harness writes logs/ and the summary relative to its CWD, so we cd into
    run-eval/ first."""
    run_dir = Path(run_dir)
    eval_dir = run_dir / "run-eval"
    preds_path = run_dir / "run-agent" / "preds.json"
    cmd = " ".join(build_eval_command(run_config, preds_path, use_uv=False))
    return f"cd {eval_dir} && {cmd}"


def run_agent_batch(run_config: dict, run_dir: Path, use_uv: bool = True) -> Path:
    """Run mini-swe-agent in batch mode into runs/<run-id>/run-agent/. Returns preds.json path.

    Host/debug utility (the production DAG uses DockerOperator). use_uv controls whether the
    command is prefixed with `uv run`."""
    agent_out = Path(run_dir) / "run-agent"
    subprocess.run(
        build_agent_command(run_config, run_dir, use_uv=use_uv),
        cwd=PROJECT_ROOT,
        env=build_agent_env(),
        check=True,
    )
    return agent_out / "preds.json"


def run_swebench_eval(run_config: dict, preds_path: Path, run_dir: Path, use_uv: bool = True) -> Path:
    """Evaluate preds.json with the SWE-bench harness into runs/<run-id>/run-eval/.

    Host/debug utility (the production DAG uses DockerOperator). The harness writes logs/ and
    the <model>.<run_id>.json summary relative to its CWD, so we run it with cwd=run-eval/."""
    eval_dir = Path(run_dir) / "run-eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_eval_command(run_config, preds_path, use_uv=use_uv),
        cwd=eval_dir,
        env={**os.environ},
        check=True,
    )
    return eval_dir


def collect_metrics(eval_dir: Path) -> dict:
    """Parse the SWE-bench summary report (<model>.<run_id>.json at the top of run-eval/)
    and return metrics, including a computed resolve_rate."""
    summaries = sorted(p for p in Path(eval_dir).glob("*.json"))
    if not summaries:
        raise FileNotFoundError(f"No SWE-bench summary report (*.json) found in {eval_dir}")
    summary = json.loads(summaries[0].read_text())

    keys = [
        "total_instances", "submitted_instances", "completed_instances",
        "resolved_instances", "unresolved_instances", "empty_patch_instances",
        "error_instances",
    ]
    metrics = {k: summary.get(k, 0) for k in keys}
    submitted = metrics.get("submitted_instances", 0) or 0
    metrics["resolve_rate"] = (metrics["resolved_instances"] / submitted) if submitted else 0.0
    return metrics


def write_metrics(run_dir: Path, metrics: dict) -> Path:
    """Write metrics.json into the run folder. Returns its path."""
    metrics_path = Path(run_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics_path


def _git_commit() -> str | None:
    """Best-effort current git commit of the pipeline repo, for provenance."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_manifest(run_config: dict, run_dir: Path, metrics: dict) -> dict:
    """Build a manifest that points to the important files in the run folder and records
    provenance + where the full artifacts live. Paths are relative to run_dir so the folder
    is self-describing and portable ('send a directory to someone')."""
    run_dir = Path(run_dir)
    artifacts: dict[str, str] = {}

    def add(key: str, rel_path: str) -> None:
        if (run_dir / rel_path).exists():
            artifacts[key] = rel_path

    add("config", "config.json")
    add("predictions", "run-agent/preds.json")
    add("agent_log", "run-agent/minisweagent.log")
    add("eval_logs", "run-eval/logs")
    add("metrics", "metrics.json")

    eval_reports = sorted((run_dir / "run-eval").glob("*.json"))
    if eval_reports:
        artifacts["eval_report"] = str(eval_reports[0].relative_to(run_dir))

    agent_dir = run_dir / "run-agent"
    instances = sorted(p.name for p in agent_dir.iterdir() if p.is_dir()) if agent_dir.exists() else []

    return {
        "run_id": run_config["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "metrics": metrics,
        "artifacts": artifacts,
        "instances": instances,
        "storage": {"local": str(run_dir), "remote_uri": None},
        "provenance": {
            "git_commit": _git_commit(),
            "model": run_config.get("model"),
            "dataset_name": run_config.get("dataset_name"),
            "subset": run_config.get("subset"),
            "split": run_config.get("split"),
            "task_slice": run_config.get("task_slice"),
        },
    }


def write_manifest(run_dir: Path, manifest: dict) -> Path:
    """Write manifest.json into the run folder. Returns its path."""
    manifest_path = Path(run_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def log_mlflow_run(run_config: dict, metrics: dict, artifact_uri: str) -> None:
    """Log params/metrics/artifact reference to MLflow.

    Runs through `uv run pipeline/mlflow_logger.py` because the isolated Airflow
    environment has no mlflow. The runner reads config.json and metrics.json from the
    run folder, so the data is sourced from disk (single source of truth)."""
    run_dir = RUNS_DIR / run_config["run_id"]
    cmd = [
        "uv", "run", "python", str(PROJECT_ROOT / "pipeline" / "mlflow_logger.py"),
        str(run_dir),
        "--artifact-uri", str(artifact_uri),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env={**os.environ}, check=True)
