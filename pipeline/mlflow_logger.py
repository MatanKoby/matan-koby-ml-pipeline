"""Log one evaluation run to MLflow from a runs/<run-id>/ folder.

Exposes `log_run(...)` (called in-process by pipeline.summarize inside the task image) and a
CLI (used by helpers.log_mlflow_run for host/debug via `uv run`). The tracking URI comes from
MLFLOW_TRACKING_URI (e.g. the docker-compose MLflow server); otherwise mlflow falls back to a
local sqlite/store. Data is read from disk (config.json + metrics.json) as the single source
of truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow

PARAM_KEYS = [
    "run_id", "split", "subset", "workers", "model",
    "task_slice", "cost_limit", "dataset_name",
]


def log_run(run_dir: str | Path, artifact_uri: str = "", experiment: str | None = None) -> None:
    """Log params, metrics, and artifact references for one run to MLflow."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    experiment = experiment or os.environ.get("MLFLOW_EXPERIMENT", "swe-bench-eval")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=config.get("run_id")):
        mlflow.log_params({k: config[k] for k in PARAM_KEYS if k in config})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.set_tag("run_id", config.get("run_id", ""))
        if artifact_uri:
            mlflow.set_tag("artifact_uri", artifact_uri)
        for name in ("config.json", "metrics.json", "manifest.json"):
            artifact = run_dir / name
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
    print(f"Logged MLflow run for run_id={config.get('run_id')} to experiment '{experiment}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a SWE-bench evaluation run to MLflow.")
    parser.add_argument("run_dir", help="Path to runs/<run-id>/")
    parser.add_argument("--artifact-uri", default="", help="Where full artifacts live (local path or S3 URI)")
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()
    log_run(args.run_dir, artifact_uri=args.artifact_uri, experiment=args.experiment)


if __name__ == "__main__":
    main()
