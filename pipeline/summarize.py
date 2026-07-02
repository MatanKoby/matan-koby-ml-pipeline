"""Summarize step, run inside the project Docker image by the DockerOperator DAG.

Given a runs/<run-id>/ folder, it parses the eval report, writes metrics.json + manifest.json,
and logs the run to MLflow (MLflow tracking URI from MLFLOW_TRACKING_URI, i.e. the compose
MLflow service). All logic is reused from helpers/mlflow_logger, so there is no duplication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.helpers import build_manifest, collect_metrics, write_manifest, write_metrics
from pipeline.mlflow_logger import log_run


def summarize(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())

    metrics = collect_metrics(run_dir / "run-eval")
    write_metrics(run_dir, metrics)
    manifest = build_manifest(config, run_dir, metrics)
    write_manifest(run_dir, manifest)
    log_run(run_dir, artifact_uri=str(run_dir))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a run and log it to MLflow.")
    parser.add_argument("run_dir", help="Path to runs/<run-id>/")
    args = parser.parse_args()
    summarize(args.run_dir)


if __name__ == "__main__":
    main()
