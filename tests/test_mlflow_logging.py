"""mlflow_logger: real MLflow logging against a local file store (no server needed)."""

import json
import sys

import pytest


def test_mlflow_logger_logs_params_and_metrics(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "r1", "split": "test", "subset": "verified", "workers": 5,
        "model": "nebius/moonshotai/Kimi-K2.6", "task_slice": "0:3",
        "cost_limit": 3.0, "dataset_name": "princeton-nlp/SWE-bench_Verified",
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "resolved_instances": 1, "submitted_instances": 3, "resolve_rate": 0.3333,
    }))

    # MLflow 3.x rejects the plain file store (./mlruns); use a sqlite backend, which is
    # what the pipeline uses by default. chdir keeps artifacts inside tmp_path.
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    mlflow.set_tracking_uri(tracking)
    monkeypatch.chdir(tmp_path)

    import pipeline.mlflow_logger as logger
    monkeypatch.setattr(sys, "argv", [
        "mlflow_logger", str(run_dir), "--artifact-uri", str(run_dir),
        "--experiment", "pytest-exp",
    ])
    logger.main()

    exp = mlflow.get_experiment_by_name("pytest-exp")
    assert exp is not None
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.run_id"] == "r1"
    assert row["params.model"] == "nebius/moonshotai/Kimi-K2.6"
    assert float(row["metrics.resolve_rate"]) == pytest.approx(0.3333)
    assert row["tags.artifact_uri"] == str(run_dir)
