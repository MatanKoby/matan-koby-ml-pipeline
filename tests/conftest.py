"""Shared fixtures for the offline test suite.

These tests validate everything up to the subprocess boundary (pure logic, command
construction, run-folder layout, metrics parsing, MLflow logging, DAG parsing). The two
heavy steps (real agent inference, real SWE-bench Docker eval) are only run for real on
the VM; here their *command construction* is tested and their *outputs* are simulated with
the checked-in sample/ data.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "sample"
sys.path.insert(0, str(REPO))  # make the top-level `pipeline` package importable


@pytest.fixture
def sample_dir() -> Path:
    return SAMPLE


@pytest.fixture
def base_config() -> dict:
    """A fully-resolved run config matching the provided sample."""
    from pipeline.helpers import build_run_config

    return build_run_config({
        "split": "test", "subset": "verified", "workers": 5,
        "model": "nebius/moonshotai/Kimi-K2.6", "task_slice": "0:3",
        "run_id": "test-run", "cost_limit": 3.0,
    })


@pytest.fixture
def staged_run_dir(tmp_path, monkeypatch, sample_dir):
    """A runs/<id>/ folder with sample/ data staged into run-agent/ and run-eval/,
    i.e. the state after a (simulated) agent + eval run. Returns (config, run_dir)."""
    import pipeline.helpers as helpers

    monkeypatch.setattr(helpers, "RUNS_DIR", tmp_path / "runs")
    config = helpers.build_run_config({
        "split": "test", "subset": "verified", "workers": 5,
        "model": "nebius/moonshotai/Kimi-K2.6", "task_slice": "0:3",
        "run_id": "test-run", "cost_limit": 3.0,
    })
    run_dir = helpers.prepare_run_dir(config)

    # Simulate run_agent output: trajectories + preds.json into run-agent/.
    for item in (sample_dir / "trajectories").iterdir():
        dst = run_dir / "run-agent" / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    # Simulate run_eval output: summary json at top of run-eval/, logs/ beneath.
    shutil.copy2(
        sample_dir / "nebius__moonshotai__Kimi-K2.6.test.json",
        run_dir / "run-eval" / "nebius__moonshotai__Kimi-K2.6.test.json",
    )
    shutil.copytree(sample_dir / "logs", run_dir / "run-eval" / "logs", dirs_exist_ok=True)
    return config, run_dir
