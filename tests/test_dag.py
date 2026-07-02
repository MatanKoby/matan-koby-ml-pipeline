"""DAG parsing: the canonical Airflow check (DagBag) run in the apache-airflow tool env.

Airflow lives in an isolated `uv tool run` environment, not the project venv, so this test
shells out to it. It is marked `airflow` and skips cleanly if that env is unavailable."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_DAGBAG_CHECK = """
from airflow.models.dagbag import DagBag
db = DagBag("dags", include_examples=False)
print("IMPORT_ERRORS:", dict(db.import_errors))
d = db.dags.get("evaluate_agent")
print("TASKS:", sorted(t.task_id for t in d.tasks) if d else None)
print("PARAMS:", sorted(d.params.keys()) if d else None)
"""


@pytest.mark.airflow
def test_dag_parses_without_import_errors():
    proc = subprocess.run(
        [
            "uv", "tool", "run", "--from", "apache-airflow",
            "--with", "apache-airflow-providers-docker",
            "python", "-c", _DAGBAG_CHECK,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    out = proc.stdout
    if "IMPORT_ERRORS:" not in out:
        pytest.skip(f"apache-airflow tool env unavailable:\n{proc.stderr[-800:]}")

    assert "IMPORT_ERRORS: {}" in out, f"DAG import errors found:\n{out}\n{proc.stderr[-1500:]}"
    assert "'prepare_run'" in out and "'run_agent'" in out
    assert "'run_eval'" in out and "'summarize_and_log'" in out
    assert "'split'" in out and "'subset'" in out and "'workers'" in out
