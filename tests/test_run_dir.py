"""prepare_run_dir + a full offline dry run of the orchestration (everything except the
two heavy Docker/API steps, which are simulated via staged sample/ data)."""

import json

import pipeline.helpers as helpers
from pipeline.helpers import collect_metrics, write_metrics


def test_prepare_run_dir_creates_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "RUNS_DIR", tmp_path / "runs")
    config = helpers.build_run_config({"run_id": "r1", "subset": "verified"})
    run_dir = helpers.prepare_run_dir(config)

    assert run_dir == tmp_path / "runs" / "r1"
    assert (run_dir / "run-agent").is_dir()
    assert (run_dir / "run-eval").is_dir()

    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["run_id"] == "r1"
    assert saved["dataset_name"] == "princeton-nlp/SWE-bench_Verified"


def test_end_to_end_dry_run(staged_run_dir):
    config, run_dir = staged_run_dir

    # Inputs / provenance present.
    assert (run_dir / "config.json").exists()
    assert (run_dir / "run-agent" / "preds.json").exists()
    assert (run_dir / "run-agent" / "astropy__astropy-12907"
            / "astropy__astropy-12907.traj.json").exists()

    # Eval summary present and parses.
    metrics = collect_metrics(run_dir / "run-eval")
    write_metrics(run_dir, metrics)

    assert (run_dir / "metrics.json").exists()
    assert metrics["submitted_instances"] == 3
    assert metrics["resolved_instances"] == 1
