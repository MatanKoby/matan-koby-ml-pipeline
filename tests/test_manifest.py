"""build_manifest + write_manifest: the run folder's self-describing pointer file."""

import json
from pathlib import Path

from pipeline.helpers import build_manifest, collect_metrics, write_manifest, write_metrics


def test_build_manifest_points_to_existing_artifacts(staged_run_dir):
    config, run_dir = staged_run_dir
    metrics = collect_metrics(run_dir / "run-eval")
    write_metrics(run_dir, metrics)

    manifest = build_manifest(config, run_dir, metrics)

    assert manifest["run_id"] == config["run_id"]
    assert manifest["metrics"]["resolved_instances"] == 1

    # Artifact pointers are relative and actually exist in the run folder.
    artifacts = manifest["artifacts"]
    assert artifacts["config"] == "config.json"
    assert artifacts["predictions"] == "run-agent/preds.json"
    assert artifacts["metrics"] == "metrics.json"
    assert artifacts["eval_report"].endswith(".test.json")
    for rel_path in artifacts.values():
        assert not Path(rel_path).is_absolute()
        assert (run_dir / rel_path).exists()

    # Instance trajectories are recorded.
    assert "astropy__astropy-12907" in manifest["instances"]
    assert len(manifest["instances"]) == 3

    # Storage + provenance.
    assert manifest["storage"]["local"] == str(run_dir)
    assert manifest["storage"]["remote_uri"] is None
    assert manifest["provenance"]["model"] == config["model"]
    assert manifest["provenance"]["dataset_name"] == "princeton-nlp/SWE-bench_Verified"


def test_write_manifest_roundtrip(staged_run_dir):
    config, run_dir = staged_run_dir
    metrics = collect_metrics(run_dir / "run-eval")
    manifest = build_manifest(config, run_dir, metrics)

    path = write_manifest(run_dir, manifest)
    assert path == run_dir / "manifest.json"
    assert json.loads(path.read_text())["run_id"] == config["run_id"]
