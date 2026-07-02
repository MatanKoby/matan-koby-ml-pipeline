"""collect_metrics + write_metrics: parsing the SWE-bench summary report."""

import json
import shutil

import pytest

from pipeline.helpers import collect_metrics, write_metrics


def test_collect_metrics_from_sample(sample_dir, tmp_path):
    eval_dir = tmp_path / "run-eval"
    (eval_dir / "logs").mkdir(parents=True)
    shutil.copy2(
        sample_dir / "nebius__moonshotai__Kimi-K2.6.test.json",
        eval_dir / "nebius__moonshotai__Kimi-K2.6.test.json",
    )
    # A json nested under logs/ must be ignored (glob is top-level only).
    (eval_dir / "logs" / "noise.json").write_text('{"resolved_instances": 999}')

    metrics = collect_metrics(eval_dir)
    assert metrics["total_instances"] == 500
    assert metrics["submitted_instances"] == 3
    assert metrics["resolved_instances"] == 1
    assert metrics["unresolved_instances"] == 2
    assert metrics["resolve_rate"] == pytest.approx(1 / 3)


def test_resolve_rate_zero_when_no_submissions(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"submitted_instances": 0, "resolved_instances": 0})
    )
    assert collect_metrics(tmp_path)["resolve_rate"] == 0.0


def test_missing_summary_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_metrics(tmp_path)


def test_write_metrics_roundtrip(tmp_path):
    path = write_metrics(tmp_path, {"resolved_instances": 1, "resolve_rate": 0.5})
    assert path == tmp_path / "metrics.json"
    assert json.loads(path.read_text())["resolve_rate"] == 0.5
