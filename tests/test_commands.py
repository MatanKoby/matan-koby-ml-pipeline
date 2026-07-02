"""Command/env construction: locks the Airflow-param -> CLI-flag mapping without needing
Docker or the API. This is the boundary the real agent/eval runs cross on the VM."""

from pathlib import Path

from pipeline.helpers import (
    build_agent_command,
    build_agent_env,
    build_eval_command,
    container_agent_command,
    container_eval_command,
)


def test_build_agent_command(base_config):
    run_dir = Path("/x/runs/test-run")
    cmd = build_agent_command(base_config, run_dir)
    assert cmd == [
        "uv", "run", "mini-extra", "swebench",
        "--subset", "verified",
        "--split", "test",
        "--model", "nebius/moonshotai/Kimi-K2.6",
        "--slice", "0:3",
        "--workers", "5",
        "--config", "swebench.yaml",
        "--config", "agent.cost_limit=3.0",
        "-o", str(run_dir / "run-agent"),
    ]


def test_build_eval_command(base_config, tmp_path):
    preds = tmp_path / "preds.json"
    preds.write_text("{}")
    cmd = build_eval_command(base_config, preds)
    assert cmd == [
        "uv", "run", "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--predictions_path", str(preds.resolve()),
        "--max_workers", "5",
        "--run_id", "test-run",
    ]


def test_eval_predictions_path_is_absolute(base_config, tmp_path, monkeypatch):
    # Even given a relative preds path, the command must resolve it (eval runs with a
    # different cwd, so a relative path would break).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "preds.json").write_text("{}")
    cmd = build_eval_command(base_config, Path("preds.json"))
    idx = cmd.index("--predictions_path")
    assert Path(cmd[idx + 1]).is_absolute()


def test_build_agent_env(tmp_path):
    (tmp_path / ".env").write_text("NEBIUS_API_KEY=secret-key\n")
    env = build_agent_env(tmp_path)
    assert env["NEBIUS_API_KEY"] == "secret-key"
    assert env["MSWEA_COST_TRACKING"] == "ignore_errors"


def test_use_uv_false_drops_uv_run_prefix(base_config, tmp_path):
    # Inside the project image the venv is on PATH, so no `uv run` prefix.
    agent = build_agent_command(base_config, tmp_path, use_uv=False)
    assert agent[:2] == ["mini-extra", "swebench"]
    eval_cmd = build_eval_command(base_config, tmp_path / "preds.json", use_uv=False)
    assert eval_cmd[:3] == ["python", "-m", "swebench.harness.run_evaluation"]


def test_container_agent_command(base_config, tmp_path):
    cmd = container_agent_command(base_config, tmp_path)
    assert cmd.startswith("mini-extra swebench ")
    assert "--config swebench.yaml --config agent.cost_limit=3.0" in cmd
    assert cmd.endswith(f"-o {tmp_path / 'run-agent'}")


def test_container_eval_command_cds_into_run_eval(base_config, tmp_path):
    cmd = container_eval_command(base_config, tmp_path)
    # Must cd into run-eval/ (harness writes relative to CWD) and reference absolute preds.
    assert cmd.startswith(f"cd {tmp_path / 'run-eval'} && python -m swebench.harness.run_evaluation")
    assert str(tmp_path / "run-agent" / "preds.json") in cmd
    assert "--run_id test-run" in cmd
