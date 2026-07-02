"""Configurable Airflow DAG: run-agent -> run-evaluation via DockerOperator.

Pipeline: prepare_run -> run_agent -> run_eval -> summarize_and_log.

The heavy steps run in the project Docker image (built from Dockerfile). The agent and eval
containers mount the host Docker socket so they can spawn the SWE-bench instance containers
(docker-out-of-docker). Artifacts land in a shared `runs` volume mounted at the same path in
the Airflow workers and the task containers; the summarize step logs to the MLflow service.

Deployed via docker-compose.yaml. Configuration comes from env vars (with sensible defaults):
  TASK_IMAGE       project image tag (default mlops-assignment-task:latest)
  RUNS_VOLUME      named volume backing runs/ (default runs_data)
  RUNS_DIR         mount path for runs/ inside all containers (default /opt/project/runs)
  DOCKER_NETWORK   compose network so summarize can reach the MLflow service
  DOCKER_HOST      Docker daemon URL (default unix://var/run/docker.sock)
Experiment values are NOT hard-coded: they come from the Airflow params below.
"""

import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from pipeline.helpers import (
    build_run_config,
    container_agent_command,
    container_eval_command,
    prepare_run_dir,
)

TASK_IMAGE = os.environ.get("TASK_IMAGE", "mlops-assignment-task:latest")
RUNS_VOLUME = os.environ.get("RUNS_VOLUME", "runs_data")
RUNS_DIR = os.environ.get("RUNS_DIR", "/opt/project/runs")
DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK") or None
DOCKER_URL = os.environ.get("DOCKER_HOST", "unix://var/run/docker.sock")

DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def _mounts():
    """Shared runs volume + host Docker socket (for nested SWE-bench containers)."""
    return [
        Mount(source=RUNS_VOLUME, target=RUNS_DIR, type="volume"),
        Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
    ]


def _docker_task(task_id, command, *, environment=None, network=None, timeout_hours=6):
    return DockerOperator(
        task_id=task_id,
        image=TASK_IMAGE,
        command=["bash", "-lc", command],
        mounts=_mounts(),
        mount_tmp_dir=False,
        docker_url=DOCKER_URL,
        network_mode=network,
        auto_remove="success",
        environment={"RUNS_DIR": RUNS_DIR, **(environment or {})},
        execution_timeout=timedelta(hours=timeout_hours),
    )


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={
        # Required.
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(1, type="integer", minimum=1),
        # Optional but useful (defaults mirror the provided sample; all overridable).
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string"),
        "task_slice": Param("0:3", type="string"),
        "run_id": Param("", type="string"),  # empty -> auto-generated timestamp
        "cost_limit": Param(3.0, type="number", minimum=0),
    },
)
def evaluate_agent():
    @task
    def prepare_run(**context) -> dict:
        config = build_run_config(context["params"])
        run_dir = prepare_run_dir(config)
        return {
            "run_dir": str(run_dir),
            "agent_cmd": container_agent_command(config, run_dir),
            "eval_cmd": container_eval_command(config, run_dir),
        }

    prep = prepare_run()

    run_agent = _docker_task(
        "run_agent",
        "{{ ti.xcom_pull(task_ids='prepare_run')['agent_cmd'] }}",
        environment={
            "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
            "MSWEA_COST_TRACKING": "ignore_errors",
        },
    )

    run_eval = _docker_task(
        "run_eval",
        "{{ ti.xcom_pull(task_ids='prepare_run')['eval_cmd'] }}",
    )

    summarize_and_log = _docker_task(
        "summarize_and_log",
        "python -m pipeline.summarize {{ ti.xcom_pull(task_ids='prepare_run')['run_dir'] }}",
        environment={"MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "")},
        network=DOCKER_NETWORK,
        timeout_hours=1,
    )

    prep >> run_agent >> run_eval >> summarize_and_log


evaluate_agent()
