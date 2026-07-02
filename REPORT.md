# REPORT: Evaluation pipeline for coding-agent experiments

An Airflow pipeline that runs [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
on a subset of SWE-bench, evaluates the patches with the
[SWE-bench](https://github.com/swe-bench/SWE-bench) harness, writes a reproducible
`runs/<run-id>/` folder, and logs parameters, metrics, and artifact references to MLflow.

## Architecture

```
Airflow DAG: evaluate_agent
  prepare_run -> run_agent -> run_eval -> summarize_and_log
    (PythonOp)   (DockerOp)   (DockerOp)   (DockerOp)
```

- **prepare_run** (Airflow worker): reads Airflow params, builds a resolved run config,
  creates `runs/<run-id>/` with `config.json`, and returns the container commands.
- **run_agent** (DockerOperator, project image): runs `mini-extra swebench ...` writing
  trajectories + `preds.json` into `runs/<run-id>/run-agent/`.
- **run_eval** (DockerOperator, project image): runs the SWE-bench harness on `preds.json`,
  writing logs + the summary report into `runs/<run-id>/run-eval/`.
- **summarize_and_log** (DockerOperator, project image): parses the report into
  `metrics.json`, writes `manifest.json`, and logs params/metrics/artifacts to MLflow.

The agent and eval containers mount the host Docker socket so they can spawn the SWE-bench
instance containers (docker-out-of-docker). All run artifacts land in a shared `runs` volume
mounted at the same path (`/opt/project/runs`) in the Airflow workers and the task
containers, so every step sees the same run folder.

Deployment is via `docker compose` (see "How to run"): it brings up Airflow
(CeleryExecutor + Postgres + Redis) and MLflow, and the `evaluate_agent` DAG runs its heavy
steps with `DockerOperator` using the project image. The provided `run-airflow-standalone.sh`
is a quick-start for the bundled example DAG only; the DockerOperator pipeline needs the
compose stack.

## Repository layout

| Path | Purpose |
|---|---|
| `dags/evaluate_agent.py` | The configurable DAG (DockerOperator). |
| `pipeline/helpers.py` | Config, run-dir, command builders, metrics, manifest. |
| `pipeline/summarize.py` | Container entrypoint for the summarize step. |
| `pipeline/mlflow_logger.py` | MLflow logging (`log_run` + CLI). |
| `Dockerfile` | Project task image (agent, eval, summarize, MLflow server). |
| `Dockerfile.airflow` | Airflow image + Docker provider. |
| `docker-compose.yaml` | Airflow + MLflow deployment. |
| `tests/` | Offline test suite (see below). |
| `runs/<run-id>/` | Per-run artifacts (gitignored in general; the real run `runs/20260702-160657/` is committed as a worked example). |

## Configuration (Airflow params)

| Param | Required | Default | Maps to |
|---|---|---|---|
| `split` | yes | `test` | `--split` |
| `subset` | yes | `verified` | `--subset` + eval `--dataset_name` |
| `workers` | yes | `1` | `--workers` / `--max_workers` |
| `model` | | `nebius/moonshotai/Kimi-K2.6` | `--model` |
| `task_slice` | | `0:3` | `--slice` |
| `run_id` | | auto (UTC timestamp) | run folder name + eval `--run_id` |
| `cost_limit` | | `3.0` | `-c agent.cost_limit=<v>` |

No experiment value is hard-coded in the task bodies; defaults are param defaults only.

## How to run (production / compose)

```bash
cp .env.example .env         # then fill NEBIUS_API_KEY
# set the two host-specific values:
#   AIRFLOW_UID -> `id -u`
#   DOCKER_GID  -> `getent group docker | cut -d: -f3`   (so Airflow can use the Docker socket)
docker compose build         # builds the Airflow image and the task image (tagged mlops-assignment-task:latest)
docker compose up -d
```

- Airflow UI: http://localhost:8080 (user/pass from `.env`, default `airflow`/`airflow`).
- MLflow UI: http://localhost:5000.
- Trigger `evaluate_agent` from the UI ("Trigger DAG w/ config") and set params, or accept
  defaults for a small `0:3` run.

## Artifact layout

```
runs/<run-id>/
  config.json          # resolved run configuration
  run-agent/
    preds.json         # instance_id -> {model_name_or_path, instance_id, model_patch}
    <instance>/<instance>.traj.json   # one trajectory per instance
    minisweagent.log
  run-eval/
    <model>.<run_id>.json             # SWE-bench summary report (metrics source)
    logs/run_evaluation/<run_id>/...  # per-instance report.json, test output, patch, logs
  metrics.json         # total/submitted/resolved/... + resolve_rate
  manifest.json        # relative pointers to every artifact + provenance + storage
```

`manifest.json` makes the folder self-describing: it lists the instances, points to each key
file with paths relative to the run folder, and records provenance (git commit, model,
dataset, subset/split/slice) and `storage` (local path + `remote_uri`). You can hand someone
the folder and they can reconstruct the whole run.

## MLflow

`summarize_and_log` logs, per run: params (`split`, `subset`, `workers`, `model`,
`task_slice`, `cost_limit`, `run_id`, `dataset_name`), metrics (`resolve_rate`,
`resolved_instances`, `submitted_instances`, and so on), the `run_id`/`artifact_uri` tags, and the
`config.json`/`metrics.json`/`manifest.json` artifacts. Compare runs in the MLflow UI.

## Rerun by run-id

- The run folder name **is** the `run_id`. Re-triggering with the same `run_id` reuses the
  folder; the agent skips instances already present in `preds.json`.
- To reproduce someone else's run, read `manifest.json` (`run_config` + `provenance.git_commit`),
  check out that commit, and trigger with the same params/`run_id`.

## Object Storage (S3): how it would be uploaded

Remote storage is **extra credit** in the rubric and is **not** implemented here (no bucket was
provisioned). The pipeline instead writes a complete local `runs/<run-id>/` and records
`storage.remote_uri: null` in the manifest. To enable uploads:

1. **Provision** a Nebius Object Storage bucket (S3-compatible): create a bucket
   (e.g. `swe-bench-runs`), create a service account with storage access, and generate a
   **static access key** (access key id + secret). Note the region S3 endpoint URL.
2. **Configure** `.env` (template already present): `S3_ENDPOINT_URL`, `S3_BUCKET`,
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
3. **Add** an `upload_artifacts` DockerOperator step between `run_eval` and
   `summarize_and_log` that mirrors `runs/<run-id>/` to `s3://$S3_BUCKET/runs/<run-id>/` with
   `boto3` (S3-compatible client using `endpoint_url=$S3_ENDPOINT_URL`), then sets
   `manifest.json`'s `storage.remote_uri` to the `s3://...` URI and tags the MLflow run with it.

Sketch:

```python
import boto3
s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
for path in run_dir.rglob("*"):
    if path.is_file():
        s3.upload_file(str(path), bucket, f"runs/{run_id}/{path.relative_to(run_dir)}")
```

## Tests

`uv run pytest` runs an offline suite that validates everything up to the container boundary:
config resolution, `.env` parsing, command/env construction (the param-to-CLI flag mapping),
`runs/<id>/` layout, metrics parsing, manifest building, real MLflow logging (sqlite
backend), and DAG parsing (via the Airflow tool env with the Docker provider). The two heavy
steps (real agent inference, SWE-bench Docker eval) are validated by a real run on the VM.

## One completed run

Ran end-to-end on a Nebius VM (8 CPU / 32 GB) via `docker compose` + DockerOperator.

- **run_id:** `20260702-160657`
- **Params:** `split=test`, `subset=verified`, `workers=1`, `model=nebius/moonshotai/Kimi-K2.6`, `task_slice=0:3`, `cost_limit=3.0`, `dataset=princeton-nlp/SWE-bench_Verified`
- **Result:** submitted 3, **resolved 2**, unresolved 1, **`resolve_rate = 0.667`**
- **Airflow:** all four tasks green (`prepare_run -> run_agent -> run_eval -> summarize_and_log`). See `screenshots/airflow_dag.png`.
- **MLflow:** logged to experiment `swe-bench-eval` with the params/metrics above. See `screenshots/mlflow_runs.png`.
- **Artifacts:** the full `runs/20260702-160657/` tree is committed to the repo (config, preds, 3 trajectories, eval report + logs, `metrics.json`, `manifest.json`), so the run is reconstructable directly from the repo.

Reproduce: trigger `evaluate_agent` with the same params (or set `run_id`), see "How to run" above.

## Notes / caveats

- **Docker socket permissions:** the Airflow worker needs the host `docker` group GID
  (`DOCKER_GID`) to use the mounted socket for DockerOperator. This is host-specific.
- **Resources:** the SWE-bench Docker images are multi-GB per instance; use the 8 CPU /
  32 GB VM (the assignment's prerequisite) with comfortable disk headroom.
- **MLflow backend:** MLflow 3.x requires a database backend (the compose server uses
  sqlite; the plain `./mlruns` file store is deprecated).
