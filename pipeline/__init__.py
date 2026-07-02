"""Pipeline helpers for the evaluate_agent Airflow DAG.

Kept importable from Airflow's isolated environment: this package's modules that
are imported by the DAG (helpers) use only the standard library at import time.
Anything needing project dependencies (mini-swe-agent, swebench, mlflow) is run
through `uv run` in the project virtualenv.
"""
