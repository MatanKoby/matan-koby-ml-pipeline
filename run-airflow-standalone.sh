set -euo pipefail

export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=false
# Put the repo root on PYTHONPATH so DAGs can import the top-level `pipeline` package.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p $AIRFLOW_HOME

echo '{"admin": "admin"}' > $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated

uv tool run apache-airflow standalone
