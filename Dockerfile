FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    docker.io \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /mlops-assignment

COPY pyproject.toml .
COPY uv.lock .

# Install runtime deps only (not the root project, which is not a package; no dev tools).
# Pin Python 3.12: mlflow's server is not yet compatible with Python 3.14.
RUN uv sync --locked --no-install-project --no-dev --python 3.12

ENV PATH="/mlops-assignment/.venv/bin:$PATH"
# So `python -m pipeline.summarize` and the pipeline imports resolve inside the container.
ENV PYTHONPATH="/mlops-assignment"

COPY scripts scripts/
COPY pipeline pipeline/

# Optional but useful if your script lacks executable bit or shebang issues:
RUN chmod +x scripts/*.sh
