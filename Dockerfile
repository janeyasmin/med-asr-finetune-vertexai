FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --no-cache --no-dev

COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-m", "trainer.task"]
