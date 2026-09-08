FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /srv/backend

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /data/artifacts \
    && chown -R app:app /data/artifacts

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --extra cloud --no-install-project

COPY app ./app
COPY prompts ./prompts
RUN uv sync --frozen --no-dev --extra cloud

ENV PATH="/srv/backend/.venv/bin:${PATH}"
USER app
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
