FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]" \
    && useradd --system --uid 10001 app \
    && mkdir /state && chown app /state

USER app
ENV STATE_DIR=/state PORT=8787
EXPOSE 8787
# /v1/health needs no token and returns 503 when the journal is unreadable.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/v1/health' % os.environ.get('PORT','8787'), timeout=4)"
CMD ["posttrain-api"]
