FROM python:3.12-slim

# Canonical context contract (run from backend/):
#   docker build -f workers/evaluator/Dockerfile .
# The shared app/ and prompts/ trees are intentionally copied from this
# context; a build from workers/evaluator/ fails closed instead of producing
# an image missing the audited objective engine.
WORKDIR /opt/ml/code
COPY workers/evaluator/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout=300 --retries=5 -r /tmp/requirements.txt
# The evaluator imports the audited objective engine, which is shared with the
# objective service but remains isolated from the coordinator process.
COPY app /opt/ml/code/app
COPY prompts /opt/ml/code/prompts
COPY workers/evaluator/evaluate.py /opt/ml/code/evaluate.py
COPY workers/evaluator/image_smoke.py /opt/ml/code/image_smoke.py
ENTRYPOINT ["python", "evaluate.py"]
