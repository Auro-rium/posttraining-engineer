FROM python:3.12-slim

# Canonical context contract (run from backend/):
#   docker build -f workers/evaluator/Dockerfile .
# The shared app/ and prompts/ trees are intentionally copied from this
# context; a build from workers/evaluator/ fails closed instead of producing
# an image missing the audited objective engine.
WORKDIR /opt/ml/code
COPY workers/evaluator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# The evaluator imports the audited objective engine, which is shared with the
# objective service but remains isolated from the coordinator process.
COPY app /opt/ml/code/app
COPY prompts /opt/ml/code/prompts
COPY workers/evaluator/evaluate.py /opt/ml/code/evaluate.py
ENTRYPOINT ["python", "evaluate.py"]
