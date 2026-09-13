FROM python:3.12-slim

# Canonical context contract (run from backend/):
#   docker build -f workers/trainer/Dockerfile .
# The shared app/ and prompts/ trees are intentionally copied from this
# context; a build from workers/trainer/ fails closed instead of producing an
# image missing the audited contracts.
WORKDIR /opt/ml/code
COPY workers/trainer/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout=300 --retries=5 -r /tmp/requirements.txt
# The worker imports the audited objective/autonomous contracts at runtime.
COPY app /opt/ml/code/app
COPY prompts /opt/ml/code/prompts
COPY scripts/stage_functiongemma_checkpoint.py /opt/ml/code/scripts/stage_functiongemma_checkpoint.py
COPY workers/trainer/train.py /opt/ml/code/train.py
COPY workers/trainer/image_smoke.py /opt/ml/code/image_smoke.py
ENTRYPOINT ["python", "train.py"]
