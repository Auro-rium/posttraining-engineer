FROM python:3.12-slim

# Canonical context contract (run from backend/):
#   docker build -f workers/trainer/Dockerfile .
# The shared app/ and prompts/ trees are intentionally copied from this
# context; a build from workers/trainer/ fails closed instead of producing an
# image missing the audited contracts.
WORKDIR /opt/ml/code
COPY workers/trainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# The worker imports the audited objective/autonomous contracts at runtime.
COPY app /opt/ml/code/app
COPY prompts /opt/ml/code/prompts
COPY workers/trainer/train.py /opt/ml/code/train.py
ENTRYPOINT ["python", "train.py"]
