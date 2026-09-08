FROM python:3.12-slim

WORKDIR /opt/ml/code
COPY workers/trainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# The worker imports the audited objective/autonomous contracts at runtime.
COPY app /opt/ml/code/app
COPY prompts /opt/ml/code/prompts
COPY workers/trainer/train.py /opt/ml/code/train.py
ENTRYPOINT ["python", "train.py"]
