FROM python:3.12-slim

WORKDIR /opt/ml/code
COPY workers/evaluator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# The evaluator imports the audited objective engine, which is shared with the
# objective service but remains isolated from the coordinator process.
COPY app /opt/ml/code/app
COPY prompts /opt/ml/code/prompts
COPY workers/evaluator/evaluate.py /opt/ml/code/evaluate.py
ENTRYPOINT ["python", "evaluate.py"]
