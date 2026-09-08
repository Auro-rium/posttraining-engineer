FROM python:3.12-slim

WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY evaluate.py .
ENTRYPOINT ["python", "evaluate.py"]
