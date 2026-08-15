FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /trainer
COPY training/requirements.txt ./requirements.txt
RUN python3 -m pip install --break-system-packages -r requirements.txt
COPY training/train.py ./train.py

ENTRYPOINT ["python3", "train.py"]

