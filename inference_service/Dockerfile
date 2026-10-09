# Build from the repository root (it needs ../config.yaml):
#   podman build -f inference_service/Dockerfile -t militarytrack-inference .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# OpenCV (pulled in by ultralytics/sahi) needs these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/inference_service

# CPU-only PyTorch first: the default PyPI wheel bundles ~2 GB of CUDA libraries.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY inference_service/requirements.txt .
RUN pip install -r requirements.txt

COPY config.yaml /srv/config.yaml
COPY inference_service/app ./app

RUN useradd --create-home --uid 1000 inference
USER inference

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
