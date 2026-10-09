# Build from the repository root (it needs ../config.yaml):
#   podman build -f inference_service/Dockerfile -t militarytrack-inference .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=120 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# OpenCV (pulled in by ultralytics/sahi) needs these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/inference_service

# uv instead of pip: pip's resolver backtracks for ages over opencv/numpy versions
# because of the numpy<2 pin, while uv resolves the same requirements in seconds.
RUN pip install --no-cache-dir uv

# CPU-only PyTorch first: the default PyPI wheel bundles ~2 GB of CUDA libraries.
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY inference_service/requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY config.yaml /srv/config.yaml
COPY inference_service/app ./app

RUN useradd --create-home --uid 1000 inference
USER inference

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
