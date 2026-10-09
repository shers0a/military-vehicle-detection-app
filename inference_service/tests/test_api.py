"""API tests that run without real model weights: the model is either absent or replaced
by a fake `run_sliced_detection`, so these exercise request handling, validation and the
response shape, not YOLO itself."""

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import detection
from app.config import get_settings
from app.main import create_app


def fake_prediction(class_name, score, bbox):
    return SimpleNamespace(
        category=SimpleNamespace(name=class_name),
        score=SimpleNamespace(value=score),
        bbox=SimpleNamespace(to_xyxy=lambda: bbox),
    )


def png_bytes(width=200, height=100):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_client(model):
    # Not used as a context manager, so the lifespan (which loads real weights) never runs.
    app = create_app()
    app.state.settings = get_settings()
    app.state.model = model
    app.state.model_error = None if model else "No model weights found (test)"
    return TestClient(app)


@pytest.fixture
def client_without_model():
    return make_client(model=None)


@pytest.fixture
def client_with_fake_model(monkeypatch):
    predictions = [
        fake_prediction("Armored_Fighting_Vehicle", 0.90, [10, 10, 30, 30]),
        fake_prediction("Armored_Fighting_Vehicle", 0.80, [50, 10, 70, 30]),
        fake_prediction("Civilian_Vehicle", 0.60, [100, 50, 120, 70]),
        fake_prediction("Small_Military_Vehicle", 0.10, [150, 50, 160, 60]),  # below default threshold
    ]
    monkeypatch.setattr(
        detection,
        "run_sliced_detection",
        lambda *args, **kwargs: SimpleNamespace(object_prediction_list=predictions),
    )
    return make_client(model=object())


def test_health_reports_degraded_without_model(client_without_model):
    body = client_without_model.get("/api/v1/health").json()

    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_config_exposes_class_list(client_without_model):
    body = client_without_model.get("/api/v1/config").json()

    assert "Armored_Fighting_Vehicle" in body["class_names"]


def test_detect_returns_503_without_model(client_without_model):
    response = client_without_model.post("/api/v1/detect", files={"file": ("a.png", png_bytes())})

    assert response.status_code == 503


@pytest.mark.parametrize(
    "field, value",
    [
        ("confidence_threshold", "1.5"),
        ("confidence_threshold", "-0.1"),
        ("slice_size", "0"),
        ("overlap_ratio", "0.95"),
    ],
)
def test_detect_rejects_out_of_range_parameters(client_without_model, field, value):
    response = client_without_model.post(
        "/api/v1/detect", files={"file": ("a.png", png_bytes())}, data={field: value}
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field


def test_detect_rejects_non_image_upload(client_with_fake_model):
    response = client_with_fake_model.post("/api/v1/detect", files={"file": ("notes.txt", b"not an image")})

    assert response.status_code == 400
    assert "notes.txt" in response.json()["detail"]


def test_detect_filters_by_confidence_and_counts_classes(client_with_fake_model):
    response = client_with_fake_model.post(
        "/api/v1/detect",
        files={"file": ("a.png", png_bytes(200, 100))},
        data={"confidence_threshold": "0.5"},
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["image_width"], body["image_height"]) == (200, 100)
    assert body["total_objects"] == 3
    assert body["counts"] == {"Armored_Fighting_Vehicle": 2, "Civilian_Vehicle": 1}
    assert body["detections"][0]["bbox"] == [10, 10, 30, 30]


def test_tactical_map_rejects_grid_smaller_than_a_pixel(client_without_model):
    response = client_without_model.post(
        "/api/v1/tactical-map",
        files={"file_t0": ("t0.png", png_bytes()), "file_t1": ("t1.png", png_bytes())},
        data={"gsd_m_per_px": "0.5", "grid_size_m": "0.1"},
    )

    assert response.status_code == 422


def test_tactical_map_excludes_civilian_vehicles(client_with_fake_model):
    # Same fake predictions for T0 and T1 -> no change anywhere, and only the
    # two military vehicles above the threshold count as tactical points.
    response = client_with_fake_model.post(
        "/api/v1/tactical-map",
        files={"file_t0": ("t0.png", png_bytes()), "file_t1": ("t1.png", png_bytes())},
        data={"confidence_threshold": "0.5"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["points_t0_count"] == body["points_t1_count"] == 2
    assert not any(any(row) for row in body["diff_matrix"])
    assert body["warnings"] == []


def test_tactical_map_warns_when_image_sizes_differ(client_with_fake_model):
    response = client_with_fake_model.post(
        "/api/v1/tactical-map",
        files={"file_t0": ("t0.png", png_bytes(200, 100)), "file_t1": ("t1.png", png_bytes(300, 100))},
    )

    assert response.status_code == 200
    assert "differ" in response.json()["warnings"][0]


@asynccontextmanager
async def _no_lifespan(app):
    yield


def test_health_stays_responsive_during_inference(monkeypatch):
    # Regression test: with `async def` endpoints, blocking inference froze the event loop
    # and /health waited for the whole detection to finish.
    started = threading.Event()

    def slow_detection(*args, **kwargs):
        started.set()
        time.sleep(2)
        return SimpleNamespace(object_prediction_list=[])

    monkeypatch.setattr(detection, "run_sliced_detection", slow_detection)
    app = create_app()
    app.router.lifespan_context = _no_lifespan
    app.state.settings = get_settings()
    app.state.model = object()
    app.state.model_error = None

    # As a context manager, TestClient serves every request from one event loop, like uvicorn.
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/api/v1/detect", files={"file": ("a.png", png_bytes())})
        assert started.wait(timeout=5)

        start = time.perf_counter()
        assert client.get("/api/v1/health").status_code == 200
        assert time.perf_counter() - start < 1.0

        assert pending.result().status_code == 200
