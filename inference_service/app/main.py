import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .detection import ModelNotAvailableError, load_model
from .routers import detect, meta, tactical

logger = logging.getLogger("inference_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    try:
        app.state.model = load_model(settings)
        app.state.model_error = None
        logger.info("Model loaded on device=%s", settings.resolve_device())
    except ModelNotAvailableError as e:
        app.state.model = None
        app.state.model_error = str(e)
        logger.warning("Model not loaded at startup: %s", e)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Military Vehicle Detection Inference Service", lifespan=lifespan)
    app.include_router(meta.router, prefix="/api/v1")
    app.include_router(detect.router, prefix="/api/v1")
    app.include_router(tactical.router, prefix="/api/v1")
    return app


app = create_app()
