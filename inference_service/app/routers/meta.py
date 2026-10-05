from fastapi import APIRouter, Request

from ..config import get_class_names, get_settings
from ..schemas import ConfigResponse, HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    settings = request.app.state.settings
    model = request.app.state.model
    if model is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            detail=request.app.state.model_error,
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=settings.resolve_device(),
        model_path=str(settings.resolve_model_path()),
    )


@router.get("/config", response_model=ConfigResponse)
def config():
    settings = get_settings()
    return ConfigResponse(
        class_names=list(get_class_names().values()),
        default_confidence_threshold=settings.default_confidence_threshold,
        default_slice_size=settings.default_slice_size,
        default_overlap_ratio=settings.default_overlap_ratio,
        default_gsd_m_per_px=settings.default_gsd_m_per_px,
        default_grid_size_m=settings.default_grid_size_m,
    )
