import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from .. import detection
from ..config import get_settings
from ..schemas import DetectResponse, Detection
from .uploads import read_rgb_image

router = APIRouter(tags=["detect"])


# Plain `def` (not `async def`): inference is blocking CPU/GPU work, so FastAPI must run
# this in its thread pool. Inside `async def` it would freeze the event loop and every
# other request (including /health) would wait until inference finished.
@router.post("/detect", response_model=DetectResponse)
def detect_objects(
    request: Request,
    file: UploadFile = File(...),
    confidence_threshold: float | None = Form(None, ge=0, le=1),
    slice_size: int | None = Form(None, ge=64, le=4096),
    overlap_ratio: float | None = Form(None, ge=0, le=0.9),
):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_error)

    settings = get_settings()
    confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.default_confidence_threshold
    slice_size = slice_size if slice_size is not None else settings.default_slice_size
    overlap_ratio = overlap_ratio if overlap_ratio is not None else settings.default_overlap_ratio

    image = read_rgb_image(file)
    image_np = np.array(image)

    result = detection.run_sliced_detection(image_np, model, slice_size, overlap_ratio)
    predictions = detection.filter_by_confidence(result, confidence_threshold)

    detections = [
        Detection(
            class_name=obj.category.name,
            confidence=obj.score.value,
            bbox=list(obj.bbox.to_xyxy()),
        )
        for obj in predictions
    ]

    return DetectResponse(
        image_width=image.width,
        image_height=image.height,
        total_objects=len(detections),
        counts=detection.get_class_counts(predictions),
        detections=detections,
        model_device=settings.resolve_device(),
        confidence_threshold_applied=confidence_threshold,
    )
