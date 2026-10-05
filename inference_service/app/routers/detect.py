import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from .. import detection
from ..config import get_settings
from ..schemas import DetectResponse, Detection

router = APIRouter(tags=["detect"])


@router.post("/detect", response_model=DetectResponse)
async def detect_objects(
    request: Request,
    file: UploadFile = File(...),
    confidence_threshold: float | None = Form(None),
    slice_size: int | None = Form(None),
    overlap_ratio: float | None = Form(None),
):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_error)

    settings = get_settings()
    confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.default_confidence_threshold
    slice_size = slice_size if slice_size is not None else settings.default_slice_size
    overlap_ratio = overlap_ratio if overlap_ratio is not None else settings.default_overlap_ratio

    image = Image.open(file.file).convert("RGB")
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
