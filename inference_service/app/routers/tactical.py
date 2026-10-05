import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from .. import detection, tactical
from ..config import get_settings
from ..schemas import TacticalMapResponse

router = APIRouter(tags=["tactical"])


@router.post("/tactical-map", response_model=TacticalMapResponse)
async def tactical_map(
    request: Request,
    file_t0: UploadFile = File(...),
    file_t1: UploadFile = File(...),
    confidence_threshold: float | None = Form(None),
    gsd_m_per_px: float | None = Form(None),
    grid_size_m: float | None = Form(None),
):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_error)

    settings = get_settings()
    confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.default_confidence_threshold
    gsd_m_per_px = gsd_m_per_px if gsd_m_per_px is not None else settings.default_gsd_m_per_px
    grid_size_m = grid_size_m if grid_size_m is not None else settings.default_grid_size_m

    image_t0 = Image.open(file_t0.file).convert("RGB")
    image_t1 = Image.open(file_t1.file).convert("RGB")

    warnings = []
    if image_t0.size != image_t1.size:
        warnings.append(
            f"Image dimensions differ: T0={image_t0.size} T1={image_t1.size}. Results may be inaccurate."
        )

    result_t0 = detection.run_sliced_detection(
        np.array(image_t0), model, settings.default_slice_size, settings.default_overlap_ratio
    )
    result_t1 = detection.run_sliced_detection(
        np.array(image_t1), model, settings.default_slice_size, settings.default_overlap_ratio
    )

    predictions_t0 = detection.filter_by_confidence(result_t0, confidence_threshold)
    predictions_t1 = detection.filter_by_confidence(result_t1, confidence_threshold)

    points_t0 = detection.extract_vehicle_points(predictions_t0)
    points_t1 = detection.extract_vehicle_points(predictions_t1)

    img_w, img_h = image_t0.size
    diff_matrix, bins_x, bins_y = tactical.calculate_tactical_heatmap(
        points_t0, points_t1, img_w, img_h, gsd_m_per_px, grid_size_m
    )
    total_area_ha, density = tactical.calculate_density(len(points_t1), bins_x, bins_y, grid_size_m)

    return TacticalMapResponse(
        image_width=img_w,
        image_height=img_h,
        bins_x=bins_x,
        bins_y=bins_y,
        diff_matrix=diff_matrix.tolist(),
        points_t0_count=len(points_t0),
        points_t1_count=len(points_t1),
        total_area_ha=total_area_ha,
        density_t1_veh_per_ha=density,
        gsd_m_per_px=gsd_m_per_px,
        grid_size_m=grid_size_m,
        warnings=warnings,
    )
