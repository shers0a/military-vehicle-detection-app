import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from .. import detection, tactical
from ..config import get_settings
from ..schemas import TacticalMapResponse
from .uploads import read_rgb_image

router = APIRouter(tags=["tactical"])


# Plain `def` for the same reason as /detect: blocking inference must run in the thread pool.
@router.post("/tactical-map", response_model=TacticalMapResponse)
def tactical_map(
    request: Request,
    file_t0: UploadFile = File(...),
    file_t1: UploadFile = File(...),
    confidence_threshold: float | None = Form(None, ge=0, le=1),
    gsd_m_per_px: float | None = Form(None, gt=0),
    grid_size_m: float | None = Form(None, gt=0),
):
    settings = get_settings()
    confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.default_confidence_threshold
    gsd_m_per_px = gsd_m_per_px if gsd_m_per_px is not None else settings.default_gsd_m_per_px
    grid_size_m = grid_size_m if grid_size_m is not None else settings.default_grid_size_m

    # A sector smaller than one pixel would explode the number of grid cells.
    if grid_size_m < gsd_m_per_px:
        raise HTTPException(
            status_code=422,
            detail=f"grid_size_m ({grid_size_m}) must be at least one pixel wide (gsd_m_per_px = {gsd_m_per_px}).",
        )

    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_error)

    image_t0 = read_rgb_image(file_t0)
    image_t1 = read_rgb_image(file_t1)

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
