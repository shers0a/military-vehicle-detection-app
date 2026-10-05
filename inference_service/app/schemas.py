from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    model_path: str | None = None
    detail: str | None = None


class ConfigResponse(BaseModel):
    class_names: list[str]
    default_confidence_threshold: float
    default_slice_size: int
    default_overlap_ratio: float
    default_gsd_m_per_px: float
    default_grid_size_m: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float] = Field(description="[x_min, y_min, x_max, y_max] in original image pixels")


class DetectResponse(BaseModel):
    image_width: int
    image_height: int
    total_objects: int
    counts: dict[str, int]
    detections: list[Detection]
    model_device: str
    confidence_threshold_applied: float


class TacticalMapResponse(BaseModel):
    image_width: int
    image_height: int
    bins_x: int
    bins_y: int
    diff_matrix: list[list[float]]
    points_t0_count: int
    points_t1_count: int
    total_area_ha: float
    density_t1_veh_per_ha: float
    gsd_m_per_px: float
    grid_size_m: float
    warnings: list[str] = []
