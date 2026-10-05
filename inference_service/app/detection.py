import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from .config import Settings

EXCLUDED_FROM_TACTICAL_POINTS = {"Civilian_Vehicle"}


class ModelNotAvailableError(RuntimeError):
    pass


def load_model(settings: Settings) -> AutoDetectionModel:
    model_path = settings.resolve_model_path()
    if model_path is None:
        raise ModelNotAvailableError(
            "No model weights found. Set MODEL_PATH or place best.pt in the "
            "inference_service directory, the project root, or "
            "./runs/detect/rezultat_militar/weights/best.pt"
        )

    device = settings.resolve_device()
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(model_path),
        confidence_threshold=settings.model_load_confidence_floor,
        device=device,
    )


def run_sliced_detection(image_np: np.ndarray, model: AutoDetectionModel, slice_size: int, overlap_ratio: float):
    return get_sliced_prediction(
        image_np,
        model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )


def filter_by_confidence(result, confidence_threshold: float):
    return [
        obj
        for obj in result.object_prediction_list
        if obj.score.value >= confidence_threshold
    ]


def get_class_counts(predictions) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in predictions:
        counts[obj.category.name] = counts.get(obj.category.name, 0) + 1
    return counts


def extract_vehicle_points(predictions, exclude_classes: set[str] = EXCLUDED_FROM_TACTICAL_POINTS) -> np.ndarray:
    coord = []
    for obj in predictions:
        if obj.category.name not in exclude_classes:
            bbox = obj.bbox.to_xyxy()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            coord.append([center_x, center_y])

    if not coord:
        return np.empty((0, 2))
    return np.array(coord)
