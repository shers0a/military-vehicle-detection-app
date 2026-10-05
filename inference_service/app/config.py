from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SERVICE_DIR.parent

DEFAULT_MODEL_SEARCH_PATHS = [
    SERVICE_DIR / "best.pt",
    PROJECT_ROOT / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "rezultat_militar" / "weights" / "best.pt",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    model_path: str | None = None
    device: str = "auto"
    model_load_confidence_floor: float = 0.05
    default_confidence_threshold: float = 0.35
    default_slice_size: int = 640
    default_overlap_ratio: float = 0.2
    default_gsd_m_per_px: float = 0.3
    default_grid_size_m: float = 50.0
    dataset_config_path: str = str(PROJECT_ROOT / "config.yaml")

    def resolve_model_path(self) -> Path | None:
        if self.model_path:
            candidate = Path(self.model_path)
            return candidate if candidate.exists() else None
        for candidate in DEFAULT_MODEL_SEARCH_PATHS:
            if candidate.exists():
                return candidate
        return None

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_class_names() -> dict[int, str]:
    settings = get_settings()
    with open(settings.dataset_config_path) as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["names"].items()}
