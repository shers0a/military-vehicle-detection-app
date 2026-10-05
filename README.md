# military-vehicle-detection-app

App built for the ROSPIN SATELLITE DATA PROCESSING MASTERCLASS 2025. Works on
RGB satellite imagery using YOLOv8 (via SAHI slicing) with two modes:

1. **Object Detection** — upload an image and detect/count vehicles, sorted into:
   - Small_Military_Vehicle
   - Large_Military_Vehicle
   - Armored_Fighting_Vehicle
   - Civilian_Vehicle
   - Military_Construction_Vehicle
2. **Tactical Map** — upload two images of the same area at different times
   (T0/T1). Returns vehicle density per hectare and a diverging heatmap over a
   50m×50m sector grid: **red (+)** = vehicles arrived, **blue (−)** = vehicles left.

## Architecture (Phase 1)

As of this phase, the app is split in two:

- **`inference_service/`** — Python FastAPI microservice. Owns the YOLOv8 +
  SAHI model and all detection/tactical-map math. Auto-detects CUDA vs CPU
  (previously hardcoded).
- **`src/`** — .NET 10 solution. `MilitaryTrack.Web` (Blazor Server) is the
  UI and calls `MilitaryTrack.Core`'s `InferenceClient` (a typed HttpClient)
  to talk to `inference_service` over HTTP. `MilitaryTrack.Core` has no
  Blazor dependency, so a future CLI can reuse it directly.

The old Streamlit app (`app.py`, `hmp.py`) is preserved under
`legacy/streamlit_app/` for reference; it is not maintained going forward.

Deferred to a later phase: Sentinel Hub integration for wide-area, low-res
(10m/pixel) change/cluster detection — Sentinel-2 cannot resolve individual
vehicles, so that will need a different, coarser model plus a
PostGIS-backed pipeline.

## Running locally

Two processes, run in separate terminals.

### 1. Inference service (Python 3.11 required — SAHI/torch/ultralytics compatibility)

```bash
sudo dnf install python3.11   # or your distro's equivalent / pyenv
cd inference_service
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set MODEL_PATH to your trained best.pt, or place best.pt
# in inference_service/, the project root, or ./runs/detect/rezultat_militar/weights/
uvicorn app.main:app --reload --port 8000
```

Check `http://localhost:8000/api/v1/health` — it should report `model_loaded: true`.

### 2. Web app (.NET 10 SDK)

```bash
dotnet run --project src/MilitaryTrack.Web
```

Open the URL printed in the console (defaults to `http://localhost:5095`).
`src/MilitaryTrack.Web/appsettings.json` points `InferenceService:BaseUrl` at
`http://localhost:8000` — change it if the inference service runs elsewhere.

Upload limit is configured explicitly (see `Upload:MaxFileSizeBytes` in
`appsettings.json`, default 1 GB) rather than relying on framework defaults.

## Training your own model

`antrenare.py` trains a YOLOv8 model — run `python antrenare.py` (unchanged
from before, still plain Python/ultralytics, not part of the web stack).
`config.yaml` is the dataset config (paths + the 5 classes above), and is
also read by `inference_service` for the class list exposed via
`GET /api/v1/config`.

Expected dataset layout:

```
dataset/
├── train/{images,labels}/
└── val/{images,labels}/
```

## Repo layout

```
inference_service/     Python FastAPI microservice (detection + tactical math)
src/MilitaryTrack.Core/  .NET class library: DTOs + InferenceClient (HTTP client)
src/MilitaryTrack.Web/   .NET Blazor Server app (UI)
legacy/streamlit_app/    Old Streamlit app, kept for reference
antrenare.py              YOLO training script
config.yaml                Dataset config (classes, paths)
```
