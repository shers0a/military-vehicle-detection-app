from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


def read_rgb_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(upload.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"'{upload.filename}' is not a supported image file.")
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail=f"'{upload.filename}' has too many pixels to process safely.")
