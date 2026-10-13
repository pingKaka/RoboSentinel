from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def decode_jpeg(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid JPEG image")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode JPEG image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
