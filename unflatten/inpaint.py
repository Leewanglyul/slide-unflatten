"""Erase text / objects from the background: LaMa when available, OpenCV otherwise."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL import Image

from .imageops import ink_mask
from .types import Line

_lama = None
_lama_failed = False


def text_mask(shape, lines: List[Line], arr: np.ndarray, pad: int = 4) -> np.ndarray:
    """Mask of actual strokes inside each OCR box (erasing whole boxes would also erase frames)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for l in lines:
        x0, y0, x1, y1 = l.box
        x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        x1, y1 = min(x1 + pad, w), min(y1 + pad, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        stroke = ink_mask(arr, (x0, y0, x1, y1)).astype(np.uint8) * 255
        k = max(5, int((y1 - y0) * 0.3)) | 1
        stroke = cv2.dilate(stroke, np.ones((k, k), np.uint8))
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], stroke)
    return mask


def box_mask(shape, boxes, pad: int = 3) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(min(x0, x1) - pad, 0), max(min(y0, y1) - pad, 0)
        x1, y1 = min(max(x0, x1) + pad, w), min(max(y0, y1) + pad, h)
        mask[y0:y1, x0:x1] = 255
    return mask


def _load_lama():
    global _lama, _lama_failed
    if _lama is not None or _lama_failed:
        return _lama
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
        from simple_lama_inpainting.models import model as lama_mod
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj = SimpleLama.__new__(SimpleLama)
        path = lama_mod.download_model(lama_mod.LAMA_MODEL_URL)
        obj.model = torch.jit.load(path, map_location=dev).eval()
        obj.device = dev
        _lama = obj
    except Exception as e:
        _lama_failed = True
        _lama = None
        _load_lama.last_error = e
    return _lama


_load_lama.last_error = None


def inpaint(img: Image.Image, mask: np.ndarray, log=None) -> Image.Image:
    if not mask.any():
        return img
    lama = _load_lama()
    if lama is not None:
        try:
            out = lama(img, Image.fromarray(mask))
            return out.crop((0, 0, img.width, img.height))
        except Exception as e:
            if log:
                log(f"LaMa error → OpenCV: {e}")
    elif log and _load_lama.last_error is not None and not getattr(inpaint, "_warned", False):
        inpaint._warned = True
        log(f"LaMa unavailable → OpenCV inpainting ({_load_lama.last_error})")
    arr = cv2.inpaint(np.array(img.convert("RGB")), mask, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(arr)
