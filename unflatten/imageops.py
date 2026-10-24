"""Small pixel helpers shared by OCR, layout and segmentation."""
from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def ink_mask(arr: np.ndarray, box) -> np.ndarray:
    """Treat the box border colour as background; True where pixels differ (strokes)."""
    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((max(0, y1 - y0), max(0, x1 - x0)), bool)
    edge = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]]).astype(np.int16)
    bg = np.median(edge, axis=0)
    return np.abs(patch.astype(np.int16) - bg).sum(axis=2) > 60


def ink_box(arr: np.ndarray, box):
    """Shrink a loose OCR box to the pixels that actually contain strokes."""
    h, w = arr.shape[:2]
    x0, y0 = max(box[0] - 2, 0), max(box[1] - 2, 0)
    x1, y1 = min(box[2] + 2, w), min(box[3] + 2, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return (x0, y0, x1, y1)
    m = ink_mask(arr, (x0, y0, x1, y1))
    rows = np.where(m.sum(axis=1) > max(1, 0.01 * m.shape[1]))[0]
    cols = np.where(m.sum(axis=0) > 0)[0]
    if len(rows) < 3 or len(cols) < 3:
        return (x0, y0, x1, y1)
    return (x0 + int(cols[0]), y0 + int(rows[0]), x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1)


def text_color(arr: np.ndarray, box) -> Tuple[int, int, int]:
    """Dominant stroke colour inside a text box (2-means; the smaller cluster is the text)."""
    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(patch) < 10:
        return (0, 0, 0)
    _, labels, centers = cv2.kmeans(patch, 2, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=2)
    fg = int(np.argmin(counts))
    return tuple(int(c) for c in centers[fg])


def border_lab(lab: np.ndarray) -> np.ndarray:
    border = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3),
                             lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])
    return np.median(border, axis=0)


def classify_background(img: Image.Image) -> str:
    """flat | gradient | photo — decides whether contour-based object splitting makes sense."""
    small = np.array(img.convert("RGB").resize((160, 90), Image.BILINEAR)).astype(np.float32)
    lab = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = border_lab(lab)
    dist = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    frac_fg = float((dist > 18).mean())
    gray = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges > 0).mean())
    if frac_fg > 0.75 and edge_density > 0.12:
        return "photo"
    if frac_fg > 0.75:
        return "gradient"
    return "flat"


def dominant_colors(images, k: int = 6):
    """k dominant non-grey colours across the deck (for the theme colour scheme)."""
    px = []
    for im in images[:12]:
        px.append(np.array(im.convert("RGB").resize((96, 54))).reshape(-1, 3))
    if not px:
        return []
    data = np.concatenate(px).astype(np.float32)
    sat = data.max(axis=1) - data.min(axis=1)
    data = data[sat > 40] if (sat > 40).sum() > 200 else data
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0),
                                    2, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.flatten(), minlength=k))
    return [tuple(int(c) for c in centers[i]) for i in order]


def to_png(img: Image.Image) -> io.BytesIO:
    b = io.BytesIO()
    img.save(b, format="PNG")
    b.seek(0)
    return b


def to_jpg(img: Image.Image, quality: int = 92) -> io.BytesIO:
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=quality)
    b.seek(0)
    return b


def upscale(img: Image.Image, factor: int) -> Image.Image:
    if factor <= 1:
        return img
    return img.resize((img.width * factor, img.height * factor), Image.LANCZOS)
