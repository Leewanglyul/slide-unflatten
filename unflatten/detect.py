"""Local object finder (no API key): open-vocabulary YOLOE, prompt-free, via ultralytics.

Used for mode "full" when Claude is not available, so pictures are still cut into separate people / things
by SAM 2 instead of one blob per connected region. The detector's class names are not reliable on
illustrations ("joker" for a cartoon child), so pieces get a neutral name.

Raw detections are filtered by `select_objects` (pure function, unit-tested):
  1. confidence >= cfg.detect_conf, area within [min_object_area, max_object_area] of the slide
  2. mostly text (OCR line boxes cover > TEXT_COVER of the box) → dropped (titles, captions)
  3. "group" boxes that contain two or more more-confident boxes → dropped (keeps the individuals)
  4. largest first; a box lying >= PART_INSIDE inside an already kept box is a part of it → dropped
Layer (back → front) is by area: bigger objects sit behind smaller ones.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

from .types import Line

TEXT_COVER = 0.5
GROUP_INSIDE = 0.8
PART_INSIDE = 0.7
NAMES = {"ko": "개체", "en": "object", "ja": "オブジェクト", "zh": "对象"}

_model = None
_model_name = None


def _area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inside(a, b) -> float:
    """Fraction of box a that lies inside box b."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = _area((x0, y0, x1, y1)) if x1 > x0 and y1 > y0 else 0.0
    return inter / _area(a) if _area(a) else 0.0


def _text_cover(box, lines: Sequence[Line], size: Tuple[int, int]) -> float:
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 <= x0 or y1 <= y0:
        return 0.0
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for ln in lines:
        lx0, ly0, lx1, ly1 = ln.box
        cx0, cy0, cx1, cy1 = max(lx0, x0), max(ly0, y0), min(lx1, x1), min(ly1, y1)
        if cx1 > cx0 and cy1 > cy0:
            m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = True
    return float(m.mean())


def select_objects(dets: Sequence[Tuple[Sequence[float], float]], lines: Sequence[Line], size: Tuple[int, int],
                   conf: float, min_area: float, max_area: float, lang: str = "ko") -> List[dict]:
    """dets: [(box [x0,y0,x1,y1], confidence)] → objects [{"name", "box", "layer"}] for segment.split_semantic."""
    w, h = size
    cand = []
    for box, c in dets:
        b = [int(round(v)) for v in box]
        a = _area(b) / (w * h)
        if c < conf or not (min_area <= a <= max_area):
            continue
        if _text_cover(b, lines, size) > TEXT_COVER:
            continue
        cand.append((b, float(c)))
    groups = set()
    for i, (bi, ci) in enumerate(cand):
        members = sum(1 for j, (bj, cj) in enumerate(cand) if j != i and cj > ci and _inside(bj, bi) >= GROUP_INSIDE)
        if members >= 2:
            groups.add(i)
    kept: List[List[int]] = []
    for b, _ in sorted((x for i, x in enumerate(cand) if i not in groups), key=lambda x: -_area(x[0])):
        if any(_inside(b, k) >= PART_INSIDE for k in kept):
            continue
        kept.append(b)
    name = NAMES.get(lang, NAMES["en"])
    # kept is largest first → layer 0 (back) is the largest
    return [{"name": name, "box": b, "layer": i} for i, b in enumerate(kept)]


def detect_objects(img: Image.Image, lines: Sequence[Line], cfg, log=None) -> List[dict]:
    """Run the local detector; [] when ultralytics or the weights are unavailable (caller falls back)."""
    global _model, _model_name
    try:
        from ultralytics import YOLOE
        if _model is None or _model_name != cfg.local_detector:
            _model, _model_name = YOLOE(cfg.local_detector), cfg.local_detector
        res = _model.predict(np.array(img.convert("RGB"))[..., ::-1], conf=min(cfg.detect_conf, 0.25), verbose=False)[0]
        dets = list(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()))
    except Exception as e:
        if log:
            log(f"  local detector unavailable ({type(e).__name__}: {str(e)[:100]})")
        return []
    return select_objects(dets, lines, img.size, cfg.detect_conf, cfg.min_object_area, cfg.max_object_area,
                          cfg.language)
