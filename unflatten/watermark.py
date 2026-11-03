"""Rule-based watermark finder: works without Claude.

Slide generators stamp a small mark near the slide edge ("A Gemini Notebook", "NotebookLM", "Made with Gamma" …).
OCR reads it like any other text, so without Claude it would come back as an editable text box. Here an OCR line
counts as a watermark when ALL of these hold:
  - its text matches one of PATTERNS (case/space-insensitive)
  - it sits in the outer band of the slide (EDGE_BAND of the width or height from any edge)
  - it is small (height <= MAX_HEIGHT of the slide height)
The returned erase box is widened to the left/right by ICON_WIDTH × line height to take in a logo icon next to
the words (e.g. the Gemini sparkle).
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .types import Line

RULE_VERSION = 1   # part of the cache key: bump when the rule changes so cached slides are re-analysed

PATTERNS = [
    r"gemini\s*notebook", r"notebook\s*lm", r"google\s*notebook",
    r"made\s*with\s*(gamma|canva|gemini|notebooklm|beautiful\.?ai|tome|pitch|genspark|felo|napkin)",
    r"created\s*with\s*(gamma|canva|gemini|notebooklm)", r"generated\s*(with|by)\s*(gamma|gemini|notebooklm|ai)",
    r"gamma\.app", r"canva\.com", r"beautiful\.ai", r"genspark", r"felo\.ai",
]
_RX = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.IGNORECASE)

EDGE_BAND = 0.12
MAX_HEIGHT = 0.05
ICON_WIDTH = 1.6
PAD = 0.3


def is_watermark_text(text: str) -> bool:
    return bool(_RX.search(" ".join(text.split())))


def find_watermarks(lines: Sequence[Line], size: Tuple[int, int]) -> Tuple[List[int], List[List[int]]]:
    """(indexes of watermark lines, erase boxes [x0, y0, x1, y1]) for an image of `size` (w, h)."""
    w, h = size
    idx, boxes = [], []
    for i, ln in enumerate(lines):
        x0, y0, x1, y1 = ln.box
        lh = y1 - y0
        if lh > MAX_HEIGHT * h or not is_watermark_text(ln.text):
            continue
        near_edge = (x0 < EDGE_BAND * w or x1 > (1 - EDGE_BAND) * w or
                     y0 < EDGE_BAND * h or y1 > (1 - EDGE_BAND) * h)
        if not near_edge:
            continue
        idx.append(i)
        pad, icon = PAD * lh, ICON_WIDTH * lh
        boxes.append([int(max(0, x0 - icon)), int(max(0, y0 - pad)),
                      int(min(w, x1 + icon)), int(min(h, y1 + pad))])
    return idx, boxes
