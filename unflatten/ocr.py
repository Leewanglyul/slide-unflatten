"""OCR backends behind one interface: lines(img) -> List[Line]."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from .imageops import ink_box
from .types import Line


class OCRBackend:
    name = "base"

    def lines(self, img: Image.Image) -> List[Line]:
        raise NotImplementedError


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    def __init__(self, languages: List[str], gpu: Optional[bool] = None):
        import easyocr
        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)

    def lines(self, img: Image.Image) -> List[Line]:
        arr = np.array(img)
        res = self.reader.readtext(arr, paragraph=False, width_ths=0.7, text_threshold=0.6, low_text=0.35)
        out: List[Line] = []
        for pts, text, conf in res:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            if (box[3] - box[1]) < 9:
                continue
            out.append(Line(box=ink_box(arr, box), text=(text or "").strip() or "?", conf=float(conf)))
        out.sort(key=lambda l: (l.box[1], l.box[0]))
        return out


class ClaudeOCRBackend(OCRBackend):
    """Fallback when no OCR engine is installed: ask Claude for line boxes (less precise)."""
    name = "claude"

    def __init__(self, service):
        self.service = service

    def lines(self, img: Image.Image) -> List[Line]:
        arr = np.array(img)
        raw = self.service.ocr_lines(img)
        out = []
        for box, text in raw:
            x0, y0, x1, y1 = [int(v) for v in box]
            x0, x1 = max(0, min(x0, x1)), min(img.width, max(x0, x1))
            y0, y1 = max(0, min(y0, y1)), min(img.height, max(y0, y1))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            out.append(Line(box=ink_box(arr, (x0, y0, x1, y1)), text=text.strip() or "?", conf=0.9))
        out.sort(key=lambda l: (l.box[1], l.box[0]))
        return out


class NullOCRBackend(OCRBackend):
    name = "none"

    def lines(self, img: Image.Image) -> List[Line]:
        return []


def easyocr_available() -> bool:
    try:
        import importlib
        return importlib.util.find_spec("easyocr") is not None
    except Exception:
        return False


def make_backend(cfg, service=None, log=print) -> OCRBackend:
    from .i18n import t
    want = cfg.ocr_backend
    if want in ("auto", "easyocr") and easyocr_available():
        return EasyOCRBackend(cfg.easyocr_langs)
    if want == "easyocr":
        log("easyocr is not installed (pip install easyocr)")
    if want in ("auto", "claude") and service is not None:
        log(t(cfg.ui_lang, "no_ocr"))
        return ClaudeOCRBackend(service)
    if want != "none":
        log(t(cfg.ui_lang, "no_ocr_none"))
    return NullOCRBackend()
