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


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR 3.x (falls back to the 2.x API). Often reads Korean more accurately than EasyOCR."""
    name = "paddleocr"

    def __init__(self, lang: str):
        try:                       # Windows: torch must load its DLLs before paddle, or torch fails to import later
            import torch  # noqa: F401
        except Exception:
            pass
        import os
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        try:
            # enable_mkldnn=False: Paddle 3.x's oneDNN CPU path fails on these models
            # ("ConvertPirAttribute2RuntimeAttribute not support"); plain CPU / GPU kernels work.
            self.engine = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                                    use_textline_orientation=False, enable_mkldnn=False)
            self.v3 = True
        except TypeError:          # PaddleOCR 2.x
            self.engine = PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)
            self.v3 = False

    def _raw(self, bgr: np.ndarray):
        """-> list of (4-point polygon, text, score)"""
        if self.v3:
            out = []
            for res in self.engine.predict(bgr):
                r = res.json.get("res", res.json) if hasattr(res, "json") else res
                for poly, text, score in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"]):
                    out.append((np.asarray(poly), text, float(score)))
            return out
        res = self.engine.ocr(bgr, cls=False) or [[]]
        return [(np.asarray(b), t, float(s)) for b, (t, s) in (res[0] or [])]

    def lines(self, img: Image.Image) -> List[Line]:
        arr = np.array(img.convert("RGB"))
        out: List[Line] = []
        for poly, text, conf in self._raw(arr[..., ::-1].copy()):
            xs, ys = poly[:, 0], poly[:, 1]
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            if (box[3] - box[1]) < 9 or not (text or "").strip():
                continue
            out.append(Line(box=ink_box(arr, box), text=text.strip(), conf=conf))
        return drop_noise(merge_words(out))


def drop_noise(lines: List[Line]) -> List[Line]:
    """Remove two kinds of PaddleOCR noise: 'text' with no letters or digits (a sparkle read as '√', a dash),
    and a weaker line lying mostly on top of a stronger one (the outline/shadow of a decorated title read as
    a second line)."""
    keep = [l for l in lines if any(ch.isalnum() for ch in l.text)]
    out = []
    for l in keep:
        x0, y0, x1, y1 = l.box
        area = max(1, (x1 - x0) * (y1 - y0))
        covered = False
        for o in keep:
            if o is l or o.conf <= l.conf:
                continue
            ix = max(0, min(x1, o.box[2]) - max(x0, o.box[0]))
            iy = max(0, min(y1, o.box[3]) - max(y0, o.box[1]))
            if ix * iy > 0.4 * area:
                covered = True
                break
        if not covered:
            out.append(l)
    return out


def merge_words(words: List[Line]) -> List[Line]:
    """PaddleOCR detects word by word; join words that sit on the same line with a normal word gap into one
    line (like EasyOCR's output), so rule-based grouping does not turn every word into its own text box."""
    words = sorted(words, key=lambda l: (l.box[1], l.box[0]))
    lines: List[List[Line]] = []
    for w in sorted(words, key=lambda l: l.box[0]):
        h = w.box[3] - w.box[1]
        best = None
        for ln in lines:
            last = ln[-1]
            lh = last.box[3] - last.box[1]
            overlap = min(last.box[3], w.box[3]) - max(last.box[1], w.box[1])
            gap = w.box[0] - last.box[2]
            if overlap > 0.6 * min(h, lh) and 0.6 < h / max(lh, 1) < 1.6 and -0.6 * max(h, lh) < gap < 1.2 * max(h, lh):
                best = ln
                break
        if best is None:
            lines.append([w])
        else:
            best.append(w)
    out = []
    for ln in lines:
        box = (min(l.box[0] for l in ln), min(l.box[1] for l in ln), max(l.box[2] for l in ln), max(l.box[3] for l in ln))
        conf = sum(l.conf * len(l.text) for l in ln) / max(1, sum(len(l.text) for l in ln))
        out.append(Line(box=box, text=" ".join(l.text for l in ln), conf=float(conf)))
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


def _installed(module: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def easyocr_available() -> bool:
    return _installed("easyocr")


def paddleocr_available() -> bool:
    return _installed("paddleocr") and _installed("paddle")


def resolve_ocr(cfg, service=None) -> str:
    """Which engine will actually run for cfg.ocr_backend: easyocr | paddleocr | claude | none.
    A chosen engine that is not installed falls back to the next available one (never silently to nothing)."""
    want = cfg.ocr_backend
    if want == "none":
        return "none"
    order = {"easyocr": ["easyocr", "paddleocr"], "paddleocr": ["paddleocr", "easyocr"],
             "claude": [], "auto": ["easyocr", "paddleocr"]}.get(want, ["easyocr", "paddleocr"])
    for name in order:
        if (name == "easyocr" and easyocr_available()) or (name == "paddleocr" and paddleocr_available()):
            return name
    return "claude" if service is not None else "none"


def make_backend(cfg, service=None, log=print) -> OCRBackend:
    from .i18n import t
    name = resolve_ocr(cfg, service)
    if cfg.ocr_backend in ("easyocr", "paddleocr") and name != cfg.ocr_backend:
        log(t(cfg.ui_lang, "ocr_missing", want=cfg.ocr_backend, got=name))
    if name == "easyocr":
        return EasyOCRBackend(cfg.easyocr_langs)
    if name == "paddleocr":
        return PaddleOCRBackend(cfg.paddle_lang)
    if name == "claude":
        if cfg.ocr_backend != "claude":
            log(t(cfg.ui_lang, "no_ocr"))
        return ClaudeOCRBackend(service)
    if cfg.ocr_backend != "none":
        log(t(cfg.ui_lang, "no_ocr_none"))
    return NullOCRBackend()
