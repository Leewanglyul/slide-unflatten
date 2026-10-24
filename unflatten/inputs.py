"""Load PPTX / PDF / images into a uniform list of slides.

Detects which slides are really "flattened" (one picture covering the slide and no real text)
so already-editable slides pass through untouched, keeps notes and the original geometry,
and reads the PDF text layer when there is one (no OCR, no API cost).
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

from .types import Line

PICTURE = 13  # MSO_SHAPE_TYPE.PICTURE
EMU_PER_INCH = 914400
DEFAULT_WIDTH_EMU = 12192000  # 16:9, 13.333 in


@dataclass
class SlideInput:
    index: int                                   # 1-based position in the deck
    kind: str                                    # flat | passthrough | text_layer
    image: Optional[Image.Image] = None          # rendered / embedded image (RGB)
    notes: str = ""
    rect_emu: Optional[Tuple[int, int, int, int]] = None   # where the picture sits (left, top, width, height)
    lines: Optional[List[Line]] = None           # PDF text layer lines (pixels of `image`)
    clean_image: Optional[Image.Image] = None    # PDF rendered without text (pixel-perfect background)
    embedded_images: List[dict] = field(default_factory=list)  # PDF: {"box": (x0,y0,x1,y1), "image": PIL}
    pic_shape_id: Optional[int] = None           # PPTX: shape id of the flattened picture (to remove)
    reason: str = ""


@dataclass
class Deck:
    path: str
    kind: str                                    # pptx | pdf | image
    slides: List[SlideInput]
    width_emu: int
    height_emu: int
    source_prs: object = None                    # python-pptx Presentation when kind == pptx

    @property
    def flat_count(self) -> int:
        return sum(1 for s in self.slides if s.kind != "passthrough")


# ───────────────────────── PPTX ─────────────────────────
def _slide_has_real_text(slide) -> bool:
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            return True
        if getattr(sh, "has_table", False) and sh.has_table:
            return True
    return False


def _crop_pic(pic) -> Image.Image:
    img = Image.open(io.BytesIO(pic.image.blob)).convert("RGB")
    cl, cr, ct, cb = pic.crop_left or 0, pic.crop_right or 0, pic.crop_top or 0, pic.crop_bottom or 0
    if any(abs(c) > 1e-4 for c in (cl, cr, ct, cb)):
        w, h = img.size
        img = img.crop((int(w * cl), int(h * ct), int(w * (1 - cr)), int(h * (1 - cb))))
    return img


def load_pptx(path: str, flat_cover: float = 0.85) -> Deck:
    from pptx import Presentation
    prs = Presentation(path)
    sw, sh = prs.slide_width, prs.slide_height
    slides: List[SlideInput] = []
    for idx, slide in enumerate(prs.slides, 1):
        notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
        pics = [s for s in slide.shapes if s.shape_type == PICTURE]
        if not pics:
            slides.append(SlideInput(idx, "passthrough", notes=notes, reason="no picture"))
            continue
        big = max(pics, key=lambda s: (s.width or 0) * (s.height or 0))
        cover = ((big.width or 0) * (big.height or 0)) / float(sw * sh)
        if cover < flat_cover or _slide_has_real_text(slide):
            slides.append(SlideInput(idx, "passthrough", notes=notes,
                                     reason=f"picture covers {cover:.0%}" if cover < flat_cover else "has real text"))
            continue
        try:
            img = _crop_pic(big)
        except Exception as e:  # unreadable blob (emf/wmf etc.)
            slides.append(SlideInput(idx, "passthrough", notes=notes, reason=f"unreadable picture: {e}"))
            continue
        slides.append(SlideInput(idx, "flat", image=img, notes=notes,
                                 rect_emu=(big.left or 0, big.top or 0, big.width, big.height),
                                 pic_shape_id=big.shape_id))
    return Deck(path, "pptx", slides, sw, sh, source_prs=prs)


# ───────────────────────── PDF ─────────────────────────
def _pdf_lines(page, scale: float) -> List[Line]:
    """Lines from the text layer, in rendered-pixel coordinates."""
    lines: List[Line] = []
    d = page.get_text("dict")
    bid = 0
    for blk in d.get("blocks", []):
        if blk.get("type", 0) != 0:
            continue
        bid += 1
        for ln in blk.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in ln["spans"]).strip()
            x0 = min(s["bbox"][0] for s in spans); y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans); y1 = max(s["bbox"][3] for s in spans)
            main = max(spans, key=lambda s: len(s["text"]))
            col = main.get("color", 0)
            color = ((col >> 16) & 255, (col >> 8) & 255, col & 255)
            bold = bool(main.get("flags", 0) & 16) or "bold" in main.get("font", "").lower()
            lines.append(Line(box=(int(x0 * scale), int(y0 * scale), int(x1 * scale) + 1, int(y1 * scale) + 1),
                              text=text, conf=1.0, font_px=float(main["size"]) * scale, bold=bold,
                              color=color, font_name=main.get("font"), block_id=bid))
    return lines


def _render(page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def load_pdf(path: str, dpi: int = 150, extract_images: bool = True, min_text_chars: int = 8) -> Deck:
    import pymupdf
    doc = pymupdf.open(path)
    slides: List[SlideInput] = []
    width_emu = height_emu = None
    for idx, page in enumerate(doc, 1):
        scale = dpi / 72.0
        img = _render(page, dpi)
        if width_emu is None:
            width_emu = DEFAULT_WIDTH_EMU
            height_emu = int(DEFAULT_WIDTH_EMU * page.rect.height / page.rect.width)
        lines = _pdf_lines(page, scale)
        n_chars = sum(len(l.text) for l in lines)
        if n_chars < min_text_chars:
            slides.append(SlideInput(idx, "flat", image=img))
            continue
        # text layer present: render the page again with text (and optionally images) removed
        clean, embedded = None, []
        try:
            tmp = pymupdf.open(path)
            p2 = tmp[idx - 1]
            for blk in p2.get_text("dict").get("blocks", []):
                if blk.get("type", 0) == 0:
                    p2.add_redact_annot(pymupdf.Rect(blk["bbox"]))
            img_mode = pymupdf.PDF_REDACT_IMAGE_NONE
            if extract_images:
                for info in p2.get_images(full=True):
                    xref = info[0]
                    try:
                        rects = p2.get_image_rects(xref)
                    except Exception:
                        rects = []
                    for r in rects:
                        area = (r.width * r.height) / (p2.rect.width * p2.rect.height)
                        if area < 0.004 or area > 0.9:
                            continue
                        try:
                            pm = pymupdf.Pixmap(tmp, xref)
                            if pm.n - pm.alpha >= 4:
                                pm = pymupdf.Pixmap(pymupdf.csRGB, pm)
                            mode = "RGBA" if pm.alpha else "RGB"
                            pil = Image.frombytes(mode, (pm.width, pm.height), pm.samples)
                        except Exception:
                            continue
                        box = (int(r.x0 * scale), int(r.y0 * scale), int(r.x1 * scale), int(r.y1 * scale))
                        embedded.append({"box": box, "image": pil, "name": f"image {len(embedded) + 1}"})
                        p2.add_redact_annot(r)
                if embedded:
                    img_mode = pymupdf.PDF_REDACT_IMAGE_REMOVE
            p2.apply_redactions(images=img_mode, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
            clean = _render(p2, dpi)
            tmp.close()
        except Exception:
            clean, embedded = None, []
        slides.append(SlideInput(idx, "text_layer", image=img, lines=lines, clean_image=clean,
                                 embedded_images=embedded))
    doc.close()
    return Deck(path, "pdf", slides, width_emu or DEFAULT_WIDTH_EMU, height_emu or int(DEFAULT_WIDTH_EMU * 9 / 16))


# ───────────────────────── images ─────────────────────────
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def load_images(paths: List[str]) -> Deck:
    slides = []
    w_emu = h_emu = None
    for idx, p in enumerate(paths, 1):
        img = Image.open(p).convert("RGB")
        if w_emu is None:
            w_emu, h_emu = DEFAULT_WIDTH_EMU, int(DEFAULT_WIDTH_EMU * img.height / img.width)
        slides.append(SlideInput(idx, "flat", image=img))
    return Deck(paths[0], "image", slides, w_emu, h_emu)


def load_deck(path: str, dpi: int = 150) -> Deck:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pptx":
        return load_pptx(path)
    if ext == ".pdf":
        return load_pdf(path, dpi=dpi)
    if ext in IMAGE_EXT:
        return load_images([path])
    raise ValueError(f"Unsupported input: {path}")
