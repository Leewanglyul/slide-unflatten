"""Text lines -> blocks (grouping, correction) and block styling (size, colour, pitch)."""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import ImageFont

from .config import FALLBACK_MEASURE
from .imageops import text_color
from .types import Block, Line

BULLET_RE = re.compile(r"^\s*[•●○◦▪■□◆◇►▶➢✓✔\-–—*·]\s+")


# ───────────── grouping without Claude ─────────────
def group_heuristic(lines: List[Line], min_conf: float = 0.3, img_h: int = 0) -> List[Block]:
    """One text box per line (most accurate positions) — PDF text layer lines keep their paragraph blocks."""
    if lines and all(l.block_id is not None for l in lines):
        return group_text_layer(lines, img_h)
    blocks = []
    for i, l in enumerate(lines):
        if l.conf < min_conf:
            continue
        text, bullet = _strip_bullet(l.text)
        blocks.append(Block(lines=[i], text=text, bullet=bullet, bold=bool(l.bold)))
    return blocks


def merge_adjacent_lines(lines: List[Line]) -> List[Line]:
    """PDF generators often emit every line as its own block. Re-join lines into paragraphs when they share
    size and weight, sit one line-pitch apart and are aligned (left edge or centre)."""
    order = sorted(range(len(lines)), key=lambda i: (lines[i].box[1], lines[i].box[0]))
    new_ids = {}
    next_id = 1
    prev = None
    for i in order:
        l = lines[i]
        if prev is not None:
            p = lines[prev]
            fs = (l.font_px or (l.box[3] - l.box[1]))
            pfs = (p.font_px or (p.box[3] - p.box[1]))
            gap = l.box[1] - p.box[3]
            cen_l = (l.box[0] + l.box[2]) / 2; cen_p = (p.box[0] + p.box[2]) / 2
            aligned = abs(l.box[0] - p.box[0]) < 0.6 * fs or abs(cen_l - cen_p) < 0.6 * fs
            same = abs(fs - pfs) <= 0.12 * max(fs, pfs) and bool(l.bold) == bool(p.bold)
            if same and aligned and -0.2 * fs < gap < 0.9 * fs:
                new_ids[i] = new_ids[prev]
                prev = i
                continue
        new_ids[i] = next_id
        next_id += 1
        prev = i
    out = []
    for i, l in enumerate(lines):
        c = Line(**{**l.__dict__})
        c.block_id = new_ids[i]
        out.append(c)
    return out


def group_text_layer(lines: List[Line], img_h: int = 0) -> List[Block]:
    """PDF text layer: lines carry a paragraph id (re-derived by merge_adjacent_lines); one block per paragraph."""
    lines = merge_adjacent_lines(lines)
    by_block = {}
    for i, l in enumerate(lines):
        by_block.setdefault(l.block_id, []).append(i)
    blocks: List[Block] = []
    for ids in by_block.values():
        ids.sort(key=lambda i: (lines[i].box[1], lines[i].box[0]))
        texts, bullets = [], []
        for i in ids:
            t, b = _strip_bullet(lines[i].text)
            texts.append(t); bullets.append(b)
        heights = [lines[i].box[3] - lines[i].box[1] for i in ids]
        xs = [lines[i].box[0] for i in ids]
        x1s = [lines[i].box[2] for i in ids]
        align = "left"
        if len(ids) > 1:
            spread_l = max(xs) - min(xs); spread_r = max(x1s) - min(x1s)
            centers = [(lines[i].box[0] + lines[i].box[2]) / 2 for i in ids]
            if spread_l > 0.15 * max(heights) * 2 and max(centers) - min(centers) < spread_l * 0.5:
                align = "center"
            elif spread_l > spread_r * 2:
                align = "right"
        role = "body"
        fpx = lines[ids[0]].font_px or 0
        if fpx > 0 and len(ids) <= 2:
            # a title is big relative to the slide (≈ 28pt+ on a 7.5in-high slide ≈ 5% of the height)
            thresh = 0.05 * img_h if img_h else 28 * 2.0
            role = "title" if all((lines[i].font_px or 0) >= thresh for i in ids) else "body"
        blocks.append(Block(lines=ids, text="\n".join(texts), bullet=all(bullets) and len(bullets) > 0,
                            bold=bool(lines[ids[0]].bold), align=align, role=role))
    return blocks


def _strip_bullet(text: str) -> Tuple[str, bool]:
    m = BULLET_RE.match(text)
    if m:
        return text[m.end():].strip(), True
    return text.strip(), False


# ───────────── grouping with Claude ─────────────
def blocks_from_claude(out, lines: List[Line]) -> Tuple[List[Block], List[dict], str, List[Tuple[int, int, int, int]]]:
    blocks: List[Block] = []
    for b in out.blocks:
        ids = sorted({i for i in b.line_ids if 0 <= i < len(lines)})
        if not ids or not b.text.strip():
            continue
        text = "\n".join(_strip_bullet(p)[0] for p in b.text.split("\n"))
        hc = (b.highlight_color or "").lstrip("#").upper() or None
        if hc and not re.fullmatch(r"[0-9A-F]{6}", hc):
            hc = None
        blocks.append(Block(lines=ids, text=text, role=b.role if b.role in ("title", "body", "label") else "body",
                            bold=b.bold, align=b.align if b.align in ("left", "center", "right") else "left",
                            bullet=b.bullet, highlights=[h for h in b.highlights if h.strip()], highlight_color=hc))
    objects = [{"name": o.name, "box": [int(v) for v in o.box], "layer": int(o.layer)}
               for o in out.objects if len(o.box) == 4]
    bg = out.background_type if out.background_type in ("flat", "gradient", "photo") else "unknown"
    wms = [tuple(int(v) for v in w) for w in out.watermarks if len(w) == 4]
    return blocks, objects, bg, wms


# ───────────── measurement fonts ─────────────
class FontMeasurer:
    """Measures text with a real font file so sizes fit the original box in both width and height."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._cache = {}
        self.regular_path = cfg.font_regular_file or self._first(cfg, bold=False)
        self.bold_path = cfg.font_bold_file or self._first(cfg, bold=True) or self.regular_path

    @staticmethod
    def _first(cfg, bold: bool) -> Optional[str]:
        for reg, bd in list(cfg.profile["measure"]) + FALLBACK_MEASURE:
            p = bd if bold else reg
            if p and os.path.exists(p):
                return p
        return None

    def font(self, bold: bool):
        path = self.bold_path if bold else self.regular_path
        if not path:
            return None
        if path not in self._cache:
            try:
                self._cache[path] = ImageFont.truetype(path, 100)
            except Exception:
                self._cache[path] = None
        return self._cache[path]

    def measure(self, text: str, bold: bool) -> Optional[Tuple[float, float]]:
        """(width, ink height) of `text` at size 100, or None."""
        f = self.font(bold)
        if f is None:
            return None
        widest = 0.0
        for t in text.split("\n"):
            if t.strip():
                widest = max(widest, f.getlength(t))
        if widest <= 0:
            return None
        sample = max(text.split("\n"), key=len)
        try:
            l, tp, r, bt = f.getbbox(sample)
            ink_h = float(bt - tp)
        except Exception:
            ink_h = 100.0
        return widest, max(ink_h, 1.0)


# ───────────── font fitting ─────────────
def fit_font(b: Block, measurer: Optional[FontMeasurer], ink_ratio: float = 0.92) -> None:
    """Set b.font_px so the chosen font fills the original box in both width and height."""
    ink_h = b.ink_h or (b.box[3] - b.box[1])
    b.font_px = ink_h / ink_ratio
    m = measurer.measure(b.text, b.bold or b.role == "title") if measurer else None
    if m:
        w100, h100 = m
        target_w = b.box[2] - b.box[0]
        by_w = target_w * 100.0 / w100 if w100 > 0 else b.font_px
        by_h = ink_h * 100.0 / h100
        est = min(by_w * 1.04, by_h)             # trust the height, never overflow the width
        if est > 0:
            b.font_px = float(est)


def refit_fonts(blocks: List[Block], measurer: FontMeasurer, ink_ratio: float = 0.92) -> List[Block]:
    """Called at assembly time: sizes measured with another font (or cached) are re-fitted to the current one."""
    for b in blocks:
        if not b.size_known and b.ink_h > 0:
            fit_font(b, measurer, ink_ratio)
    return blocks


# ───────────── finalize ─────────────
def finalize_blocks(blocks: List[Block], lines: List[Line], arr: np.ndarray, measurer: FontMeasurer,
                    ink_ratio: float = 0.92) -> List[Block]:
    H = arr.shape[0]
    for b in blocks:
        ls = [lines[i] for i in b.lines]
        b.box = (min(l.box[0] for l in ls), min(l.box[1] for l in ls),
                 max(l.box[2] for l in ls), max(l.box[3] for l in ls))
        n = b.text.count("\n") + 1
        heights = sorted(l.box[3] - l.box[1] for l in ls)
        ink_h = heights[len(heights) // 2]
        b.ink_h = float(ink_h)
        known = [l.font_px for l in ls if l.font_px]
        if known:                                   # PDF text layer: exact size
            b.font_px = float(sorted(known)[len(known) // 2])
            b.size_known = True
        else:
            fit_font(b, measurer, ink_ratio)
        if n > 1:
            b.pitch_px = (b.box[3] - b.box[1] - ink_h) / (n - 1)
        known_col = [l.color for l in ls if l.color is not None]
        if known_col:
            b.color = tuple(int(np.median([c[k] for c in known_col])) for k in range(3))
        else:
            cols = [text_color(arr, l.box) for l in ls]
            b.color = tuple(int(np.median([c[k] for c in cols])) for k in range(3))
        if b.role == "title" or b.font_px > 0.07 * H:
            b.bold = True
    return blocks
