"""Split illustrations from the background into transparent pieces (simple contours or Claude + SAM 2)."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from .imageops import border_lab
from .inpaint import inpaint
from .types import Piece

_sam = None


# ───────── edge handling shared by both splitters ─────────
def _local_bg(lab: np.ndarray, box, pad: int = 12) -> np.ndarray:
    """Median colour of a ring just outside the object's box: the surface it actually sits on
    (a coloured card, a panel), which may differ from the slide's outer border."""
    h, w = lab.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0, ox1, oy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    ring = np.ones((oy1 - oy0, ox1 - ox0), bool)
    ring[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = False
    sub = lab[oy0:oy1, ox0:ox1][ring]
    return np.median(sub, axis=0) if len(sub) else border_lab(lab)


def _grow(lab: np.ndarray, m: np.ndarray, radius: int, take) -> np.ndarray:
    """Grow `m` by up to `radius` px into pixels for which take(lab, local_bg) is True."""
    if radius <= 0 or not m.any():
        return m
    ys, xs = np.where(m)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    bg = _local_bg(lab, box, pad=radius + 8)
    k = 2 * radius + 1
    ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return m | (ring & take(lab, bg))


def _rim(lab, bg):
    """Sticker outlines and rims: lighter than the surface, or clearly a different colour.
    Soft drop shadows (a little darker, same hue) are left out of the cut-out."""
    dist = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    lighter = lab[..., 0] > bg[0] + 10
    return (lighter & (dist >= 18)) | (dist >= 40)


def _not_surface(lab, bg):
    """Anything that is not the surface: rims and soft shadows alike (used when clearing the background)."""
    return np.sqrt(((lab - bg) ** 2).sum(axis=2)) >= 10


def trim_surface(lab: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Drop surface-coloured areas that SAM swallowed at the edge of a mask (the card showing between two
    figures). Only regions connected to the outside of the mask go; surface-coloured parts enclosed by the
    object (a pink shirt inside a sticker outline) stay."""
    if not m.any():
        return m
    ys, xs = np.where(m)
    bg = _local_bg(lab, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    surface = m & (np.sqrt(((lab - bg) ** 2).sum(axis=2)) < 12)
    if not surface.any():
        return m
    n, lbl = cv2.connectedComponents(surface.astype(np.uint8), connectivity=4)
    edge = surface & (cv2.dilate((~m).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    drop = np.unique(lbl[edge])
    trimmed = m & ~np.isin(lbl, drop[drop > 0])
    return trimmed if trimmed.sum() > 0.3 * m.sum() else m     # never gut an object that is mostly surface-coloured


def refine_mask(lab: np.ndarray, m: np.ndarray, expand: int, exclude=None) -> np.ndarray:
    """Trim surface the mask swallowed, then grow by up to `expand` px into the outline / sticker rim, judged
    against the surface the object sits on (a coloured card), not the slide border."""
    grown = _grow(lab, trim_surface(lab, m), expand, _rim)
    if exclude is not None:
        grown &= ~exclude | m
    return grown


def background_mask(lab: np.ndarray, union: np.ndarray, expand: int) -> np.ndarray:
    """What to erase from the background behind the cut-outs: the objects plus their rims and soft
    drop shadows, so no grey halo is left where a picture used to be."""
    n, lbl = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    grown = np.zeros(union.shape, bool)
    for c in range(1, n):            # each object against its own surface (a slide may hold several cards)
        grown |= _grow(lab, lbl == c, max(8, 3 * expand), _not_surface)
    return cv2.dilate(grown.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))


def soft_alpha(m: np.ndarray, feather: float) -> np.ndarray:
    """Binary mask -> uint8 alpha with an anti-aliased rim `feather` px wide; the interior stays opaque."""
    if feather <= 0:
        return m.astype(np.uint8) * 255
    blur = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=feather)
    core = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return (np.where(core, 1.0, np.clip(blur * 1.5, 0, 1)) * 255).astype(np.uint8)


def _piece(rgb: np.ndarray, m: np.ndarray, feather: float, name: str) -> Piece:
    """Crop one piece with a soft edge; the crop leaves room for the feathered rim."""
    h, w = m.shape
    alpha = soft_alpha(m, feather)
    ys, xs = np.where(alpha > 0)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([rgb[by0:by1, bx0:bx1], alpha[by0:by1, bx0:bx1]])
    return Piece(x=int(bx0), y=int(by0), w=int(bx1 - bx0), h=int(by1 - by0), name=name, rgba=rgba)


def unique_names(names: List[str]) -> List[str]:
    seen = {}
    out = []
    for n in names:
        n = (n or "").strip() or "object"
        seen[n] = seen.get(n, 0) + 1
        out.append(n)
    counts = {}
    res = []
    for n in out:
        if seen[n] > 1:
            counts[n] = counts.get(n, 0) + 1
            res.append(f"{n} {counts[n]}")
        else:
            res.append(n)
    return res


# ───────── simple: connected blobs that differ from the border colour ─────────
def find_objects(clean: np.ndarray, min_area: float, max_area: float) -> List[tuple]:
    h, w = clean.shape[:2]
    lab = cv2.cvtColor(clean, cv2.COLOR_RGB2LAB).astype(np.int16)
    bg = border_lab(lab)
    dist = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    fg = (dist > 18).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area * w * h < area < max_area * w * h):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [c], -1, 255, cv2.FILLED)
        objs.append((x, y, bw, bh, m))
    objs.sort(key=lambda o: (o[1], o[0]))
    return objs


def split_simple(clean: Image.Image, cfg, log=None) -> Tuple[Image.Image, List[Piece]]:
    arr = np.array(clean.convert("RGB"))
    objs = find_objects(arr, cfg.min_object_area, cfg.max_object_area)
    if not objs:
        return clean, []
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.int16)
    omask = np.zeros(arr.shape[:2], bool)
    pieces = []
    for n, (x, y, w, h, m) in enumerate(objs, 1):
        m = refine_mask(lab, m > 0, cfg.edge_expand)
        omask |= m
        pieces.append(_piece(arr, m, cfg.edge_feather, f"object {n}"))
    background = inpaint(clean, background_mask(lab, omask, cfg.edge_expand), log)
    return background, pieces


# ───────── semantic: Claude boxes + SAM 2 masks ─────────
def sam_masks(img: Image.Image, boxes: List[List[int]], model_name: str) -> List[np.ndarray]:
    global _sam
    import torch
    from ultralytics import SAM
    if _sam is None:
        name = model_name if torch.cuda.is_available() else model_name.replace("_l.", "_b.")
        _sam = SAM(name)
    res = _sam(np.array(img)[..., ::-1], bboxes=boxes, verbose=False)[0]
    masks = res.masks.data.cpu().numpy() > 0.5
    h, w = img.height, img.width
    return [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.shape != (h, w) else m for m in masks]


def split_semantic(clean: Image.Image, objs: List[dict], cfg, log=None) -> Tuple[Image.Image, List[Piece]]:
    """Cut Claude's objects with SAM 2, back-to-front. Parts of a back object hidden by a front object are
    filled in so moving the front object leaves no hole."""
    arr = np.array(clean.convert("RGB"))
    h, w = arr.shape[:2]
    boxes, keep = [], []
    for o in objs:
        x0, y0, x1, y1 = [int(v) for v in o["box"]]
        x0, x1 = sorted((max(0, x0), min(w, x1)))
        y0, y1 = sorted((max(0, y0), min(h, y1)))
        area = (x1 - x0) * (y1 - y0)
        if area < cfg.min_object_area * w * h * 0.25 or area > 0.9 * w * h:
            continue
        boxes.append([x0, y0, x1, y1])
        keep.append(o)
    if not boxes:
        return clean, []
    masks = sam_masks(clean, boxes, cfg.sam_model)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.int16)
    bg = border_lab(lab)
    fg = (np.sqrt(((lab - bg) ** 2).sum(axis=2)) >= 18).astype(np.uint8)
    order = sorted(range(len(keep)), key=lambda i: (keep[i].get("layer", 0), -int(masks[i].sum())))
    sam_only = [m.copy() for m in masks]
    claimed = np.zeros(fg.shape, bool)
    for k in order:   # add thin foreground strokes SAM missed (strings, stems) that touch the mask
        x0, y0, x1, y1 = boxes[k]
        others = claimed.copy()
        for j, mj in enumerate(sam_only):
            if j != k:
                others |= mj
        sub_fg = fg[y0:y1, x0:x1] & ~others[y0:y1, x0:x1]
        n, lbl = cv2.connectedComponents(sub_fg, connectivity=8)
        near = cv2.dilate(masks[k][y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        touch = np.unique(lbl[near & (sub_fg > 0)])
        add = np.isin(lbl, touch[touch > 0])
        masks[k][y0:y1, x0:x1] |= add
        claimed[y0:y1, x0:x1] |= add
    for k in order:   # sticker outlines / soft rims against the surface each object sits on
        others = np.zeros(fg.shape, bool)
        for j, mj in enumerate(sam_only):
            if j != k:
                others |= mj
        masks[k] = refine_mask(lab, masks[k], cfg.edge_expand, exclude=others)
    names = unique_names([o.get("name", "") for o in keep])
    pieces: List[Piece] = []
    union = np.zeros((h, w), np.uint8)
    for pos, i in enumerate(order):
        m = masks[i]
        if m.sum() < 50:
            continue
        front = np.zeros((h, w), bool)
        for j in order[pos + 1:]:
            front |= masks[j]
        front = cv2.dilate(front.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        visible = m & ~front
        x0, y0, x1, y1 = boxes[i]
        near = np.zeros((h, w), bool)
        near[y0:y1, x0:x1] = cv2.dilate(visible.astype(np.uint8), np.ones((13, 13), np.uint8))[y0:y1, x0:x1] > 0
        hidden = front & near & m
        rgb = arr
        if hidden.any():
            pad = 24
            cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
            hm = cv2.dilate(hidden.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
            sub = inpaint(Image.fromarray(arr[cy0:cy1, cx0:cx1]), hm[cy0:cy1, cx0:cx1], log)
            rgb = arr.copy()
            rgb[cy0:cy1, cx0:cx1] = np.array(sub)
            filled_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
            hidden &= np.sqrt(((filled_lab - bg) ** 2).sum(axis=2)) >= 18
            hidden = cv2.morphologyEx(hidden.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        alpha = visible | hidden
        if not alpha.any():
            continue
        pieces.append(_piece(rgb, alpha, cfg.edge_feather, names[i]))
        union |= m.astype(np.uint8)
    background = inpaint(clean, background_mask(lab, union > 0, cfg.edge_expand), log)
    return background, pieces


def pieces_from_embedded(embedded: List[dict]) -> List[Piece]:
    """PDF: embedded raster images become pieces directly (exact position, no segmentation)."""
    pieces = []
    for n, e in enumerate(embedded, 1):
        x0, y0, x1, y1 = e["box"]
        im = e["image"].convert("RGBA").resize((max(1, x1 - x0), max(1, y1 - y0)), Image.LANCZOS)
        pieces.append(Piece(x=x0, y=y0, w=im.width, h=im.height, name=e.get("name") or f"image {n}", rgba=np.array(im)))
    return pieces
