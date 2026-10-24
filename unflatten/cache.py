"""Per-slide cache so fonts/assembly can change without re-running OCR, Claude, LaMa or SAM."""
from __future__ import annotations

import hashlib
import io
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from .types import Piece, SlideAnalysis


class SlideCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    @staticmethod
    def key(image: Image.Image, cfg) -> str:
        h = hashlib.sha1()
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        h.update(buf.getvalue())
        h.update(json.dumps(cfg.cache_key_fields(), sort_keys=True).encode())
        return h.hexdigest()[:20]

    def _dir(self, key: str) -> str:
        return os.path.join(self.root, key)

    def has(self, key: str) -> bool:
        return os.path.exists(os.path.join(self._dir(key), "analysis.json"))

    def save(self, key: str, analysis: SlideAnalysis, background: Image.Image, pieces: List[Piece]):
        d = self._dir(key)
        os.makedirs(os.path.join(d, "pieces"), exist_ok=True)
        background.save(os.path.join(d, "background.png"))
        plist = []
        for i, p in enumerate(pieces):
            path = os.path.join(d, "pieces", f"{i:03d}.png")
            if p.rgba is not None:
                Image.fromarray(p.rgba).save(path)
            elif p.png_path and os.path.abspath(p.png_path) != os.path.abspath(path):
                Image.open(p.png_path).save(path)
            q = Piece(p.x, p.y, p.w, p.h, p.name, png_path=path)
            plist.append(q.to_dict())
        with open(os.path.join(d, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump({"analysis": analysis.to_dict(), "pieces": plist}, f, ensure_ascii=False, indent=1)

    def load(self, key: str) -> Tuple[SlideAnalysis, Image.Image, List[Piece]]:
        d = self._dir(key)
        with open(os.path.join(d, "analysis.json"), encoding="utf-8") as f:
            data = json.load(f)
        analysis = SlideAnalysis.from_dict(data["analysis"])
        background = Image.open(os.path.join(d, "background.png")).convert("RGB")
        pieces = []
        for pd in data["pieces"]:
            p = Piece.from_dict(pd)
            p.rgba = np.array(Image.open(p.png_path).convert("RGBA"))
            pieces.append(p)
        return analysis, background, pieces

    def update_blocks(self, key: str, blocks) -> None:
        """Let a user edit recognised text (e.g. from the UI table) and keep it for the next rebuild."""
        d = self._dir(key)
        with open(os.path.join(d, "analysis.json"), encoding="utf-8") as f:
            data = json.load(f)
        data["analysis"]["blocks"] = [b.to_dict() for b in blocks]
        with open(os.path.join(d, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
