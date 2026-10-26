"""Per-slide cache so fonts/assembly can change without re-running OCR, Claude, LaMa or SAM."""
from __future__ import annotations

import hashlib
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
    def key(image: Image.Image, cfg, ocr_backend: str = "none", claude: bool = False, model: str = "") -> str:
        """Hash of the pixels + the recognition config + the backends that will actually run."""
        img = image.convert("RGB")
        h = hashlib.sha1()
        h.update(f"{img.width}x{img.height}".encode())
        h.update(np.asarray(img).tobytes())
        fields = dict(cfg.cache_key_fields(), ocr=ocr_backend, claude=claude, model=model if claude else "")
        h.update(json.dumps(fields, sort_keys=True).encode())
        return h.hexdigest()[:20]

    def _dir(self, key: str) -> str:
        return os.path.join(self.root, key)

    def has(self, key: str) -> bool:
        return os.path.exists(os.path.join(self._dir(key), "analysis.json"))

    def save(self, key: str, analysis: SlideAnalysis, background: Image.Image, pieces: List[Piece],
             split_params: dict = None):
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
            json.dump({"analysis": analysis.to_dict(), "pieces": plist, "split_params": split_params or {}},
                      f, ensure_ascii=False, indent=1)

    def split_params(self, key: str) -> dict:
        """Settings the stored background/pieces were cut with ({} for entries written before this existed)."""
        with open(os.path.join(self._dir(key), "analysis.json"), encoding="utf-8") as f:
            return json.load(f).get("split_params") or {}

    def load_analysis(self, key: str) -> SlideAnalysis:
        with open(os.path.join(self._dir(key), "analysis.json"), encoding="utf-8") as f:
            return SlideAnalysis.from_dict(json.load(f)["analysis"])

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
