"""Shared data types (JSON round-trippable for the cache)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 in source pixels


@dataclass
class Line:
    box: Box
    text: str
    conf: float = 1.0
    font_px: Optional[float] = None      # known size (PDF text layer)
    bold: Optional[bool] = None
    color: Optional[Tuple[int, int, int]] = None
    font_name: Optional[str] = None
    block_id: Optional[int] = None       # paragraph grouping from the PDF text layer

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["box"] = tuple(d["box"])
        if d.get("color") is not None:
            d["color"] = tuple(d["color"])
        return cls(**d)


@dataclass
class Block:
    lines: List[int]
    text: str
    role: str = "body"                   # title | body | label
    bold: bool = False
    align: str = "left"
    bullet: bool = False
    box: Box = (0, 0, 0, 0)
    color: Tuple[int, int, int] = (0, 0, 0)
    font_px: float = 20.0
    pitch_px: float = 0.0
    highlights: List[str] = field(default_factory=list)
    highlight_color: Optional[str] = None   # hex, e.g. "FFF59D"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["box"] = tuple(d["box"])
        d["color"] = tuple(d["color"])
        return cls(**d)


@dataclass
class Piece:
    x: int
    y: int
    w: int
    h: int
    name: str
    png_path: Optional[str] = None       # stored file (cache)
    rgba: Optional[object] = None        # numpy array when in memory

    def to_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "name": self.name, "png_path": self.png_path}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class SlideAnalysis:
    """Everything recognition produced for one slide (cached)."""
    lines: List[Line]
    blocks: List[Block]
    objects: List[dict]                  # Claude: {name, box, layer}
    background_type: str = "flat"        # flat | gradient | photo | unknown
    watermarks: List[Box] = field(default_factory=list)
    used_claude: bool = False
    source: str = "ocr"                  # ocr | text_layer | claude_ocr | none

    def to_dict(self):
        return {"lines": [l.to_dict() for l in self.lines], "blocks": [b.to_dict() for b in self.blocks],
                "objects": self.objects, "background_type": self.background_type,
                "watermarks": [list(w) for w in self.watermarks], "used_claude": self.used_claude, "source": self.source}

    @classmethod
    def from_dict(cls, d):
        return cls(lines=[Line.from_dict(x) for x in d["lines"]], blocks=[Block.from_dict(x) for x in d["blocks"]],
                   objects=d.get("objects", []), background_type=d.get("background_type", "flat"),
                   watermarks=[tuple(w) for w in d.get("watermarks", [])],
                   used_claude=d.get("used_claude", False), source=d.get("source", "ocr"))


@dataclass
class SlideSummary:
    index: int
    status: str                          # ok | failed | passthrough | text_layer
    n_lines: int = 0
    n_blocks: int = 0
    n_pieces: int = 0
    low_conf: int = 0
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0
    cache_key: Optional[str] = None
    texts: List[str] = field(default_factory=list)
