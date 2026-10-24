"""Font naming, measurement files and bundling."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple


def font_family(path: str) -> str:
    """Font family name as PowerPoint/Windows sees it (name ID 1, English preferred)."""
    from fontTools.ttLib import TTFont
    try:
        names = TTFont(path, lazy=True, fontNumber=0)["name"]
    except Exception:
        return os.path.splitext(os.path.basename(path))[0]
    for pid, lang in ((3, 0x409), (3, None), (1, None)):
        for r in names.names:
            if r.nameID == 1 and r.platformID == pid and (lang is None or r.langID == lang):
                try:
                    return r.toUnicode()
                except Exception:
                    continue
    return os.path.splitext(os.path.basename(path))[0]


def font_weight(path: str) -> int:
    from fontTools.ttLib import TTFont
    try:
        return int(TTFont(path, lazy=True, fontNumber=0)["OS/2"].usWeightClass)
    except Exception:
        return 400


def assign_font_files(paths: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Thinnest file -> regular, heaviest -> bold (None when only one file)."""
    fonts = sorted((font_weight(p), p) for p in paths if p.lower().endswith((".ttf", ".otf", ".ttc")))
    if not fonts:
        return None, None
    reg = fonts[0][1]
    bold = fonts[-1][1] if len(fonts) > 1 else None
    return reg, bold


@dataclass
class FontPlan:
    regular_name: str
    bold_name: str
    fake_bold: bool          # bold text uses the regular face + bold attribute
    regular_file: Optional[str]
    bold_file: Optional[str]

    def for_block(self, bold: bool, title: bool, title_use_bold_font: bool) -> Tuple[str, bool]:
        heavy = bold or (title_use_bold_font and title)
        if not heavy:
            return self.regular_name, False
        if self.fake_bold:
            return self.regular_name, True
        return self.bold_name, self.bold_name == self.regular_name


def plan_fonts(cfg) -> FontPlan:
    if cfg.font_regular_file:
        reg = font_family(cfg.font_regular_file)
        if cfg.font_bold_file:
            bold = font_family(cfg.font_bold_file)
            return FontPlan(reg, bold, fake_bold=(bold == reg), regular_file=cfg.font_regular_file, bold_file=cfg.font_bold_file)
        return FontPlan(reg, reg, fake_bold=True, regular_file=cfg.font_regular_file, bold_file=None)
    reg, bold = cfg.regular_name, cfg.bold_name
    return FontPlan(reg, bold, fake_bold=(reg == bold), regular_file=None, bold_file=None)


def bundle_fonts(plan: FontPlan, out_dir: str) -> List[str]:
    copied = []
    for p in (plan.regular_file, plan.bold_file):
        if p and os.path.exists(p):
            dst_dir = os.path.join(out_dir, "fonts")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(p))
            if os.path.abspath(dst) != os.path.abspath(p):
                shutil.copy2(p, dst)
            copied.append(dst)
    return copied
