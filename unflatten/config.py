"""Configuration: presets, language profiles, model table."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import List, Optional

# ───────────── model table (first-party API prices, USD per 1M tokens) ─────────────
MODELS = {
    "claude-opus-5-5":   {"label": "Opus 5.5 (권장 · 가장 정확)", "in": 4.00, "out": 20.00},
    "claude-sonnet-5-5": {"label": "Sonnet 5.5 (절반 가격)",      "in": 2.00, "out": 10.00},
}
DEFAULT_MODEL = "claude-opus-5-5"
# rough per-slide token budget used for the estimate (image ≈ 1.6k + prompt + line listing / structured answer)
EST_INPUT_TOKENS = 3200
EST_OUTPUT_TOKENS = 1600
USD_KRW = 1400.0

# ───────────── processing presets ─────────────
# mode: text    = text boxes only, background untouched except text removal (fast, cheap)
#       objects = text + whole illustrations as pieces (coarse)
#       full    = text + every object inside illustrations as pieces (fine)
PRESETS = {
    "text":    {"split_objects": False, "object_detail": "none",   "label_ko": "텍스트만 (빠름·저렴)",      "label_en": "Text only (fast)"},
    "objects": {"split_objects": True,  "object_detail": "coarse", "label_ko": "텍스트 + 그림 통째",       "label_en": "Text + whole pictures"},
    "full":    {"split_objects": True,  "object_detail": "fine",   "label_ko": "완전 분리 (그림 속 개체까지)", "label_en": "Full split (objects inside pictures)"},
}

# ───────────── language profiles ─────────────
# easyocr: EasyOCR language codes; lang_tag: PPTX run language; fonts: default PowerPoint font names
# measure: candidate font files used to fit text width/height (first existing wins)
_WIN = "C:/Windows/Fonts/"
_LINUX_CJK = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
LANGUAGE_PROFILES = {
    "ko": {"label": "한국어 + 영어", "easyocr": ["ko", "en"], "lang_tag": "ko-KR",
           "font_regular": "맑은 고딕", "font_bold": "맑은 고딕",
           "measure": [(_WIN + "malgun.ttf", _WIN + "malgunbd.ttf"),
                       ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
                       (_LINUX_CJK[0], _LINUX_CJK[0]), (_LINUX_CJK[1], _LINUX_CJK[1])]},
    "en": {"label": "English", "easyocr": ["en"], "lang_tag": "en-US",
           "font_regular": "Calibri", "font_bold": "Calibri",
           "measure": [(_WIN + "calibri.ttf", _WIN + "calibrib.ttf"),
                       ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                       ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")]},
    "ja": {"label": "日本語 + English", "easyocr": ["ja", "en"], "lang_tag": "ja-JP",
           "font_regular": "Yu Gothic", "font_bold": "Yu Gothic",
           "measure": [(_WIN + "YuGothR.ttc", _WIN + "YuGothB.ttc"), (_LINUX_CJK[0], _LINUX_CJK[0]), (_LINUX_CJK[1], _LINUX_CJK[1])]},
    "zh": {"label": "中文 + English", "easyocr": ["ch_sim", "en"], "lang_tag": "zh-CN",
           "font_regular": "Microsoft YaHei", "font_bold": "Microsoft YaHei",
           "measure": [(_WIN + "msyh.ttc", _WIN + "msyhbd.ttc"), (_LINUX_CJK[0], _LINUX_CJK[0]), (_LINUX_CJK[1], _LINUX_CJK[1])]},
}
# always try these as a last resort for measurement (Latin only)
FALLBACK_MEASURE = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    (_WIN + "arial.ttf", _WIN + "arialbd.ttf"),
]


@dataclass
class Config:
    # what to do
    mode: str = "full"                      # text | objects | full
    language: str = "ko"                    # key of LANGUAGE_PROFILES
    slide_range: Optional[str] = None       # "1-3,7"  (1-based); None = all
    # Claude
    use_claude: bool = True
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None           # None -> ANTHROPIC_API_KEY env
    effort: str = "low"
    # OCR / vision backends
    ocr_backend: str = "auto"               # auto | easyocr | claude | none
    sam_model: str = "sam2.1_l.pt"
    pdf_dpi: int = 150
    # fonts
    font_regular_file: Optional[str] = None
    font_bold_file: Optional[str] = None
    font_regular: Optional[str] = None      # None -> language default
    font_bold: Optional[str] = None
    title_use_bold_font: bool = True
    use_theme_fonts: bool = True            # text runs reference the theme fonts (+mj/+mn)
    # output shape
    upscale: int = 1                        # 1 = keep source resolution (recommended)
    keep_hidden_original: bool = True
    title_as_placeholder: bool = True
    group_shapes: bool = True
    body_word_wrap: bool = True
    remove_watermark: bool = True
    bundle_fonts: bool = True               # copy uploaded font files next to the output
    make_report: bool = True
    # tuning
    mask_pad: int = 4
    min_object_area: float = 0.004
    max_object_area: float = 0.45
    edge_expand: int = 4                    # px a picture's cut-out may grow to take in its outline / sticker rim
    edge_feather: float = 1.0               # px of anti-aliased (soft) edge on picture cut-outs; 0 = hard edge
    # infra
    cache_dir: Optional[str] = None         # None -> <out_dir>/.unflatten-cache
    out_dir: Optional[str] = None
    workers: int = 2                        # Claude look-ahead threads
    ui_lang: str = "ko"

    # ---- derived ----
    @property
    def preset(self) -> dict:
        return PRESETS.get(self.mode, PRESETS["full"])

    @property
    def split_objects(self) -> bool:
        return self.preset["split_objects"]

    @property
    def object_detail(self) -> str:
        return self.preset["object_detail"]

    @property
    def profile(self) -> dict:
        return LANGUAGE_PROFILES.get(self.language, LANGUAGE_PROFILES["ko"])

    @property
    def easyocr_langs(self) -> List[str]:
        return self.profile["easyocr"]

    @property
    def lang_tag(self) -> str:
        return self.profile["lang_tag"]

    @property
    def regular_name(self) -> str:
        return self.font_regular or self.profile["font_regular"]

    @property
    def bold_name(self) -> str:
        return self.font_bold or self.profile["font_bold"]

    def resolved_api_key(self) -> Optional[str]:
        key = (self.api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'")
        return key or None

    def cache_key_fields(self) -> dict:
        """Config parts that change recognition results (not assembly). Which OCR/Claude backend actually ran
        is added by the pipeline (resolved at run time), so degraded runs never masquerade as full ones."""
        return {"mode": self.mode, "language": self.language, "dpi": self.pdf_dpi, "watermark": self.remove_watermark}

    def split_params(self) -> dict:
        """Parts that change only the erase/split step. Cached recognition (OCR, Claude) is reused when
        these change; just the cut-outs and the background are redone."""
        return {"edge_expand": self.edge_expand, "edge_feather": self.edge_feather, "mask_pad": self.mask_pad,
                "min_object_area": self.min_object_area, "max_object_area": self.max_object_area}

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("api_key", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def parse_slide_range(spec: Optional[str], total: int) -> List[int]:
    """'1-3,7' -> [1,2,3,7] (1-based, clipped to total). None/'' -> all."""
    if not spec or not str(spec).strip():
        return list(range(1, total + 1))
    out: List[int] = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a else 1
            b = int(b) if b else total
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted({i for i in out if 1 <= i <= total})


def estimate_cost_usd(n_slides: int, model: str) -> float:
    m = MODELS.get(model, MODELS[DEFAULT_MODEL])
    return n_slides * (EST_INPUT_TOKENS * m["in"] + EST_OUTPUT_TOKENS * m["out"]) / 1e6
