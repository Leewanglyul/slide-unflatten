"""All Claude API access: slide analysis (text grouping + correction + objects), fallback OCR, key check, cost."""
from __future__ import annotations

import base64
import io
import threading
from typing import List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel

from .config import MODELS, DEFAULT_MODEL
from .types import Line

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_SIDE = 1568          # the API downsizes larger images anyway; sending more just costs upload time / 5 MB limit


class BlockOut(BaseModel):
    line_ids: List[int]
    text: str
    role: str                 # title | body | label
    bold: bool
    align: str                # left | center | right
    bullet: bool
    highlights: List[str]
    highlight_color: Optional[str] = None   # hex like "FFF59D" when a marker/highlight colour is visible


class ObjOut(BaseModel):
    name: str
    box: List[int]            # x0, y0, x1, y1 px
    layer: int                # 0 = back


class SlideOut(BaseModel):
    blocks: List[BlockOut]
    objects: List[ObjOut]
    background_type: str      # flat | gradient | photo
    watermarks: List[List[int]]   # boxes of logos / "Made with ..." marks to erase


class OcrLineOut(BaseModel):
    box: List[int]
    text: str


class OcrOut(BaseModel):
    lines: List[OcrLineOut]


class ClaudeService:
    def __init__(self, cfg):
        import anthropic
        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=cfg.resolved_api_key())
        self.model = cfg.model if cfg.model in MODELS else DEFAULT_MODEL
        self.effort = cfg.effort
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()
        self._fallbacks_ok = True

    # ───────── cost ─────────
    def _account(self, resp):
        u = getattr(resp, "usage", None)
        if u is None:
            return
        with self._lock:
            self.input_tokens += getattr(u, "input_tokens", 0) or 0
            self.output_tokens += getattr(u, "output_tokens", 0) or 0

    @property
    def cost_usd(self) -> float:
        m = MODELS.get(self.model, MODELS[DEFAULT_MODEL])
        return (self.input_tokens * m["in"] + self.output_tokens * m["out"]) / 1e6

    # ───────── key check ─────────
    def validate(self) -> Tuple[bool, str]:
        key = self.cfg.resolved_api_key()
        if not key:
            return False, "ANTHROPIC_API_KEY not set"
        if not key.startswith("sk-ant-"):
            return False, "key should start with sk-ant-"
        try:
            self.client.models.retrieve(self.model)   # free call
            return True, f"ok ({self.model})"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:160]}"

    # ───────── low-level call ─────────
    @staticmethod
    def _prepare(img: Image.Image) -> Tuple[bytes, float]:
        """JPEG bytes no larger than MAX_SIDE and the scale applied (sent px / source px)."""
        scale = min(1.0, MAX_SIDE / float(max(img.width, img.height)))
        im = img.convert("RGB")
        if scale < 1.0:
            im = im.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), scale

    def _parse(self, img_bytes: bytes, prompt: str, schema, max_tokens: int = 16000):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(img_bytes).decode()}},
            {"type": "text", "text": prompt},
        ]
        kwargs = dict(model=self.model, max_tokens=max_tokens, output_config={"effort": self.effort},
                      messages=[{"role": "user", "content": content}], output_format=schema)
        resp = None
        if self._fallbacks_ok:
            try:   # server-side fallback: a safety-classifier refusal is re-run on another model automatically
                resp = self.client.beta.messages.parse(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            except TypeError:                        # SDK too old for the parameter
                self._fallbacks_ok = False
            except Exception as e:
                msg = str(e).lower()
                if type(e).__name__ in ("BadRequestError", "NotFoundError") and ("fallback" in msg or "beta" in msg):
                    self._fallbacks_ok = False       # only this feature is unsupported; retry without it
                else:
                    raise
        if resp is None:
            resp = self.client.messages.parse(**kwargs)
        self._account(resp)
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return None
        return resp.parsed_output

    # ───────── slide analysis ─────────
    def analyze(self, img: Image.Image, lines: List[Line], object_detail: str) -> Optional[SlideOut]:
        data, s = self._prepare(img)
        sw, sh = round(img.width * s), round(img.height * s)
        listing = "\n".join(f"{i}: box={tuple(int(v * s) for v in l.box)} text={l.text!r}" for i, l in enumerate(lines))
        prompt = _analysis_prompt(sw, sh, listing, object_detail, self.cfg.remove_watermark)
        out = self._parse(data, prompt, SlideOut)
        if out is None or s >= 1.0:
            return out
        inv = 1.0 / s                                 # map boxes back to source pixels
        for o in out.objects:
            o.box = [int(round(v * inv)) for v in o.box]
        out.watermarks = [[int(round(v * inv)) for v in w] for w in out.watermarks]
        return out

    # ───────── fallback OCR ─────────
    def ocr_lines(self, img: Image.Image):
        data, s = self._prepare(img)
        sw, sh = round(img.width * s), round(img.height * s)
        prompt = (f"이 이미지는 {sw}x{sh} 픽셀의 슬라이드입니다. 보이는 모든 텍스트 줄을 찾아 "
                  "lines 로 돌려주세요. 각 줄은 시각적으로 한 줄(줄바꿈 단위)이며 box=[x0,y0,x1,y1] 픽셀로 글자 영역을 "
                  "최대한 정확히 감싸세요. 워터마크·로고 글자는 제외합니다. 텍스트는 이미지에 쓰인 그대로 적습니다.")
        out = self._parse(data, prompt, OcrOut, max_tokens=8000)
        if out is None:
            return []
        inv = 1.0 / s
        return [([int(round(v * inv)) for v in l.box], l.text) for l in out.lines if len(l.box) == 4]


def _analysis_prompt(width: int, height: int, listing: str, object_detail: str, watermark: bool) -> str:
    if object_detail == "fine":
        obj = ("objects: 슬라이드의 그림 개체(삽화, 사진, 아이콘, 스티커, 장식 박스·말풍선)를 나열하세요. "
               "하나의 그림이 여러 사물로 이루어져 있으면(예: 저울 위의 동전·기름통·펭귄) 바탕 사물과 그 위의 사물을 각각 따로 나열하세요.")
    elif object_detail == "coarse":
        obj = ("objects: 슬라이드의 그림 개체(삽화, 사진, 아이콘, 스티커, 장식 박스·말풍선)를 나열하세요. "
               "하나로 보이는 그림(저울과 그 위 물건 전체, 사진 한 장, 스티커 하나)은 통째로 한 개체로 나열하세요.")
    else:
        obj = "objects: 빈 목록 [] 으로 두세요."
    wm = ("watermarks: 'Made with ...', 'Gemini', 'NotebookLM', 서비스 로고처럼 슬라이드 내용이 아닌 워터마크가 있으면 "
          "그 영역의 [x0,y0,x1,y1] 픽셀 박스(3px 여유)를 나열하세요. 없으면 []."
          if watermark else "watermarks: 빈 목록 [] 으로 두세요.")
    return f"""이 슬라이드 이미지({width}x{height}px)에서 OCR이 찾은 텍스트 줄 목록입니다(box는 x0,y0,x1,y1 픽셀).

{listing if listing else "(OCR 결과 없음)"}

작업:
1. blocks: 시각적으로 하나의 텍스트 덩어리(제목, 한 박스 안의 문단/목록, 라벨)에 속하는 줄들을 묶어 블록으로 만드세요. 같은 박스 안이라도 서로 다른 글머리 항목은 한 블록 안에서 줄바꿈(\\n)으로 구분합니다.
2. text: OCR 글자는 틀린 경우가 많으니(특히 꾸민 제목 글씨) 반드시 이미지를 직접 읽고 정확한 문장을 쓰세요. 이미지에 쓰인 언어 그대로 적고, 줄바꿈은 이미지와 같은 위치에 \\n 으로 넣으세요. 글머리 기호(●, •, -)는 text에 넣지 말고 bullet=true 로 표시하세요.
3. role: "title"(슬라이드 제목), "body"(본문), "label"(그림에 붙은 짧은 라벨) 중 하나.
4. bold: 굵은 글씨면 true. align: "left" | "center" | "right".
5. highlights: 형광펜 강조나 색이 다른 강조 단어가 있으면 그 단어들. highlight_color: 형광펜 색이 보이면 hex(예 "FFF59D"), 아니면 null.
6. 텍스트가 아닌 것(그림 속 무늬, 워터마크 로고 글자 등)으로 잘못 인식된 줄은 어느 블록에도 넣지 마세요.
7. OCR이 놓친 텍스트는 새로 만들지 말고 무시하세요.
8. background_type: 슬라이드 바탕이 단색이면 "flat", 그라데이션이면 "gradient", 사진·복잡한 패턴이면 "photo".
9. {wm}
10. {obj}
- box는 개체 전체를 감싸는 [x0, y0, x1, y1] 픽셀 좌표(2~3px 여유). layer: 겹칠 때의 앞뒤 순서(0 = 가장 뒤). 박스·말풍선·액자는 그 위의 내용보다 뒤.
- 슬라이드 전체 배경, 텍스트 글자 자체, 폭이 이미지의 2% 미만인 아주 작은 점·반짝이는 제외하세요.
- name은 슬라이드 언어로 짧게(예: "펭귄", "왼쪽 사진")."""
