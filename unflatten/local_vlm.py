"""Local vision-language model (no API key, nothing leaves the computer): Ollama + an open model such as qwen3-vl.

Drop-in for ClaudeService in the pipeline (same attributes and analyze/ocr_lines/validate), but it is asked only
for what a local model does reliably: grouping OCR lines into blocks, correcting their text by reading the image,
roles / bold / align / bullets, background type. Picture objects come from the local detector (detect.py) and
watermarks from the rule (watermark.py), because open models often answer boxes in 0-1000 relative coordinates
instead of pixels.

Setup (Windows): install Ollama (https://ollama.com) → `ollama pull qwen3-vl:8b` → Config(vision_backend="ollama").
"""
from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from PIL import Image

from .claude_client import OcrOut, SlideOut, _analysis_prompt
from .types import Line

MAX_SIDE = 1280           # smaller than for Claude: image tokens dominate a local model's time and memory
TIMEOUT_SEC = 600


def _inline_refs(schema: dict) -> dict:
    """Pydantic puts nested models under $defs/$ref; inline them so any JSON-schema-to-grammar converter copes."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


class LocalVLMService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.local_vlm_model
        self.url = cfg.ollama_url.rstrip("/")
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    # ───────── HTTP ─────────
    def _post(self, path: str, body: dict, timeout: float = TIMEOUT_SEC) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path: str, timeout: float = 5) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def validate(self) -> Tuple[bool, str]:
        try:
            tags = self._get("/api/tags")
        except Exception as e:
            return False, f"Ollama not reachable at {self.url} ({type(e).__name__}). Install/start Ollama"
        names = {m.get("name", "") for m in tags.get("models", [])}
        want = self.model if ":" in self.model else self.model + ":latest"
        if want not in names:
            return False, f"model {self.model} not pulled. Run: ollama pull {self.model}"
        return True, f"ok (local {self.model})"

    # ───────── call ─────────
    @staticmethod
    def _prepare(img: Image.Image) -> Tuple[str, float]:
        import io
        scale = min(1.0, MAX_SIDE / float(max(img.width, img.height)))
        im = img.convert("RGB")
        if scale < 1.0:
            im = im.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode(), scale

    def _chat(self, image_b64: str, prompt: str, schema_model):
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": prompt + "\n\nJSON으로만 답하세요.", "images": [image_b64]}],
                "format": _inline_refs(schema_model.model_json_schema()),
                "options": {"temperature": 0, "num_ctx": self.cfg.local_vlm_ctx}}
        resp = self._post("/api/chat", body)
        with self._lock:
            self.input_tokens += int(resp.get("prompt_eval_count") or 0)
            self.output_tokens += int(resp.get("eval_count") or 0)
        content = (resp.get("message") or {}).get("content") or ""
        try:
            return schema_model.model_validate_json(content)
        except Exception:
            return None     # malformed answer → the pipeline falls back to heuristics for this slide

    # ───────── slide analysis ─────────
    def analyze(self, img: Image.Image, lines: List[Line], object_detail: str) -> Optional[SlideOut]:
        data, s = self._prepare(img)
        sw, sh = round(img.width * s), round(img.height * s)
        listing = "\n".join(f"{i}: box={tuple(int(v * s) for v in l.box)} text={l.text!r}" for i, l in enumerate(lines))
        # objects and watermarks are left to detect.py / watermark.py (see module docstring)
        prompt = _analysis_prompt(sw, sh, listing, "none", False)
        out = self._chat(data, prompt, SlideOut)
        if out is None:
            return None
        n = len(lines)
        for b in out.blocks:                                   # drop ids a small model may invent
            b.line_ids = [i for i in b.line_ids if 0 <= i < n]
        out.blocks = [b for b in out.blocks if b.line_ids]
        out.objects, out.watermarks = [], []
        if out.background_type not in ("flat", "gradient", "photo"):
            out.background_type = "flat"
        return out

    def ocr_lines(self, img: Image.Image):
        """Last-resort OCR when no OCR engine is installed. Box quality depends on the model."""
        data, s = self._prepare(img)
        sw, sh = round(img.width * s), round(img.height * s)
        prompt = (f"이 이미지는 {sw}x{sh} 픽셀의 슬라이드입니다. 보이는 모든 텍스트 줄을 찾아 lines 로 돌려주세요. "
                  "각 줄은 box=[x0,y0,x1,y1] 픽셀 좌표로 감싸고, 텍스트는 이미지에 쓰인 그대로 적습니다.")
        out = self._chat(data, prompt, OcrOut)
        if out is None:
            return []
        inv = 1.0 / s
        return [([int(round(v * inv)) for v in l.box], l.text) for l in out.lines if len(l.box) == 4]
