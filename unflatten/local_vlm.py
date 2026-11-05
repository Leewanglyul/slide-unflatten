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
from pydantic import BaseModel

from .claude_client import BlockOut, OcrOut, SlideOut, _analysis_prompt
from .types import Line

class LocalBlock(BaseModel):
    """Short answer for a small local model: every field costs tokens, and long lists invite repetition."""
    line_ids: List[int]
    text: str
    role: str                 # title | body | label
    bold: bool
    align: str                # left | center | right
    bullet: bool


class LocalSlideOut(BaseModel):
    blocks: List[LocalBlock]
    background_type: str      # flat | gradient | photo


COMPACT = ("\n\n/no_think\n위 1~8번만 하세요(형광펜·워터마크·그림 개체는 답하지 않음). "
           "생각 과정 없이, 줄바꿈과 들여쓰기 없는 한 줄 JSON으로만 답하세요.")


class LocalVLMError(RuntimeError):
    """The local model answered, but not with usable JSON (logged by the pipeline, slide falls back to rules)."""


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

    # 1st try deterministic; if the answer runs away (a small model can repeat itself inside a JSON string until it
    # hits the limit) or is not valid JSON, try once more with a little randomness and a repetition penalty
    ATTEMPTS = ({"temperature": 0},
                {"temperature": 0.3, "repeat_penalty": 1.15, "repeat_last_n": 256})

    def _chat(self, image_b64: str, prompt: str, schema_model):
        errors = []
        for opts in self.ATTEMPTS:
            try:
                return self._chat_once(image_b64, prompt, schema_model, opts)
            except LocalVLMError as e:
                errors.append(str(e))
        raise LocalVLMError(" / retry: ".join(errors))

    def _chat_once(self, image_b64: str, prompt: str, schema_model, opts: dict):
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": prompt + COMPACT, "images": [image_b64]}],
                "format": _inline_refs(schema_model.model_json_schema()),
                # num_predict caps the answer: a whole slide fits in well under 4096 tokens, a runaway answer does not
                "options": {"num_ctx": self.cfg.local_vlm_ctx, "num_predict": self.cfg.local_vlm_max_answer, **opts},
                # qwen3-vl and similar "thinking" models otherwise write a long reasoning trace before the JSON,
                # which takes minutes per slide on a local GPU; a model without the switch rejects it → retry
                "think": False}
        try:
            resp = self._post("/api/chat", body)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            body.pop("think")
            resp = self._post("/api/chat", body)
        with self._lock:
            self.input_tokens += int(resp.get("prompt_eval_count") or 0)
            self.output_tokens += int(resp.get("eval_count") or 0)
        msg = resp.get("message") or {}
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
        if resp.get("done_reason") == "length" and not content.strip() and thinking:
            raise LocalVLMError(f"model spent all {resp.get('eval_count')} tokens thinking (no answer); "
                                f"use a non-thinking model tag or a newer Ollama")
        if resp.get("done_reason") == "length":
            # the tail shows what the model was repeating (for tuning the prompt / schema later)
            raise LocalVLMError(f"answer cut off at {resp.get('eval_count')} tokens "
                                f"(limit {self.cfg.local_vlm_max_answer}, prompt {resp.get('prompt_eval_count')}); "
                                f"ends {content[-80:]!r}")
        try:
            return schema_model.model_validate_json(content)
        except Exception as e:   # malformed answer → the pipeline logs it and uses heuristics for this slide
            raise LocalVLMError(f"invalid JSON ({type(e).__name__}: {str(e).splitlines()[0][:80]}); "
                                f"answer starts {content[:60]!r}") from e

    # ───────── slide analysis ─────────
    def analyze(self, img: Image.Image, lines: List[Line], object_detail: str) -> Optional[SlideOut]:
        data, s = self._prepare(img)
        sw, sh = round(img.width * s), round(img.height * s)
        listing = "\n".join(f"{i}: box={tuple(int(v * s) for v in l.box)} text={l.text!r}" for i, l in enumerate(lines))
        # objects and watermarks are left to detect.py / watermark.py (see module docstring)
        prompt = _analysis_prompt(sw, sh, listing, "none", False)
        short = self._chat(data, prompt, LocalSlideOut)
        n = len(lines)
        blocks = []
        for b in short.blocks:
            ids = [i for i in b.line_ids if 0 <= i < n]      # drop ids a small model may invent
            if ids:
                blocks.append(BlockOut(line_ids=ids, text=b.text, role=b.role, bold=b.bold, align=b.align,
                                       bullet=b.bullet, highlights=[], highlight_color=None))
        bg = short.background_type if short.background_type in ("flat", "gradient", "photo") else "flat"
        return SlideOut(blocks=blocks, objects=[], background_type=bg, watermarks=[])

    def ocr_lines(self, img: Image.Image):
        """Last-resort OCR when no OCR engine is installed. Box quality depends on the model."""
        data, s = self._prepare(img)
        sw, sh = round(img.width * s), round(img.height * s)
        prompt = (f"이 이미지는 {sw}x{sh} 픽셀의 슬라이드입니다. 보이는 모든 텍스트 줄을 찾아 lines 로 돌려주세요. "
                  "각 줄은 box=[x0,y0,x1,y1] 픽셀 좌표로 감싸고, 텍스트는 이미지에 쓰인 그대로 적습니다.")
        try:
            out = self._chat(data, prompt, OcrOut)
        except LocalVLMError:
            return []
        inv = 1.0 / s
        return [([int(round(v * inv)) for v in l.box], l.text) for l in out.lines if len(l.box) == 4]
