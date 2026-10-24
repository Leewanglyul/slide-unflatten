"""Before/after preview images and an HTML report of what happened per slide."""
from __future__ import annotations

import base64
import html
import io
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from .types import Block, Piece, SlideSummary  # noqa: F401


def render_preview(background: Image.Image, pieces: List[Piece], blocks: List[Block], measurer) -> Image.Image:
    """Approximate what PowerPoint will show (composite layers + text drawn with the measurement font)."""
    canvas = background.convert("RGBA").copy()
    for p in pieces:
        if p.rgba is None:
            continue
        im = Image.fromarray(p.rgba).convert("RGBA")
        canvas.alpha_composite(im, (max(0, p.x), max(0, p.y)))
    draw = ImageDraw.Draw(canvas)
    for b in blocks:
        heavy = b.bold or b.role == "title"
        base = measurer.font(heavy) if measurer else None
        size = max(6, int(round(b.font_px)))
        try:
            font = ImageFont.truetype(base.path, size) if base is not None else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        x, y = b.box[0], b.box[1]
        pitch = b.pitch_px if b.pitch_px > 0 else b.font_px * 1.2
        for k, para in enumerate(b.text.split("\n")):
            tw = draw.textlength(para, font=font) if hasattr(draw, "textlength") else 0
            bw = b.box[2] - b.box[0]
            xx = x
            if b.align == "center":
                xx = x + (bw - tw) / 2
            elif b.align == "right":
                xx = x + bw - tw
            prefix = "• " if b.bullet else ""
            draw.text((xx, y + k * pitch), prefix + para, fill=tuple(b.color) + (255,), font=font)
    return canvas.convert("RGB")


def side_by_side(src: Image.Image, preview: Image.Image, width: int = 640) -> Image.Image:
    h = int(src.height * width / src.width)
    a = src.resize((width, h), Image.LANCZOS)
    b = preview.resize((width, h), Image.LANCZOS)
    out = Image.new("RGB", (width * 2 + 8, h), (40, 40, 40))
    out.paste(a, (0, 0)); out.paste(b, (width + 8, 0))
    return out


def _b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html_report(path: str, title: str, summaries: List[SlideSummary], previews: List[Optional[str]],
                      cost_line: str, lang: str = "ko") -> str:
    """`previews` are image paths (or None) matching `summaries`."""
    ko = lang == "ko"
    head = ["#", "상태" if ko else "status", "줄" if ko else "lines", "블록" if ko else "blocks",
            "그림" if ko else "pieces", "시간(s)" if ko else "sec", "경고" if ko else "warnings"]
    rows = []
    for s, pv in zip(summaries, previews):
        cls = {"ok": "ok", "text_layer": "ok", "passthrough": "pass", "failed": "fail"}.get(s.status, "")
        warn = "; ".join(s.warnings)
        img_html = f'<img src="{_b64(Image.open(pv))}" loading="lazy">' if pv else ""
        texts = "<br>".join(html.escape(t) for t in s.texts[:12])
        rows.append(f"""<tr class="{cls}"><td>{s.index}</td><td>{html.escape(s.status)}</td><td>{s.n_lines}</td>
<td>{s.n_blocks}</td><td>{s.n_pieces}</td><td>{s.seconds:.1f}</td><td>{html.escape(warn)}</td></tr>
<tr class="{cls} detail"><td colspan="7">{img_html}<div class="texts">{texts}</div></td></tr>""")
    doc = f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,'Malgun Gothic',sans-serif;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #ddd;padding:6px 8px;font-size:14px;vertical-align:top}}
th{{text-align:left;background:#eee}} tr.fail td{{background:#fde8e8}} tr.pass td{{color:#777}}
tr.detail td{{padding:8px 8px 20px}} img{{max-width:100%;border:1px solid #ccc;border-radius:4px}}
.texts{{font-size:12px;color:#555;margin-top:6px;white-space:pre-wrap}} .cost{{color:#555;margin:8px 0 16px}}
</style></head><body><h1>{html.escape(title)}</h1><p class="cost">{html.escape(cost_line)}</p>
<p>{'왼쪽: 원본 · 오른쪽: 변환 결과 미리보기 (글꼴은 PowerPoint에서 다르게 보일 수 있음)' if ko else 'Left: original · right: preview of the editable result'}</p>
<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(rows)}</table></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
