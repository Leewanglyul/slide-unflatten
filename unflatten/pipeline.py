"""Orchestration: load → recognise → erase → split → assemble, with per-slide isolation, cache and report."""
from __future__ import annotations

import json
import os
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from . import builder
from .cache import SlideCache
from .config import Config, estimate_cost_usd, parse_slide_range, USD_KRW
from .fonts import bundle_fonts, plan_fonts
from .i18n import t
from .imageops import classify_background, dominant_colors
from .inputs import Deck, SlideInput, load_deck
from .inpaint import box_mask, inpaint, text_mask
from .layout import FontMeasurer, blocks_from_claude, finalize_blocks, group_heuristic
from .ocr import NullOCRBackend, make_backend
from .report import render_preview, side_by_side, write_html_report
from .segment import pieces_from_embedded, split_semantic, split_simple
from .types import Piece, SlideAnalysis, SlideSummary

Progress = Callable[[int, int, str], None]
SEC_PER_SLIDE_GPU = 15
SEC_PER_SLIDE_CPU = 90


@dataclass
class Result:
    out_path: str
    manifest_path: str
    report_path: Optional[str]
    summaries: List[SlideSummary]
    previews: List[Optional[Image.Image]] = field(default_factory=list)
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    used_claude: bool = False
    font_files: List[str] = field(default_factory=list)

    def cost_line(self, lang="ko") -> str:
        if not self.used_claude:
            return t(lang, "warn_noclaude")
        return t(lang, "cost", inp=self.input_tokens, out=self.output_tokens, usd=self.cost_usd)


# ───────────── estimate ─────────────
def estimate(path: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    deck = load_deck(path, dpi=cfg.pdf_dpi)
    wanted = set(parse_slide_range(cfg.slide_range, len(deck.slides)))
    flat = [s for s in deck.slides if s.kind == "flat" and s.index in wanted]
    text_layer = [s for s in deck.slides if s.kind == "text_layer" and s.index in wanted]
    passthrough = [s for s in deck.slides if s.kind == "passthrough"]
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        pass
    per = SEC_PER_SLIDE_GPU if gpu else SEC_PER_SLIDE_CPU
    if cfg.mode == "text":
        per *= 0.5
    minutes = max(1, round((len(flat) * per + len(text_layer) * 2) / 60))
    usd = estimate_cost_usd(len(flat), cfg.model) if cfg.use_claude else 0.0
    return {"n_slides": len(deck.slides), "n_flat": len(flat), "n_text_layer": len(text_layer),
            "n_pass": len(passthrough), "minutes": minutes, "usd": usd, "krw": usd * USD_KRW, "gpu": gpu,
            "line": t(cfg.ui_lang, "estimate", n=len(flat) + len(text_layer), min=minutes, usd=usd, krw=usd * USD_KRW)}


# ───────────── per-slide recognition ─────────────
def _recognise(si: SlideInput, cfg: Config, ocr, service, log) -> SlideAnalysis:
    img = si.image
    if si.kind == "text_layer":
        lines = si.lines or []
        blocks = group_heuristic(lines, img_h=img.height)
        return SlideAnalysis(lines=lines, blocks=blocks, objects=[], background_type=classify_background(img),
                             watermarks=[], used_claude=False, source="text_layer")
    lines = ocr.lines(img)
    return SlideAnalysis(lines=lines, blocks=[], objects=[], background_type="unknown", source=ocr.name)


def _analyse_with_claude(si: SlideInput, analysis: SlideAnalysis, cfg: Config, service):
    out = service.analyze(si.image, analysis.lines, cfg.object_detail)
    if out is None:
        return None
    blocks, objects, bg, wms = blocks_from_claude(out, analysis.lines)
    return blocks, objects, bg, wms


def _erase_and_split(si: SlideInput, analysis: SlideAnalysis, cfg: Config, log):
    img = si.image
    arr = np.array(img)
    kept = sorted({i for b in analysis.blocks for i in b.lines})
    if si.kind == "text_layer" and si.clean_image is not None:
        clean = si.clean_image
    else:
        log("  " + t(cfg.ui_lang, "erase", n=len(kept)))
        mask = text_mask(arr.shape, [analysis.lines[i] for i in kept], arr, cfg.mask_pad)
        clean = inpaint(img, mask, log)
    if cfg.remove_watermark and analysis.watermarks:
        log("  " + t(cfg.ui_lang, "watermark", n=len(analysis.watermarks)))
        clean = inpaint(clean, box_mask(arr.shape, analysis.watermarks), log)
    pieces: List[Piece] = []
    background = clean
    if si.kind == "text_layer" and si.embedded_images and cfg.split_objects:
        pieces = pieces_from_embedded(si.embedded_images)
    elif cfg.split_objects:
        if analysis.background_type == "photo":
            log("  " + t(cfg.ui_lang, "split_skip_photo"))
        else:
            if cfg.object_detail in ("coarse", "fine") and analysis.objects:
                log("  " + t(cfg.ui_lang, "split_sam", names=", ".join(o["name"] for o in analysis.objects)))
                try:
                    background, pieces = split_semantic(clean, analysis.objects, cfg, log)
                except Exception as e:
                    log("  " + t(cfg.ui_lang, "sam_fail", err=str(e)[:120]))
            if not pieces:
                log("  " + t(cfg.ui_lang, "split_simple"))
                background, pieces = split_simple(clean, cfg, log)
    return background, pieces


# ───────────── main ─────────────
def convert(path: str, cfg: Optional[Config] = None, out_path: Optional[str] = None,
            progress: Optional[Progress] = None, log=print, ocr_override=None) -> Result:
    cfg = cfg or Config()
    t0 = time.time()
    lang = cfg.ui_lang
    progress = progress or (lambda d, n, m: None)

    deck = load_deck(path, dpi=cfg.pdf_dpi)
    wanted = set(parse_slide_range(cfg.slide_range, len(deck.slides)))
    out_dir = cfg.out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = out_path or os.path.join(out_dir, f"{stem}_editable.pptx")
    cache = SlideCache(cfg.cache_dir or os.path.join(out_dir, ".unflatten-cache"))

    # services
    service = None
    if cfg.use_claude and cfg.resolved_api_key():
        try:
            from .claude_client import ClaudeService
            service = ClaudeService(cfg)
            ok, msg = service.validate()
            if not ok:
                log(t(lang, "claude_fail", err=msg))
                service = None
        except Exception as e:
            log(t(lang, "claude_fail", err=str(e)[:160]))
            service = None
    ocr = ocr_override or (None)
    plan = plan_fonts(cfg)
    measurer = FontMeasurer(cfg)

    work = [s for s in deck.slides if s.kind in ("flat", "text_layer") and s.index in wanted]
    total = len(work)
    summaries: List[SlideSummary] = []
    previews: List[Optional[Image.Image]] = []
    results: Dict[int, tuple] = {}     # index -> (analysis, background, pieces, cache_key)

    # stage A: recognition (OCR / text layer / cache lookup)
    pending_claude = {}
    for n, si in enumerate(work, 1):
        key = SlideCache.key(si.image, cfg)
        if cache.has(key):
            analysis, background, pieces = cache.load(key)
            results[si.index] = (analysis, background, pieces, key, True)
            progress(n - 1, total, f"[{si.index}] {t(lang, 'cached')}")
            continue
        progress(n - 1, total, f"[{si.index}] {t(lang, 'text_layer' if si.kind == 'text_layer' else 'ocr')}")
        log(f"[{si.index}/{len(deck.slides)}] " + t(lang, 'text_layer' if si.kind == 'text_layer' else 'ocr'))
        if ocr is None and si.kind == "flat":
            ocr = make_backend(cfg, service, log)
        try:
            analysis = _recognise(si, cfg, ocr or NullOCRBackend(), service, log)
        except Exception as e:
            log(t(lang, "slide_fail", i=si.index, err=f"{type(e).__name__}: {e}"))
            results[si.index] = (None, None, None, key, False)
            continue
        results[si.index] = (analysis, None, None, key, False)
        if si.kind == "flat" and service is not None:
            pending_claude[si.index] = (si, analysis)

    # stage B: Claude analysis in parallel threads (network bound) ...
    futures = {}
    if pending_claude:
        pool = ThreadPoolExecutor(max_workers=max(1, cfg.workers))
        for idx, (si, analysis) in pending_claude.items():
            futures[idx] = pool.submit(_analyse_with_claude, si, analysis, cfg, service)
    else:
        pool = None

    # stage C: ... while erase / split / build run slide by slide
    for n, si in enumerate(work, 1):
        ts = time.time()
        analysis, background, pieces, key, cached = results[si.index]
        summ = SlideSummary(index=si.index, status="ok", cache_key=key)
        try:
            if analysis is None:
                raise RuntimeError("recognition failed")
            if not cached:
                if si.index in futures:
                    progress(n - 1, total, f"[{si.index}] {t(lang, 'claude')}")
                    try:
                        got = futures[si.index].result()
                    except Exception as e:
                        got = None
                        log("  " + t(lang, "claude_fail", err=f"{type(e).__name__}: {str(e)[:120]}"))
                        if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                            log("  " + t(lang, "claude_auth"))
                            service = None
                    if got is not None:
                        analysis.blocks, analysis.objects, analysis.background_type, analysis.watermarks = got
                        analysis.used_claude = True
                if not analysis.blocks:
                    analysis.blocks = group_heuristic(analysis.lines, img_h=si.image.height)
                if analysis.background_type == "unknown":
                    analysis.background_type = classify_background(si.image)
                arr = np.array(si.image)
                analysis.blocks = finalize_blocks(analysis.blocks, analysis.lines, arr, measurer)
                progress(n - 1, total, f"[{si.index}] {t(lang, 'erase', n=len(analysis.lines))}")
                background, pieces = _erase_and_split(si, analysis, cfg, log)
                cache.save(key, analysis, background, pieces)
            summ.status = "text_layer" if si.kind == "text_layer" else "ok"
            summ.n_lines = len(analysis.lines)
            summ.n_blocks = len(analysis.blocks)
            summ.n_pieces = len(pieces)
            summ.low_conf = sum(1 for l in analysis.lines if l.conf < 0.5)
            summ.texts = [b.text.replace("\n", " / ") for b in analysis.blocks]
            if summ.low_conf:
                summ.warnings.append(t(lang, "warn_lowconf", n=summ.low_conf))
            if si.kind == "flat" and not analysis.used_claude:
                summ.warnings.append(t(lang, "warn_noclaude"))
            results[si.index] = (analysis, background, pieces, key, True)
            if cfg.make_report:
                pv = render_preview(background, pieces, analysis.blocks, measurer)
                previews.append(side_by_side(si.image, pv))
            else:
                previews.append(None)
        except Exception as e:
            log(t(lang, "slide_fail", i=si.index, err=f"{type(e).__name__}: {e}"))
            log(traceback.format_exc(limit=3))
            summ.status = "failed"
            summ.warnings.append(f"{type(e).__name__}: {str(e)[:100]}")
            results[si.index] = (None, None, None, key, False)
            previews.append(side_by_side(si.image, si.image) if cfg.make_report else None)
        summ.seconds = time.time() - ts
        summaries.append(summ)
        progress(n, total, f"[{si.index}] {t(lang, 'done', path='')}".rstrip(": "))
    if pool:
        pool.shutdown(wait=False)

    # stage D: assemble the deck
    _assemble(deck, results, wanted, cfg, plan, out_path)

    # passthrough / skipped summaries for the report
    for s in deck.slides:
        if s.kind == "passthrough":
            summaries.append(SlideSummary(index=s.index, status="passthrough", warnings=[s.reason] if s.reason else []))
            previews.append(None)
        elif s.index not in wanted:
            summaries.append(SlideSummary(index=s.index, status="skipped"))
            previews.append(None)
    order = sorted(range(len(summaries)), key=lambda i: summaries[i].index)
    summaries = [summaries[i] for i in order]
    previews = [previews[i] for i in order]

    res = Result(out_path=out_path, manifest_path=os.path.join(out_dir, f"{stem}_manifest.json"), report_path=None,
                 summaries=summaries, previews=previews, seconds=time.time() - t0, used_claude=service is not None)
    if service is not None:
        res.cost_usd, res.input_tokens, res.output_tokens = service.cost_usd, service.input_tokens, service.output_tokens
    if cfg.bundle_fonts:
        res.font_files = bundle_fonts(plan, out_dir)
    if cfg.make_report:
        res.report_path = write_html_report(os.path.join(out_dir, f"{stem}_report.html"), f"{stem} — unflatten report",
                                            summaries, previews, res.cost_line(lang), lang)
        log(t(lang, "report", path=res.report_path))
    with open(res.manifest_path, "w", encoding="utf-8") as f:
        json.dump({"source": os.path.abspath(path), "output": os.path.abspath(out_path), "config": cfg.to_dict(),
                   "cache_dir": cache.root, "slides": [{"index": s.index, "status": s.status, "cache_key": s.cache_key,
                                                        "texts": s.texts} for s in summaries]},
                  f, ensure_ascii=False, indent=1)
    log(t(lang, "done", path=out_path))
    if service is not None:
        log(res.cost_line(lang))
    return res


def _assemble(deck: Deck, results, wanted, cfg: Config, plan, out_path: str):
    from pptx import Presentation
    sw, sh = deck.width_emu, deck.height_emu
    has_pass = any(s.kind == "passthrough" for s in deck.slides)
    if deck.kind == "pptx":
        prs = deck.source_prs
        theme_ok = not has_pass          # don't re-font a deck that still has the author's own slides
    else:
        prs = Presentation()
        prs.slide_width, prs.slide_height = sw, sh
        theme_ok = True
    if cfg.use_theme_fonts and theme_ok:
        imgs = [s.image for s in deck.slides if s.image is not None]
        builder.apply_theme(prs, plan, dominant_colors(imgs) if imgs else [])
    # when theme fonts can't be used, fall back to explicit names
    eff_cfg = cfg if (cfg.use_theme_fonts and theme_ok) else _with(cfg, use_theme_fonts=False)

    for si in deck.slides:
        if deck.kind == "pptx":
            slide = prs.slides[si.index - 1]
            if si.kind == "passthrough":
                continue
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            if si.notes:
                slide.notes_slide.notes_text_frame.text = si.notes
        if si.index not in wanted:
            if deck.kind != "pptx":
                builder.add_failed_slide(slide, si.image, si.rect_emu, (sw, sh), cfg)
            continue
        analysis, background, pieces, key, ok = results.get(si.index, (None, None, None, None, False))
        if deck.kind == "pptx":
            if si.pic_shape_id is not None:
                builder.remove_shape_by_id(slide, si.pic_shape_id)
            builder.remove_empty_placeholders(slide)
        if not ok or analysis is None:
            builder.add_failed_slide(slide, si.image, si.rect_emu, (sw, sh), cfg)
            continue
        builder.build_slide(slide, si.image, background, pieces, analysis.blocks, eff_cfg, plan,
                            si.rect_emu, (sw, sh), pdf_font_names=(si.kind == "text_layer"))
    prs.save(out_path)


def _with(cfg: Config, **kw) -> Config:
    d = cfg.to_dict()
    d.update(kw)
    c = Config.from_dict(d)
    c.api_key = cfg.api_key
    return c


# ───────────── rebuild from manifest (assembly-only changes: fonts, grouping, wrap…) ─────────────
def rebuild(manifest_path: str, overrides: Optional[dict] = None, progress: Optional[Progress] = None,
            log=print) -> Result:
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    cfg = Config.from_dict(man["config"])
    cfg.cache_dir = man.get("cache_dir") or cfg.cache_dir
    for k, v in (overrides or {}).items():
        if k in Config.__dataclass_fields__:
            setattr(cfg, k, v)
    return convert(man["source"], cfg, out_path=man.get("output"), progress=progress, log=log)


def bundle_zip(res: Result) -> str:
    zpath = os.path.splitext(res.out_path)[0] + "_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(res.out_path, os.path.basename(res.out_path))
        if res.report_path and os.path.exists(res.report_path):
            z.write(res.report_path, os.path.basename(res.report_path))
        for fpath in res.font_files:
            z.write(fpath, "fonts/" + os.path.basename(fpath))
        if os.path.exists(res.manifest_path):
            z.write(res.manifest_path, os.path.basename(res.manifest_path))
    return zpath
