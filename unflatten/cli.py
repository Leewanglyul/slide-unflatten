"""Command line: convert · estimate · rebuild · ui · doctor"""
from __future__ import annotations

import argparse
import os
import sys

from .config import Config, MODELS, PRESETS, LANGUAGE_PROFILES


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--mode", choices=list(PRESETS), default="full", help="text | objects | full")
    p.add_argument("--lang", choices=list(LANGUAGE_PROFILES), default="ko", help="slide language profile")
    p.add_argument("--model", choices=list(MODELS), default="claude-opus-5-5")
    p.add_argument("--no-claude", action="store_true", help="heuristics only (no API calls)")
    p.add_argument("--ocr", choices=["auto", "easyocr", "claude", "none"], default="auto")
    p.add_argument("--font", help="regular .ttf/.otf for body text")
    p.add_argument("--font-bold", help="bold .ttf/.otf for titles/emphasis")
    p.add_argument("--font-name", help="font name to write when no font file is given")
    p.add_argument("--slides", help="range, e.g. 1-3,7")
    p.add_argument("--out", help="output directory")
    p.add_argument("--cache", help="cache directory")
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--edge-expand", type=int, default=4,
                   help="px a picture cut-out may grow to take in its outline/sticker rim (0 = off)")
    p.add_argument("--edge-feather", type=float, default=1.0,
                   help="px of soft (anti-aliased) edge on picture cut-outs (0 = hard edge)")
    p.add_argument("--no-hidden-original", action="store_true")
    p.add_argument("--no-groups", action="store_true")
    p.add_argument("--no-theme-fonts", action="store_true")
    p.add_argument("--no-watermark-removal", action="store_true")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--ui-lang", choices=["ko", "en"], default="ko")


def cfg_from_args(a) -> Config:
    return Config(mode=a.mode, language=a.lang, model=a.model, use_claude=not a.no_claude, ocr_backend=a.ocr,
                  font_regular_file=a.font, font_bold_file=a.font_bold, font_regular=a.font_name, font_bold=a.font_name,
                  slide_range=a.slides, out_dir=a.out, cache_dir=a.cache, upscale=a.upscale,
                  keep_hidden_original=not a.no_hidden_original, group_shapes=not a.no_groups,
                  use_theme_fonts=not a.no_theme_fonts, remove_watermark=not a.no_watermark_removal,
                  make_report=not a.no_report, ui_lang=a.ui_lang,
                  edge_expand=a.edge_expand, edge_feather=a.edge_feather)


def print_summary(res):
    print()
    print(f"{'#':>3}  {'status':<11}{'lines':>6}{'blocks':>7}{'pieces':>7}{'sec':>7}  warnings")
    for s in res.summaries:
        print(f"{s.index:>3}  {s.status:<11}{s.n_lines:>6}{s.n_blocks:>7}{s.n_pieces:>7}{s.seconds:>7.1f}  {'; '.join(s.warnings)}")
    print(f"\n{res.out_path}")
    if res.report_path:
        print(res.report_path)
    print(res.cost_line("en"))


def doctor():
    import importlib
    print("python", sys.version.split()[0])
    for m, what in [("pptx", "PPTX"), ("pymupdf", "PDF"), ("cv2", "OpenCV"), ("fontTools", "fonts"),
                    ("anthropic", "Claude API"), ("easyocr", "OCR (optional)"), ("ultralytics", "SAM 2 (optional)"),
                    ("simple_lama_inpainting", "LaMa (optional)"), ("gradio", "UI (optional)")]:
        try:
            importlib.import_module(m)
            print("  ✅", what, f"({m})")
        except Exception as e:
            print("  ⚠️ ", what, f"({m}) →", type(e).__name__)
    try:
        import torch
        print("  GPU:", "yes" if torch.cuda.is_available() else "no")
    except Exception:
        print("  GPU: torch not installed")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            from .claude_client import ClaudeService
            ok, msg = ClaudeService(Config()).validate()
            print("  API key:", "✅" if ok else "❌", msg)
        except Exception as e:
            print("  API key: ❌", e)
    else:
        print("  API key: not set (ANTHROPIC_API_KEY) → heuristic mode")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="unflatten", description="Image-only slides → editable PPTX")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert PPTX/PDF/images")
    c.add_argument("files", nargs="+")
    c.add_argument("--zip", action="store_true", help="bundle pptx + report + fonts into a zip")
    _add_common(c)

    e = sub.add_parser("estimate", help="count slides, estimate time and cost")
    e.add_argument("files", nargs="+")
    _add_common(e)

    r = sub.add_parser("rebuild", help="re-assemble from a manifest (fonts/grouping changes, no re-recognition)")
    r.add_argument("manifest")
    r.add_argument("--font"); r.add_argument("--font-bold"); r.add_argument("--font-name")
    r.add_argument("--no-groups", action="store_true"); r.add_argument("--no-theme-fonts", action="store_true")
    r.add_argument("--no-hidden-original", action="store_true"); r.add_argument("--upscale", type=int)
    r.add_argument("--edge-expand", type=int); r.add_argument("--edge-feather", type=float)

    u = sub.add_parser("ui", help="launch the Gradio web UI")
    u.add_argument("--share", action="store_true")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--ui-lang", choices=["ko", "en"], default="ko")

    sub.add_parser("doctor", help="check dependencies, GPU and API key")

    a = ap.parse_args(argv)
    if a.cmd == "doctor":
        doctor(); return 0
    if a.cmd == "ui":
        from .app import launch
        launch(share=a.share, port=a.port, ui_lang=a.ui_lang); return 0
    if a.cmd == "estimate":
        from .pipeline import estimate
        cfg = cfg_from_args(a)
        for f in a.files:
            info = estimate(f, cfg)
            print(f"{f}: {info['line']}  (flat {info['n_flat']}, pdf-text {info['n_text_layer']}, kept {info['n_pass']})")
        return 0
    if a.cmd == "rebuild":
        from .pipeline import rebuild
        ov = {}
        if a.font: ov["font_regular_file"] = a.font
        if a.font_bold: ov["font_bold_file"] = a.font_bold
        if a.font_name: ov["font_regular"] = ov["font_bold"] = a.font_name
        if a.no_groups: ov["group_shapes"] = False
        if a.no_theme_fonts: ov["use_theme_fonts"] = False
        if a.no_hidden_original: ov["keep_hidden_original"] = False
        if a.upscale: ov["upscale"] = a.upscale
        if a.edge_expand is not None: ov["edge_expand"] = a.edge_expand
        if a.edge_feather is not None: ov["edge_feather"] = a.edge_feather
        res = rebuild(a.manifest, ov)
        print_summary(res); return 0
    if a.cmd == "convert":
        from .pipeline import convert, bundle_zip
        cfg = cfg_from_args(a)
        code = 0
        for f in a.files:
            try:
                res = convert(f, cfg)
                print_summary(res)
                if a.zip:
                    print(bundle_zip(res))
                if any(s.status == "failed" for s in res.summaries):
                    code = 2
            except Exception as ex:
                print(f"❌ {f}: {type(ex).__name__}: {ex}")
                code = 1
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
