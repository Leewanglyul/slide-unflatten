"""One-screen Gradio UI: upload → options → estimate → convert → preview → download. Also rebuild + text fixes."""
from __future__ import annotations

import json
import os
import tempfile
from typing import List

from .config import Config, LANGUAGE_PROFILES, MODELS, PRESETS
from .fonts import assign_font_files

UI = {
    "ko": dict(title="🧩 이미지 슬라이드 → 편집 가능한 PPTX", sub="NotebookLM·Gemini·Canva·Gamma 등에서 받은 통이미지 슬라이드를 글자와 그림을 고칠 수 있는 PPTX로 바꿉니다.",
               files="변환할 파일 (PPTX · PDF · PNG/JPG, 여러 개 가능)", mode="처리 방식", lang="슬라이드 언어", model="Claude 모델",
               use_claude="Claude 사용 (오타 교정·문단 묶기·개체 인식)", key="API 키 (비워 두면 환경 변수 ANTHROPIC_API_KEY)",
               fonts="글꼴 파일 (.ttf/.otf, 보통+굵게 두 개 권장, 선택)", slides="슬라이드 범위 (예: 1-3,7 · 비우면 전체)",
               adv="고급 설정", hidden="원본 이미지를 숨김 층으로 보관", groups="그림·텍스트를 그룹으로 묶기", theme="테마 글꼴 사용 (마스터에서 한 번에 바꾸기)",
               wm="워터마크 제거", wrap="본문 자동 줄바꿈", estimate="예상 시간·비용 보기", run="변환 시작", status="진행 상황",
               gallery="원본(왼쪽) · 결과 미리보기(오른쪽)", table="슬라이드별 결과", download="다운로드",
               rebuild_tab="재조립 (글꼴만 바꾸기)", convert_tab="변환", manifest="manifest.json (변환 때 함께 저장됨)",
               rebuild="재조립", fix_tab="텍스트 수정", fix_help="표에서 인식된 텍스트를 고친 뒤 저장하면, 재조립 때 반영됩니다.",
               save_fix="수정 저장", no_files="파일을 올려 주세요.", est_fmt="{n}장 처리 예정 (PDF 텍스트 {t}장 · 유지 {p}장) · 예상 {m}분 · 예상 비용 약 ${usd:.2f} ({krw:,.0f}원)"),
    "en": dict(title="🧩 Image slides → editable PPTX", sub="Turn image-only slides from NotebookLM, Gemini, Canva or Gamma into PPTX with real text boxes and movable pictures.",
               files="Files to convert (PPTX · PDF · PNG/JPG)", mode="Mode", lang="Slide language", model="Claude model",
               use_claude="Use Claude (text correction, grouping, object detection)", key="API key (blank = ANTHROPIC_API_KEY env)",
               fonts="Font files (.ttf/.otf, regular + bold recommended, optional)", slides="Slide range (e.g. 1-3,7; blank = all)",
               adv="Advanced", hidden="Keep original as hidden layer", groups="Group pictures / text", theme="Use theme fonts",
               wm="Remove watermarks", wrap="Word-wrap body text", estimate="Estimate time & cost", run="Convert", status="Progress",
               gallery="Original (left) · preview (right)", table="Per-slide results", download="Downloads",
               rebuild_tab="Rebuild (fonts only)", convert_tab="Convert", manifest="manifest.json (saved with each conversion)",
               rebuild="Rebuild", fix_tab="Fix text", fix_help="Edit recognised text in the table, save, then rebuild.",
               save_fix="Save fixes", no_files="Please upload files.", est_fmt="{n} slides to process (PDF text {t} · kept {p}) · ~{m} min · ~${usd:.2f}"),
}


def _cfg(mode_label, lang_label, model_label, use_claude, key, font_files, slides, hidden, groups, theme, wm, wrap, ui_lang) -> Config:
    mode = next((k for k, v in PRESETS.items() if v[f"label_{ui_lang}"] == mode_label), "full")
    lang = next((k for k, v in LANGUAGE_PROFILES.items() if v["label"] == lang_label), "ko")
    model = next((k for k, v in MODELS.items() if v["label"] == model_label), "claude-opus-5-5")
    reg, bold = assign_font_files([f.name if hasattr(f, "name") else f for f in (font_files or [])])
    return Config(mode=mode, language=lang, model=model, use_claude=bool(use_claude), api_key=(key or "").strip() or None,
                  font_regular_file=reg, font_bold_file=bold, slide_range=(slides or "").strip() or None,
                  keep_hidden_original=hidden, group_shapes=groups, use_theme_fonts=theme, remove_watermark=wm,
                  body_word_wrap=wrap, ui_lang=ui_lang, out_dir=tempfile.mkdtemp(prefix="unflatten_"))


def build_demo(ui_lang: str = "ko"):
    import gradio as gr
    from .pipeline import convert, estimate, rebuild, bundle_zip
    from .cache import SlideCache
    S = UI[ui_lang]

    with gr.Blocks(title=S["title"]) as demo:
        gr.Markdown(f"# {S['title']}\n{S['sub']}")
        with gr.Tabs():
            with gr.Tab(S["convert_tab"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        files = gr.File(label=S["files"], file_count="multiple", file_types=[".pptx", ".pdf", ".png", ".jpg", ".jpeg", ".webp"])
                        mode = gr.Radio([v[f"label_{ui_lang}"] for v in PRESETS.values()], value=PRESETS["full"][f"label_{ui_lang}"], label=S["mode"])
                        lang = gr.Dropdown([v["label"] for v in LANGUAGE_PROFILES.values()], value=LANGUAGE_PROFILES["ko"]["label"], label=S["lang"])
                        use_claude = gr.Checkbox(True, label=S["use_claude"])
                        model = gr.Dropdown([v["label"] for v in MODELS.values()], value=MODELS["claude-opus-5-5"]["label"], label=S["model"])
                        key = gr.Textbox(label=S["key"], type="password")
                        fonts = gr.File(label=S["fonts"], file_count="multiple", file_types=[".ttf", ".otf"])
                        slides = gr.Textbox(label=S["slides"])
                        with gr.Accordion(S["adv"], open=False):
                            hidden = gr.Checkbox(True, label=S["hidden"])
                            groups = gr.Checkbox(True, label=S["groups"])
                            theme = gr.Checkbox(True, label=S["theme"])
                            wm = gr.Checkbox(True, label=S["wm"])
                            wrap = gr.Checkbox(True, label=S["wrap"])
                        with gr.Row():
                            est_btn = gr.Button(S["estimate"])
                            run_btn = gr.Button(S["run"], variant="primary")
                    with gr.Column(scale=2):
                        status = gr.Textbox(label=S["status"], lines=6)
                        gallery = gr.Gallery(label=S["gallery"], columns=1, height=420)
                        table = gr.Dataframe(label=S["table"], interactive=False, wrap=True)
                        downloads = gr.File(label=S["download"], file_count="multiple")

                inputs = [files, mode, lang, model, use_claude, key, fonts, slides, hidden, groups, theme, wm, wrap]

                def do_estimate(files, *opts):
                    if not files:
                        return S["no_files"]
                    cfg = _cfg(*opts, ui_lang)
                    lines = []
                    for f in files:
                        info = estimate(f.name, cfg)
                        lines.append(f"{os.path.basename(f.name)}: " + S["est_fmt"].format(
                            n=info["n_flat"] + info["n_text_layer"], t=info["n_text_layer"], p=info["n_pass"],
                            m=info["minutes"], usd=info["usd"], krw=info["krw"]))
                    return "\n".join(lines)

                def do_convert(files, *opts, progress=gr.Progress()):
                    if not files:
                        return S["no_files"], [], None, None
                    cfg = _cfg(*opts, ui_lang)
                    logs: List[str] = []
                    outs, imgs, rows = [], [], []
                    for f in files:
                        name = os.path.basename(f.name)

                        def prog(done, total, msg):
                            progress((done, max(total, 1)), desc=f"{name} {msg}")

                        res = convert(f.name, cfg, progress=prog, log=lambda m: logs.append(str(m)))
                        outs.append(res.out_path)
                        if res.report_path:
                            outs.append(res.report_path)
                        outs.append(res.manifest_path)
                        if res.font_files:
                            outs.append(bundle_zip(res))
                        imgs += [(pv, f"{name} #{s.index}") for s, pv in zip(res.summaries, res.previews) if pv is not None]
                        rows += [[name, s.index, s.status, s.n_lines, s.n_blocks, s.n_pieces, round(s.seconds, 1), "; ".join(s.warnings)]
                                 for s in res.summaries]
                        logs.append(res.cost_line(ui_lang))
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=["file", "#", "status", "lines", "blocks", "pieces", "sec", "warnings"])
                    return "\n".join(logs[-40:]), imgs, df, outs

                est_btn.click(do_estimate, inputs, status)
                run_btn.click(do_convert, inputs, [status, gallery, table, downloads])

            with gr.Tab(S["fix_tab"]):
                gr.Markdown(S["fix_help"])
                man_fix = gr.File(label=S["manifest"], file_count="single")
                fix_table = gr.Dataframe(headers=["slide", "block", "text"], interactive=True, wrap=True)
                load_btn = gr.Button("Load")
                save_btn = gr.Button(S["save_fix"], variant="primary")
                fix_status = gr.Textbox(label=S["status"])

                def load_texts(mf):
                    if not mf:
                        return None
                    man = json.load(open(mf.name, encoding="utf-8"))
                    rows = []
                    for s in man["slides"]:
                        for bi, txt in enumerate(s.get("texts") or []):
                            rows.append([s["index"], bi, txt])
                    import pandas as pd
                    return pd.DataFrame(rows, columns=["slide", "block", "text"])

                def save_texts(mf, df):
                    if not mf or df is None:
                        return S["no_files"]
                    man = json.load(open(mf.name, encoding="utf-8"))
                    cache = SlideCache(man["cache_dir"])
                    by_slide = {}
                    for _, r in df.iterrows():
                        by_slide.setdefault(int(r["slide"]), {})[int(r["block"])] = str(r["text"])
                    n = 0
                    for s in man["slides"]:
                        if s["index"] in by_slide and s.get("cache_key") and cache.has(s["cache_key"]):
                            analysis, _, _ = cache.load(s["cache_key"])
                            for bi, txt in by_slide[s["index"]].items():
                                if bi < len(analysis.blocks):
                                    analysis.blocks[bi].text = txt.replace(" / ", "\n")
                                    n += 1
                            cache.update_blocks(s["cache_key"], analysis.blocks)
                            s["texts"] = [b.text.replace("\n", " / ") for b in analysis.blocks]
                    json.dump(man, open(mf.name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                    return f"saved {n} blocks → rebuild to apply"

                load_btn.click(load_texts, man_fix, fix_table)
                save_btn.click(save_texts, [man_fix, fix_table], fix_status)

            with gr.Tab(S["rebuild_tab"]):
                man = gr.File(label=S["manifest"], file_count="single")
                fonts2 = gr.File(label=S["fonts"], file_count="multiple", file_types=[".ttf", ".otf"])
                groups2 = gr.Checkbox(True, label=S["groups"]); theme2 = gr.Checkbox(True, label=S["theme"])
                hidden2 = gr.Checkbox(True, label=S["hidden"])
                rb_btn = gr.Button(S["rebuild"], variant="primary")
                rb_status = gr.Textbox(label=S["status"], lines=4)
                rb_out = gr.File(label=S["download"], file_count="multiple")

                def do_rebuild(mf, ff, g, th, hd, progress=gr.Progress()):
                    if not mf:
                        return S["no_files"], None
                    reg, bold = assign_font_files([f.name for f in (ff or [])])
                    ov = {"group_shapes": g, "use_theme_fonts": th, "keep_hidden_original": hd}
                    if reg:
                        ov["font_regular_file"], ov["font_bold_file"] = reg, bold
                    logs = []
                    res = rebuild(mf.name, ov, progress=lambda d, t, m: progress((d, max(t, 1)), desc=m), log=lambda m: logs.append(str(m)))
                    outs = [res.out_path] + ([res.report_path] if res.report_path else [])
                    return "\n".join(logs[-20:]), outs

                rb_btn.click(do_rebuild, [man, fonts2, groups2, theme2, hidden2], [rb_status, rb_out])
    return demo


def launch(share: bool = False, port: int = 7860, ui_lang: str = "ko", inline: bool = False):
    demo = build_demo(ui_lang)
    import gradio as gr
    kw = dict(share=share, server_port=port, inline=inline, show_error=True)
    try:
        kw["theme"] = gr.themes.Soft()          # Gradio ≥ 6 takes theme on launch()
        demo.queue().launch(**kw)
    except TypeError:
        kw.pop("theme", None)
        demo.queue().launch(**kw)
