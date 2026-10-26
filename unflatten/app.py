"""One-screen Gradio UI: upload → options → estimate → convert → preview → download. Also rebuild + text fixes."""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from typing import List

CACHE_DIR = os.environ.get("UNFLATTEN_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "unflatten"))
_SESSION_DIR = tempfile.mkdtemp(prefix="unflatten_ui_")        # one per process, removed on exit
atexit.register(lambda: shutil.rmtree(_SESSION_DIR, ignore_errors=True))


def _path(f) -> str:
    """Gradio hands back str paths or objects with .name depending on version."""
    return f if isinstance(f, str) else getattr(f, "name", str(f))


def _job_dir() -> str:
    return tempfile.mkdtemp(prefix="job_", dir=_SESSION_DIR)

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
               rebuild_tab="재조립 (글꼴·가장자리 바꾸기)", edge_expand="그림 가장자리 넓히기 (px)", edge_feather="그림 가장자리 부드럽게 (px)", convert_tab="변환", manifest="manifest.json (변환 때 함께 저장됨)",
               rebuild="재조립", fix_tab="텍스트 수정", fix_help="표에서 인식된 텍스트를 고친 뒤 저장하면, 재조립 때 반영됩니다.",
               save_fix="수정 저장", no_files="파일을 올려 주세요.", est_fmt="{n}장 처리 예정 (PDF 텍스트 {t}장 · 유지 {p}장) · 예상 {m}분 · 예상 비용 약 ${usd:.2f} ({krw:,.0f}원)"),
    "en": dict(title="🧩 Image slides → editable PPTX", sub="Turn image-only slides from NotebookLM, Gemini, Canva or Gamma into PPTX with real text boxes and movable pictures.",
               files="Files to convert (PPTX · PDF · PNG/JPG)", mode="Mode", lang="Slide language", model="Claude model",
               use_claude="Use Claude (text correction, grouping, object detection)", key="API key (blank = ANTHROPIC_API_KEY env)",
               fonts="Font files (.ttf/.otf, regular + bold recommended, optional)", slides="Slide range (e.g. 1-3,7; blank = all)",
               adv="Advanced", hidden="Keep original as hidden layer", groups="Group pictures / text", theme="Use theme fonts",
               wm="Remove watermarks", wrap="Word-wrap body text", estimate="Estimate time & cost", run="Convert", status="Progress",
               gallery="Original (left) · preview (right)", table="Per-slide results", download="Downloads",
               rebuild_tab="Rebuild (fonts · edges)", edge_expand="Picture edge expand (px)", edge_feather="Picture edge feather (px)", convert_tab="Convert", manifest="manifest.json (saved with each conversion)",
               rebuild="Rebuild", fix_tab="Fix text", fix_help="Edit recognised text in the table, save, then rebuild.",
               save_fix="Save fixes", no_files="Please upload files.", est_fmt="{n} slides to process (PDF text {t} · kept {p}) · ~{m} min · ~${usd:.2f}"),
}

INFO = {
    "ko": dict(
        mode="글자만: 글자만 편집 가능하게, 그림은 배경에 그대로(가장 빠름·저렴) · 글자+그림: 그림을 덩어리 단위로 떼어 냄 · "
             "세부 개체: 그림 속 사람·물건까지 따로 떼어 내고, 겹쳐 가려진 부분은 AI가 채움",
        lang="슬라이드에 쓰인 언어. 글자 인식과 기본 글꼴을 정합니다.",
        model="Opus: 가장 정확(장당 약 50~80원) · Sonnet: 비용 절반. Claude 사용을 켰을 때만 씁니다.",
        use_claude="끄면 비용 없이 규칙 기반으로만 처리합니다. 오타가 남고, 꾸민 글씨·기울어진 글씨는 이미지로 남을 수 있습니다.",
        fonts="보통 굵기 + 굵은 것 두 개를 함께 올리면 본문·제목에 자동 배정되고, 글자 크기도 이 글꼴 폭에 맞춥니다.",
        slides="지정하지 않은 장은 원본 그대로 남습니다.",
        hidden="PowerPoint 선택 창에서 켜면 원본과 겹쳐 비교할 수 있습니다. 끄면 파일 용량이 줄어듭니다.",
        groups="그림끼리·글자끼리 묶어 선택 창을 정리합니다. 그룹 안 개체는 한 번 더 클릭하면 하나씩 선택됩니다.",
        theme="디자인 → 변형 → 글꼴에서 한 번에 바꿀 수 있게 테마 글꼴로 지정합니다.",
        wm="슬라이드 구석의 생성 도구 표시(로고 글씨)를 지웁니다. Claude가 있을 때만 작동합니다.",
        wrap="본문 상자 폭에 맞춰 자동 줄바꿈합니다. 끄면 원본 줄바꿈을 그대로 지킵니다.",
        edge_expand="외곽선·흰 스티커 테두리를 몇 px까지 그림 조각에 포함할지. 테두리가 잘리면 ↑, 배경색 띠가 붙으면 ↓, 0이면 끔.",
        edge_feather="그림 조각 테두리를 몇 px 부드럽게 할지. 계단처럼 거칠면 ↑, 흐릿하면 ↓, 0이면 딱딱한 경계.",
    ),
    "en": dict(
        mode="Text only: editable text, pictures stay in the background (fastest, cheapest) · Text + pictures: pictures as whole "
             "pieces · Detailed objects: people and things inside a picture separately, hidden parts filled in by AI",
        lang="Language written on the slides; sets text recognition and default fonts.",
        model="Opus: most accurate · Sonnet: half the cost. Used only when Claude is on.",
        use_claude="Off = rule-based only, no API cost. Typos stay; decorated or tilted text may stay in the image.",
        fonts="Upload a regular and a bold file together: they go to body and titles, and sizes fit that font's width.",
        slides="Slides left out stay as they are.",
        hidden="Turn it on in PowerPoint's Selection Pane to compare with the original. Off = smaller file.",
        groups="Groups pictures and text boxes to keep the Selection Pane tidy. Click again inside a group to pick one item.",
        theme="Uses theme fonts so Design → Variants → Fonts changes them all at once.",
        wm="Removes generator badges in slide corners. Needs Claude.",
        wrap="Wrap body text to the box width. Off keeps the original line breaks.",
        edge_expand="How many px a picture cut-out may grow to take in its outline / white sticker rim. Rim cut off → raise; "
                    "a band of background colour → lower; 0 = off.",
        edge_feather="How many px of soft edge on picture cut-outs. Jagged → raise; blurry → lower; 0 = hard edge.",
    ),
}


def _cfg(mode_label, lang_label, model_label, use_claude, key, font_files, slides, hidden, groups, theme, wm, wrap,
         edge_expand, edge_feather, ui_lang) -> Config:
    mode = next((k for k, v in PRESETS.items() if v[f"label_{ui_lang}"] == mode_label), "full")
    lang = next((k for k, v in LANGUAGE_PROFILES.items() if v["label"] == lang_label), "ko")
    model = next((k for k, v in MODELS.items() if v["label"] == model_label), "claude-opus-5-5")
    reg, bold = assign_font_files([_path(f) for f in (font_files or [])])
    return Config(mode=mode, language=lang, model=model, use_claude=bool(use_claude), api_key=(key or "").strip() or None,
                  font_regular_file=reg, font_bold_file=bold, slide_range=(slides or "").strip() or None,
                  keep_hidden_original=hidden, group_shapes=groups, use_theme_fonts=theme, remove_watermark=wm,
                  body_word_wrap=wrap, edge_expand=int(edge_expand), edge_feather=float(edge_feather),
                  ui_lang=ui_lang, out_dir=_job_dir(), cache_dir=CACHE_DIR)


def build_demo(ui_lang: str = "ko"):
    import gradio as gr
    from .pipeline import convert, estimate, rebuild, bundle_zip
    from .cache import SlideCache
    S = UI[ui_lang]
    I = INFO[ui_lang]

    with gr.Blocks(title=S["title"]) as demo:
        gr.Markdown(f"# {S['title']}\n{S['sub']}")
        with gr.Tabs():
            with gr.Tab(S["convert_tab"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        files = gr.File(label=S["files"], file_count="multiple", file_types=[".pptx", ".pdf", ".png", ".jpg", ".jpeg", ".webp"])
                        mode = gr.Radio([v[f"label_{ui_lang}"] for v in PRESETS.values()], value=PRESETS["full"][f"label_{ui_lang}"], label=S["mode"], info=I["mode"])
                        lang = gr.Dropdown([v["label"] for v in LANGUAGE_PROFILES.values()], value=LANGUAGE_PROFILES["ko"]["label"], label=S["lang"], info=I["lang"])
                        use_claude = gr.Checkbox(True, label=S["use_claude"], info=I["use_claude"])
                        model = gr.Dropdown([v["label"] for v in MODELS.values()], value=MODELS["claude-opus-5-5"]["label"], label=S["model"], info=I["model"])
                        key = gr.Textbox(label=S["key"], type="password")
                        fonts = gr.File(label=S["fonts"], file_count="multiple", file_types=[".ttf", ".otf"])
                        gr.Markdown(I["fonts"])
                        slides = gr.Textbox(label=S["slides"], info=I["slides"])
                        with gr.Accordion(S["adv"], open=False):
                            hidden = gr.Checkbox(True, label=S["hidden"], info=I["hidden"])
                            groups = gr.Checkbox(True, label=S["groups"], info=I["groups"])
                            theme = gr.Checkbox(True, label=S["theme"], info=I["theme"])
                            wm = gr.Checkbox(True, label=S["wm"], info=I["wm"])
                            wrap = gr.Checkbox(True, label=S["wrap"], info=I["wrap"])
                            edge_expand = gr.Slider(0, 12, value=4, step=1, label=S["edge_expand"], info=I["edge_expand"])
                            edge_feather = gr.Slider(0, 3, value=1.0, step=0.5, label=S["edge_feather"], info=I["edge_feather"])
                        with gr.Row():
                            est_btn = gr.Button(S["estimate"])
                            run_btn = gr.Button(S["run"], variant="primary")
                    with gr.Column(scale=2):
                        status = gr.Textbox(label=S["status"], lines=6)
                        gallery = gr.Gallery(label=S["gallery"], columns=1, height=420)
                        table = gr.Dataframe(label=S["table"], interactive=False, wrap=True)
                        downloads = gr.File(label=S["download"], file_count="multiple")

                inputs = [files, mode, lang, model, use_claude, key, fonts, slides, hidden, groups, theme, wm, wrap, edge_expand, edge_feather]

                def do_estimate(files, *opts):
                    if not files:
                        return S["no_files"]
                    cfg = _cfg(*opts, ui_lang)
                    lines = []
                    for f in files:
                        info = estimate(_path(f), cfg)
                        lines.append(f"{os.path.basename(_path(f))}: " + S["est_fmt"].format(
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
                        name = os.path.basename(_path(f))

                        def prog(done, total, msg):
                            progress((done, max(total, 1)), desc=f"{name} {msg}")

                        res = convert(_path(f), cfg, progress=prog, log=lambda m: logs.append(str(m)))
                        outs.append(res.out_path)
                        if res.report_path:
                            outs.append(res.report_path)
                        outs.append(res.manifest_path)
                        if res.font_files:
                            outs.append(bundle_zip(res))
                        imgs += [(pv, f"{name} #{s.index}") for s, pv in zip(res.summaries, res.previews) if pv]
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
                    with open(_path(mf), encoding="utf-8") as fh:
                        man = json.load(fh)
                    rows = []
                    for s in man["slides"]:
                        for bi, txt in enumerate(s.get("texts") or []):
                            rows.append([s["index"], bi, txt])
                    import pandas as pd
                    return pd.DataFrame(rows, columns=["slide", "block", "text"])

                def save_texts(mf, df):
                    if not mf or df is None:
                        return S["no_files"]
                    with open(_path(mf), encoding="utf-8") as fh:
                        man = json.load(fh)
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
                                    analysis.blocks[bi].text = txt.replace(" ⏎ ", "\n")
                                    n += 1
                            cache.update_blocks(s["cache_key"], analysis.blocks)
                            s["texts"] = [b.text.replace("\n", " ⏎ ") for b in analysis.blocks]
                    # write back to the uploaded copy AND to the manifest next to the real output
                    targets = {_path(mf)}
                    real = os.path.join(os.path.dirname(man.get("output", "")), os.path.basename(_path(mf)))
                    if man.get("output") and os.path.isdir(os.path.dirname(man["output"])):
                        targets.add(real)
                    for tpath in targets:
                        with open(tpath, "w", encoding="utf-8") as fh:
                            json.dump(man, fh, ensure_ascii=False, indent=1)
                    return f"saved {n} blocks → rebuild to apply"

                load_btn.click(load_texts, man_fix, fix_table)
                save_btn.click(save_texts, [man_fix, fix_table], fix_status)

            with gr.Tab(S["rebuild_tab"]):
                man = gr.File(label=S["manifest"], file_count="single")
                fonts2 = gr.File(label=S["fonts"], file_count="multiple", file_types=[".ttf", ".otf"])
                groups2 = gr.Checkbox(True, label=S["groups"]); theme2 = gr.Checkbox(True, label=S["theme"])
                hidden2 = gr.Checkbox(True, label=S["hidden"])
                expand2 = gr.Slider(0, 12, value=4, step=1, label=S["edge_expand"], info=I["edge_expand"])
                feather2 = gr.Slider(0, 3, value=1.0, step=0.5, label=S["edge_feather"], info=I["edge_feather"])
                rb_btn = gr.Button(S["rebuild"], variant="primary")
                rb_status = gr.Textbox(label=S["status"], lines=4)
                rb_out = gr.File(label=S["download"], file_count="multiple")

                def do_rebuild(mf, ff, g, th, hd, ex, fe, progress=gr.Progress()):
                    if not mf:
                        return S["no_files"], None
                    reg, bold = assign_font_files([_path(f) for f in (ff or [])])
                    ov = {"group_shapes": g, "use_theme_fonts": th, "keep_hidden_original": hd,
                          "edge_expand": int(ex), "edge_feather": float(fe)}
                    if reg:
                        ov["font_regular_file"], ov["font_bold_file"] = reg, bold
                    logs = []
                    res = rebuild(_path(mf), ov, progress=lambda d, t, m: progress((d, max(t, 1)), desc=m), log=lambda m: logs.append(str(m)))
                    outs = [res.out_path] + ([res.report_path] if res.report_path else [])
                    return "\n".join(logs[-20:]), outs

                rb_btn.click(do_rebuild, [man, fonts2, groups2, theme2, hidden2, expand2, feather2], [rb_status, rb_out])
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
