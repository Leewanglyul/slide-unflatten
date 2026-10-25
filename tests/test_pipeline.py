import os
import zipfile

from lxml import etree
from pptx import Presentation

from unflatten.builder import iter_shapes
from unflatten.config import Config
from unflatten.inputs import load_deck
from unflatten.pipeline import convert, estimate, rebuild, bundle_zip
from tests.conftest import FakeOCR, FONT, FONT_B, FONT_MONO

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def _xml(slide):
    return etree.tostring(slide._element, encoding="unicode")


def test_pptx_detection(pptx_mixed):
    path, _ = pptx_mixed
    deck = load_deck(path)
    kinds = [s.kind for s in deck.slides]
    assert kinds == ["flat", "passthrough", "passthrough"]
    assert deck.slides[0].notes == "speaker notes 1"
    assert deck.slides[0].rect_emu[2] == deck.width_emu


def test_pdf_text_layer_detection(pdf_text):
    deck = load_deck(pdf_text)
    assert [s.kind for s in deck.slides] == ["text_layer", "flat"]
    s = deck.slides[0]
    assert any("Deck Title" in l.text for l in s.lines)
    assert s.clean_image is not None
    assert len(s.embedded_images) == 1
    title = next(l for l in s.lines if "Deck Title" in l.text)
    assert title.bold and title.font_px > 60   # 36pt at 150dpi ≈ 75px


def test_convert_flat_png_no_claude(flat_png, tmp_path):
    path, lines = flat_png
    cfg = Config(mode="full", language="en", use_claude=False, out_dir=str(tmp_path / "out"),
                 font_regular_file=FONT, font_bold_file=FONT_B)
    ocr = FakeOCR(lines)
    res = convert(path, cfg, ocr_override=ocr, log=lambda *a: None)
    assert os.path.exists(res.out_path) and os.path.exists(res.report_path) and os.path.exists(res.manifest_path)
    s = res.summaries[0]
    assert s.status == "ok" and s.n_blocks == 3 and s.n_pieces >= 2
    prs = Presentation(res.out_path)
    slide = prs.slides[0]
    xml = _xml(slide)
    texts = [sh.text_frame.text for sh in iter_shapes(slide.shapes) if sh.has_text_frame]
    assert all(pv is None or os.path.exists(pv) for pv in res.previews) and res.previews[0]
    assert "Polar Frontiers Report" in texts
    assert "Ice is melting faster than expected" in texts           # bullet glyph stripped
    assert 'buChar char="•"' in xml                                 # real bullets
    assert 'hidden="1"' in xml                                      # hidden original
    assert "+mn-lt" in xml or "+mj-lt" in xml                       # theme font references
    assert "<p:grpSp>" in xml                                       # grouping
    assert 'descr="' in xml                                         # alt text
    # fonts bundled next to output
    assert len(res.font_files) == 2
    z = bundle_zip(res)
    assert set(n.split("/")[0] for n in zipfile.ZipFile(z).namelist()) >= {"fonts", os.path.basename(res.out_path)}


def test_cache_and_rebuild_skip_recognition(flat_png, tmp_path):
    path, lines = flat_png
    cfg = Config(mode="objects", language="en", use_claude=False, out_dir=str(tmp_path / "out"))
    ocr = FakeOCR(lines)
    res = convert(path, cfg, ocr_override=ocr, log=lambda *a: None)
    assert ocr.calls == 1
    # second run hits the cache: FakeOCR not consulted again, output rebuilt with other assembly options
    res2 = rebuild(res.manifest_path, {"group_shapes": False, "keep_hidden_original": False, "font_regular": "Arial",
                                       "use_theme_fonts": False}, log=lambda *a: None)
    prs = Presentation(res2.out_path)
    xml = _xml(prs.slides[0])
    assert 'hidden="1"' not in xml and "<p:grpSp>" not in xml and 'typeface="Arial"' in xml
    assert ocr.calls == 1


def test_convert_pptx_keeps_other_slides(pptx_mixed, tmp_path):
    path, lines = pptx_mixed
    cfg = Config(mode="text", language="en", use_claude=False, out_dir=str(tmp_path / "out"))
    res = convert(path, cfg, ocr_override=FakeOCR(lines), log=lambda *a: None)
    statuses = {s.index: s.status for s in res.summaries}
    assert statuses == {1: "ok", 2: "passthrough", 3: "passthrough"}
    prs = Presentation(res.out_path)
    assert len(prs.slides) == 3
    assert prs.slides[0].notes_slide.notes_text_frame.text == "speaker notes 1"
    assert prs.slides[1].shapes.title.text == "Real editable title"
    assert any(sh.has_text_frame and "only text" in sh.text_frame.text for sh in iter_shapes(prs.slides[2].shapes))
    # flattened picture removed, text boxes present, no re-fonting of the author's theme
    s1_xml = _xml(prs.slides[0])
    assert "Polar Frontiers Report" in s1_xml and "+mn-lt" not in s1_xml
    pics = [sh for sh in iter_shapes(prs.slides[0].shapes) if sh.shape_type == 13]
    assert any('hidden="1"' in etree.tostring(p_._element, encoding="unicode") for p_ in pics)
    assert sum(1 for p_ in pics if p_.image.content_type == "image/png") == 1   # the original PNG survived
    assert res.summaries[0].n_pieces == 0   # text mode: background kept whole


def test_convert_pdf_text_layer_without_ocr(pdf_text, tmp_path):
    cfg = Config(mode="objects", language="en", use_claude=False, ocr_backend="none", out_dir=str(tmp_path / "out"))
    res = convert(pdf_text, cfg, log=lambda *a: None)
    assert [s.status for s in res.summaries] == ["text_layer", "ok"]
    prs = Presentation(res.out_path)
    texts = [sh.text_frame.text for sh in iter_shapes(prs.slides[0].shapes) if sh.has_text_frame]
    assert "Deck Title From PDF" in texts
    assert any("First body line" in t and "Second body line" in t for t in texts)   # paragraph kept together
    assert res.summaries[0].n_pieces == 1   # embedded image became a piece
    assert res.summaries[1].n_lines == 0    # raster page: no OCR engine → background only, no crash
    info = estimate(pdf_text, cfg)
    assert info["n_text_layer"] == 1 and info["n_flat"] == 1 and info["usd"] == 0


def test_slide_range_and_failure_isolation(pdf_text, tmp_path, monkeypatch):
    import unflatten.pipeline as P
    cfg = Config(mode="text", language="en", use_claude=False, ocr_backend="none", slide_range="1", out_dir=str(tmp_path / "out"))

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(P, "_erase_and_split", boom)
    res = convert(pdf_text, cfg, log=lambda *a: None)
    assert res.summaries[0].status == "failed" and res.summaries[1].status == "skipped"
    prs = Presentation(res.out_path)
    assert len(prs.slides) == 2            # deck stays complete: failed + skipped slides keep the original image


def test_pdf_background_photo_and_transparent_icon(tmp_path):
    """Extracting an icon must not wipe a full-page background photo; icon keeps its alpha; gray images work."""
    import io
    import numpy as np
    import pymupdf
    from PIL import Image, ImageDraw
    doc = pymupdf.open()
    page = doc.new_page(width=960, height=540)
    b = io.BytesIO(); Image.new("RGB", (960, 540), (217, 120, 60)).save(b, "PNG")
    page.insert_image(page.rect, stream=b.getvalue())
    icon = Image.new("RGBA", (120, 120), (0, 0, 0, 0)); ImageDraw.Draw(icon).ellipse((10, 10, 110, 110), fill=(20, 200, 80, 255))
    ib = io.BytesIO(); icon.save(ib, "PNG")
    page.insert_image(pymupdf.Rect(700, 100, 820, 220), stream=ib.getvalue())
    gb = io.BytesIO(); Image.new("L", (100, 100), 128).save(gb, "PNG")
    page.insert_image(pymupdf.Rect(100, 350, 200, 450), stream=gb.getvalue())
    page.insert_text((50, 80), "Title over a photo background", fontsize=36, fontname="hebo", color=(1, 1, 1))
    p = tmp_path / "bg.pdf"; doc.save(p)
    deck = load_deck(str(p))
    s = deck.slides[0]
    assert s.kind == "text_layer" and len(s.embedded_images) == 2
    assert tuple(np.array(s.clean_image)[270, 480]) == (217, 120, 60)       # background photo survived
    modes = {e["image"].mode for e in s.embedded_images}
    assert "RGBA" in modes
    icon_img = next(e["image"] for e in s.embedded_images if e["image"].mode == "RGBA")
    assert np.array(icon_img)[0, 0, 3] == 0                                   # transparent corner kept
    s.release(); assert s._data is None


def test_rebuild_refits_font_sizes(flat_png, tmp_path):
    """Rebuilding with a wider font must re-fit sizes so text stays inside its box."""
    import re
    path, lines = flat_png
    mono = FONT_MONO
    cfg = Config(mode="text", language="en", use_claude=False, out_dir=str(tmp_path / "out"), font_regular_file=FONT, font_bold_file=FONT_B)
    res = convert(path, cfg, ocr_override=FakeOCR(lines), log=lambda *a: None)
    sizes1 = [int(v) for v in re.findall(r'<a:rPr[^>]*\bsz="(\d+)"', _xml(Presentation(res.out_path).slides[0]))]
    res2 = rebuild(res.manifest_path, {"font_regular_file": mono, "font_bold_file": mono}, log=lambda *a: None)
    sizes2 = [int(v) for v in re.findall(r'<a:rPr[^>]*\bsz="(\d+)"', _xml(Presentation(res2.out_path).slides[0]))]
    assert sizes1 and sizes2 and sizes2 != sizes1
    assert all(b <= a for a, b in zip(sizes1, sizes2))                         # monospace is wider → smaller sizes


def test_duplicate_slides_recognised_once(tmp_path, flat_png):
    """Two identical pages are OCR'd once and both end up in the deck."""
    import pymupdf
    path, lines = flat_png
    doc = pymupdf.open()
    for _ in range(2):
        pg = doc.new_page(width=960, height=540); pg.insert_image(pg.rect, filename=path)
    pdf = tmp_path / "dup.pdf"; doc.save(pdf)
    cfg = Config(mode="text", language="en", use_claude=False, out_dir=str(tmp_path / "out"))
    ocr = FakeOCR(lines)
    res = convert(str(pdf), cfg, ocr_override=ocr, log=lambda *a: None)
    assert ocr.calls == 1
    assert [s.status for s in res.summaries] == ["ok", "ok"]
    assert len(Presentation(res.out_path).slides) == 2


def test_cache_key_reflects_backends(flat_png):
    from unflatten.cache import SlideCache
    from PIL import Image
    img = Image.open(flat_png[0])
    cfg = Config()
    k_none = SlideCache.key(img, cfg, "easyocr", False)
    k_claude = SlideCache.key(img, cfg, "easyocr", True, "claude-opus-5-5")
    k_other_model = SlideCache.key(img, cfg, "easyocr", True, "claude-sonnet-5-5")
    assert len({k_none, k_claude, k_other_model}) == 3
    assert SlideCache.key(img, cfg, "easyocr", False, "claude-opus-5-5") == k_none   # model irrelevant without Claude
