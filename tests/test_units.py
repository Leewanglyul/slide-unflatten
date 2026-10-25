from unflatten.builder import _segments
from unflatten.config import Config, parse_slide_range, estimate_cost_usd
from unflatten.layout import _strip_bullet, group_text_layer
from unflatten.segment import unique_names
from unflatten.types import Line


def test_slide_range():
    assert parse_slide_range("1-3,7", 10) == [1, 2, 3, 7]
    assert parse_slide_range(None, 3) == [1, 2, 3]
    assert parse_slide_range("8-", 10) == [8, 9, 10]
    assert parse_slide_range("0,99", 5) == []


def test_unique_names():
    assert unique_names(["나침반", "펭귄", "나침반", ""]) == ["나침반 1", "펭귄", "나침반 2", "object"]


def test_segments_highlight():
    segs = _segments("Ice is melting fast", ["melting"])
    assert segs == [("Ice is ", False), ("melting", True), (" fast", False)]
    assert _segments("plain", []) == [("plain", False)]


def test_strip_bullet():
    assert _strip_bullet("• hello") == ("hello", True)
    assert _strip_bullet("- dash item") == ("dash item", True)
    assert _strip_bullet("no bullet") == ("no bullet", False)


def test_config_roundtrip_and_cost():
    c = Config(mode="text", language="en", api_key="sk-ant-x")
    d = c.to_dict()
    assert "api_key" not in d
    c2 = Config.from_dict(d)
    assert c2.mode == "text" and c2.language == "en" and not c2.split_objects
    assert estimate_cost_usd(10, "claude-sonnet-5-5") < estimate_cost_usd(10, "claude-opus-5-5")


def test_group_text_layer_merges_paragraph():
    lines = [Line((10, 10, 300, 40), "a", block_id=1, font_px=30),
             Line((10, 50, 280, 80), "b", block_id=1, font_px=30),
             Line((10, 200, 100, 220), "• c", block_id=2, font_px=18)]
    blocks = group_text_layer(lines, img_h=500)
    assert len(blocks) == 2
    assert blocks[0].text == "a\nb" and blocks[0].role == "title"
    assert blocks[1].text == "c" and blocks[1].bullet


def test_text_color_bold_title_on_light_background():
    """A tight ink box around heavy bold text holds more stroke than background pixels;
    the text colour must still be the stroke colour, not the background."""
    import numpy as np
    from unflatten.imageops import ink_box, text_color
    arr = np.full((120, 400, 3), 250, np.uint8)            # near-white slide
    arr[40:80, 50:350] = (40, 60, 120)                      # dark navy "bold strokes"
    arr[40:80, 120:126] = 250                               # a few thin gaps between letters
    arr[40:80, 230:236] = 250
    box = ink_box(arr, (45, 35, 355, 85))
    r, g, b = text_color(arr, box)
    assert (r, g, b) != (250, 250, 250) and r < 100 and b > g
