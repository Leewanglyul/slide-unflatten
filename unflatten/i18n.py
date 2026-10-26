"""Tiny ko/en string table for logs and UI."""
from __future__ import annotations

_STRINGS = {
    "ko": {
        "ocr": "글자 인식",
        "text_layer": "PDF 텍스트 레이어 사용 (OCR 생략)",
        "passthrough": "이미 편집 가능한 슬라이드 → 그대로 유지",
        "claude": "Claude 분석",
        "claude_fail": "Claude 호출 실패 → 규칙 기반으로 진행: {err}",
        "claude_auth": "API 키 문제로 남은 슬라이드도 규칙 기반으로 진행합니다.",
        "erase": "글자 지우기 ({n}줄)",
        "watermark": "워터마크 제거 ({n}곳)",
        "split_sam": "개체 분리(SAM 2): {names}",
        "split_simple": "개체 분리(단순)",
        "split_skip_photo": "사진 배경 → 개체 분리 생략",
        "sam_fail": "SAM 2 실패 → 단순 분리: {err}",
        "lama_fail": "LaMa 사용 불가 → OpenCV 인페인팅: {err}",
        "slide_fail": "슬라이드 {i} 실패 → 원본 이미지로 대체: {err}",
        "cached": "캐시 사용",
        "resplit": "가장자리 설정이 바뀌어 그림 분리만 다시 함 (인식 결과 재사용, API 비용 없음)",
        "done": "완료: {path}",
        "report": "리포트: {path}",
        "no_ocr": "OCR 엔진이 없어 Claude로 글자 위치를 찾습니다 (정확도 낮음).",
        "no_ocr_none": "OCR 엔진과 Claude가 모두 없어 글자를 인식할 수 없습니다. 배경만 넣습니다.",
        "estimate": "{n}장 · 예상 {min}분 · 예상 비용 약 ${usd:.2f} (약 {krw:,.0f}원)",
        "cost": "Claude 사용량: 입력 {inp:,} / 출력 {out:,} 토큰 ≈ ${usd:.3f}",
        "status_ok": "완료", "status_fail": "실패", "status_pass": "유지", "status_text": "PDF 텍스트",
        "warn_lowconf": "신뢰도 낮은 줄 {n}개",
        "warn_noclaude": "Claude 없이 처리",
    },
    "en": {
        "ocr": "OCR",
        "text_layer": "PDF text layer used (OCR skipped)",
        "passthrough": "Already editable → kept as is",
        "claude": "Claude analysis",
        "claude_fail": "Claude call failed → heuristic grouping: {err}",
        "claude_auth": "API key problem; remaining slides use heuristics.",
        "erase": "Erasing text ({n} lines)",
        "watermark": "Removing watermark ({n})",
        "split_sam": "Splitting objects (SAM 2): {names}",
        "split_simple": "Splitting objects (simple)",
        "split_skip_photo": "Photo background → object split skipped",
        "sam_fail": "SAM 2 failed → simple split: {err}",
        "lama_fail": "LaMa unavailable → OpenCV inpainting: {err}",
        "slide_fail": "Slide {i} failed → original image kept: {err}",
        "cached": "from cache",
        "resplit": "edge settings changed: redoing the split only (recognition reused, no API cost)",
        "done": "Done: {path}",
        "report": "Report: {path}",
        "no_ocr": "No OCR engine; asking Claude for text positions (less precise).",
        "no_ocr_none": "Neither OCR nor Claude available; background only.",
        "estimate": "{n} slides · ~{min} min · ~${usd:.2f}",
        "cost": "Claude usage: {inp:,} in / {out:,} out tokens ≈ ${usd:.3f}",
        "status_ok": "ok", "status_fail": "failed", "status_pass": "kept", "status_text": "PDF text",
        "warn_lowconf": "{n} low-confidence lines",
        "warn_noclaude": "processed without Claude",
    },
}


def t(lang: str, key: str, **kw) -> str:
    table = _STRINGS.get(lang, _STRINGS["en"])
    s = table.get(key) or _STRINGS["en"].get(key) or key
    try:
        return s.format(**kw)
    except (KeyError, IndexError):
        return s
