# slide-unflatten

**이미지로만 된 슬라이드(NotebookLM · Gemini · Canva · Gamma · 스캔 · 캡처)를 글자와 그림을 고칠 수 있는 PPTX로 바꿉니다.**

Turn image-only slides into editable PPTX: real text boxes, movable picture pieces, a clean background, real bullets, theme fonts.

```
입력  PPTX(통이미지) · PDF · PNG/JPG
 │
 ├─ 입력 판별      통이미지 슬라이드만 변환, 이미 편집 가능한 슬라이드·노트·슬라이드 크기는 그대로 유지
 ├─ PDF 텍스트층   글자가 실제 텍스트인 PDF는 OCR·API 없이 좌표·크기·색을 그대로 복원
 ├─ 글자 인식      EasyOCR 또는 PaddleOCR(위치) + Claude(오타 교정 · 문단 묶기 · 제목/본문 · 강조 · 글머리 · 워터마크 · 개체 찾기)
 ├─ 글자 지우기    LaMa 인페인팅(없으면 OpenCV)
 ├─ 그림 분리      Claude 상자 + SAM 2 윤곽, 가려진 부분 채우기 (사진 배경이면 자동 생략)
 ├─ 캐시           슬라이드별 인식 결과 저장 → 글꼴·옵션만 바꿔 몇 초 만에 재조립
 └─ 조립           PPTX + HTML 리포트(전후 비교) + manifest.json (+ 글꼴 묶음 zip)
```

## 설치

```bash
pip install -e ".[claude,ui]"              # 핵심 + Claude + 웹 UI (CPU, PDF 텍스트층·OpenCV 지우기)
pip install -e ".[all]"                    # + EasyOCR, SAM 2, LaMa (GPU 권장)
pip install --no-deps simple-lama-inpainting   # LaMa는 옛 Pillow를 요구하므로 --no-deps
pip install -e ".[paddle]"                 # 선택: PaddleOCR (Python 3.9~3.13)
export ANTHROPIC_API_KEY=sk-ant-...        # 선택. 없으면 규칙 기반으로 동작
unflatten doctor                           # 의존성 · GPU · 키 확인
```

Colab에서는 `notebooks/unflatten_colab.ipynb`를 열고 **런타임 → 모두 실행**하면 됩니다(폼으로 설정, 웹 UI 포함).

## 사용

```bash
unflatten estimate deck.pptx                         # 장수 · 예상 시간 · 예상 비용
unflatten convert deck.pptx                          # → deck_editable.pptx, deck_report.html, deck_manifest.json
unflatten convert deck.pdf --mode text --lang en     # 텍스트만, 영어
unflatten convert *.png --font NanumGothic.ttf --font-bold NanumGothicBold.ttf --zip
unflatten convert deck.pptx --slides 1-3,7 --model claude-sonnet-5-5
unflatten rebuild deck_manifest.json --font Pretendard-Regular.otf --font-bold Pretendard-Bold.otf   # 재인식 없이 재조립
unflatten ui --share                                 # Gradio 한 화면 UI (업로드 → 옵션 → 예상 → 변환 → 미리보기 → 다운로드)
```

파이썬에서:

```python
from unflatten import Config, convert
res = convert("deck.pptx", Config(mode="full", language="ko", model="claude-opus-5-5"))
print(res.out_path, res.report_path, res.cost_line())
```

### 처리 방식 (`--mode`)

| 값 | 결과 | 비용·시간 |
|---|---|---|
| `text` | 텍스트 상자 + 글자 지운 배경 한 장 (그림은 나누지 않고 배경에 그대로) | 가장 빠름, Claude 호출 1회/장 |
| `objects` | + 그림을 통째로 조각 분리 | 중간 |
| `full` (기본) | + 그림 속 개별 사물까지 분리, 가려진 부분 채움 | SAM 2 사용, GPU 권장 |

### 글자 인식 엔진 (`--ocr`)

| 값 | 특징 |
|---|---|
| `auto` (기본) | 설치된 것 중 EasyOCR → PaddleOCR → Claude 순 |
| `easyocr` | 기본 설치. 둥근 꾸밈 제목 글씨에 강한 편 |
| `paddleocr` | 본문 한글 인식이 더 정확한 경우가 많음(조사 오인식이 적음). 단어 단위 결과를 줄로 합치고, 꾸밈 제목의 외곽선이 한 번 더 읽힌 줄·기호만 있는 줄은 버림. CPU에서는 EasyOCR보다 느림 |
| `claude` | OCR 없이 Claude가 글자 위치까지 찾음. 위치 정확도가 낮아 OCR이 없을 때만 권장 |

고른 엔진이 설치되어 있지 않으면 다른 무료 엔진으로 대신하고 로그에 알립니다. Windows에서 PaddlePaddle과 PyTorch를 함께 쓰면 PyTorch를 먼저 불러와야 해서, 백엔드가 그 순서를 지킵니다.

### 결과 PPTX 구조

| 층 | 이름 | 비고 |
|---|---|---|
| 맨 위 | **텍스트** 그룹 (`텍스트 1 · 첫 단어…`) | 제목은 실제 제목 자리표시자 → 개요 보기·접근성·Google Slides 호환 |
| 중간 | **그림** 그룹 (`그림 1 · 펭귄`) | 투명 PNG, 뒤→앞 순서, 대체 텍스트에 개체 이름 |
| 아래 | `배경` | 글자(와 분리한 그림)를 지운 이미지 |
| 숨김 | `원본(숨김)` | 선택 창에서 켜서 비교 (`--no-hidden-original`로 생략) |

- 글자 런은 테마 글꼴(`+mj`/`+mn`)을 참조합니다. **슬라이드 마스터 → 글꼴**에서 한 번만 바꾸면 전체에 반영됩니다. (원본 PPTX에 이미 편집 가능한 슬라이드가 섞여 있으면 저자의 테마를 건드리지 않고 글꼴 이름을 직접 씁니다.)
- 형광펜 강조는 실제 형광펜(`a:highlight`)과 굵게, 글머리 기호는 실제 단락 글머리(`a:buChar`)로 들어갑니다.
- 슬라이드 색상 팔레트에서 뽑은 6색이 테마 강조색(accent1~6)에 들어갑니다.
- 올린 글꼴 파일은 결과 옆 `fonts/`에 복사되고 `--zip`이면 함께 묶입니다. 결과를 여는 PC에도 같은 글꼴이 있어야 같은 모양으로 보입니다.

### 검수 · 수정 · 재조립

1. `*_report.html`: 슬라이드별 상태 · 줄/블록/조각 수 · 경고 · 원본/미리보기 비교. 실패한 장은 빨간색.
2. UI의 **텍스트 수정** 탭(또는 `*_manifest.json`과 캐시의 `analysis.json`)에서 인식 텍스트를 고치고,
3. **재조립**(`unflatten rebuild`)하면 OCR·Claude·SAM을 다시 돌리지 않고 몇 초 만에 PPTX가 다시 만들어집니다.

### 비용

Claude는 슬라이드 1장당 이미지 1장 + 줄 목록을 보내고 구조화된 JSON을 받습니다. 장당 대략:

| 모델 | 장당 | 100장 |
|---|---|---|
| `claude-opus-5-5` (기본, 가장 정확) | ≈ $0.045 | ≈ $4.5 |
| `claude-sonnet-5-5` | ≈ $0.022 | ≈ $2.2 |

`unflatten estimate`가 변환 전에 알려 주고, 변환 후에는 실제 토큰 사용량으로 계산한 비용이 로그와 리포트에 남습니다. 안전 분류기가 거절하면 서버 측 fallback으로 다른 모델이 자동 처리하고, 그래도 안 되면 규칙 기반으로 계속 진행합니다.

### 언어 (`--lang`)

`ko`(한국어+영어, 기본) · `en` · `ja` · `zh`. OCR 언어, 기본 글꼴(맑은 고딕 / Calibri / Yu Gothic / Microsoft YaHei), 텍스트 런의 언어 태그, 글자 폭 측정용 글꼴이 함께 바뀝니다. UI·로그 언어는 `--ui-lang ko|en`.

## 구조

```
unflatten/
  config.py      Config · 프리셋 · 언어 프로필 · 모델 가격표 · 슬라이드 범위
  inputs.py      PPTX/PDF/이미지 로딩, 통이미지 판별, PDF 텍스트층 + 텍스트 지운 렌더 + 내장 이미지 추출
  ocr.py         EasyOCR / PaddleOCR / Claude / 없음 백엔드, 엔진 선택(resolve_ocr)
  claude_client.py  구조화 출력(블록·개체·배경 종류·워터마크), fallback, 토큰·비용 집계
  layout.py      규칙 기반 묶기, PDF 문단 병합, 글꼴 파일로 폭·높이 맞춤, 색 추출
  inpaint.py     글자/워터마크 마스크, LaMa → OpenCV
  segment.py     단순 분리 · Claude+SAM 2 분리 · PDF 내장 이미지 조각
  builder.py     PPTX 조립(그룹, 이름, 대체 텍스트, 제목 자리표시자, 테마 글꼴·색, 글머리, 형광펜)
  cache.py       슬라이드별 캐시 (analysis.json + background.png + pieces/)
  pipeline.py    convert · estimate · rebuild · bundle_zip (장별 실패 격리, Claude 병렬 호출)
  report.py      전후 비교 미리보기 + HTML 리포트
  cli.py / app.py   CLI / Gradio UI
tests/           합성 슬라이드로 전 경로 회귀 테스트 (GPU·API 없이 실행)
notebooks/       Colab 노트북 (얇은 껍데기)
```

## 캐시

CLI는 출력 폴더의 `.unflatten-cache/`, 웹 UI는 `~/.cache/unflatten/`(환경변수 `UNFLATTEN_CACHE`로 변경)에 슬라이드별 인식 결과를 둡니다. 키는 슬라이드 픽셀 + 인식 설정 + 실제로 돈 백엔드(OCR 종류·Claude 사용 여부·모델)이므로, 키 없이 돌린 결과가 나중에 Claude 결과인 척 재사용되지 않습니다. 재조립은 manifest의 키를 그대로 씁니다.

## 한계

- 사진 배경 슬라이드는 그림 분리를 건너뜁니다(글자만 분리).
- 세로쓰기, 손글씨, 곡선 따라 흐르는 글자는 인식이 약합니다.
- 글자가 벡터 외곽선으로 변환된 PDF는 텍스트층이 없어 OCR 경로로 갑니다.
- 결과는 100% 원본과 같지 않습니다. 리포트에서 경고가 있는 장을 먼저 확인하세요.

## 테스트

```bash
pip install -e ".[dev]" && pytest -q
```
