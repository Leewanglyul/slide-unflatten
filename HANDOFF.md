# slide-unflatten 인수인계 문서

작성일: 2026-10-24 · 상태: **코드 완성, 로컬 커밋 1개, GitHub 미업로드**

## 1. 무엇을 만드는가

NotebookLM·Gemini·Canva·Gamma 등에서 내보낸 **통이미지 슬라이드(PPTX/PDF/PNG)** 를
글자와 그림을 고칠 수 있는 PPTX로 바꾸는 도구. 원래 Colab 노트북 한 장(`slide_unflatten_1.ipynb`)이었던 것을
범용 서비스로 재구성했다: **pip 패키지 + CLI + Gradio 웹 UI + 얇은 Colab 노트북**.

처리 흐름: 입력 판별 → (PDF 텍스트층이면 바로 복원) → OCR(EasyOCR) → Claude(교정·묶기·개체 탐지) →
글자 지우기(LaMa/OpenCV) → 그림 분리(SAM 2 / 단순) → 캐시 저장 → PPTX 조립 → HTML 리포트.

## 2. 전달물

| 파일 | 내용 |
|---|---|
| `slide-unflatten.bundle` | git 저장소 전체(커밋 포함). `git clone slide-unflatten.bundle slide-unflatten` 으로 복원 |
| `slide-unflatten-src.zip` | 같은 소스의 zip (git 이력 없음) |
| `HANDOFF.md` | 이 문서 (저장소 루트에도 포함) |
| `README.md` | 설치·사용·구조·한계 설명 |

## 3. GitHub에 올리는 방법

```bash
git clone slide-unflatten.bundle slide-unflatten
cd slide-unflatten
git remote set-url origin https://github.com/<GitHub-ID>/slide-unflatten.git   # 빈 저장소를 먼저 만들 것
git push -u origin main
```
올린 뒤 `notebooks/unflatten_colab.ipynb` 1번 셀의 `REPO_URL`을 실제 주소로 바꾼다.

## 4. 저장소 구조

```
pyproject.toml           패키지 정의. extras: claude / ocr / segment / inpaint / ui / all / dev
requirements-colab.txt   Colab용 핀 버전
unflatten/
  config.py        Config 데이터클래스, 프리셋(text/objects/full), 언어 프로필(ko/en/ja/zh), 모델 가격표, 슬라이드 범위 파서
  i18n.py          로그·UI 문자열 ko/en
  types.py         Line / Block / Piece / SlideAnalysis / SlideSummary (JSON 왕복 가능)
  inputs.py        PPTX·PDF·이미지 로딩. 통이미지 판별(그림이 85% 이상 덮고 실제 텍스트 없음), 노트·그림 위치 보존,
                   PDF 텍스트층 추출 + 텍스트만 지운 렌더(redaction) + 내장 이미지 추출
  ocr.py           EasyOCR / Claude(엔진 없을 때) / None 백엔드
  claude_client.py 구조화 출력(pydantic) 호출, 서버 측 fallback, 토큰·비용 집계, 키 검증, 프롬프트
  layout.py        규칙 기반 묶기, PDF 문단 병합, 글꼴 파일로 폭·높이 맞춤(FontMeasurer), 색 추출, finalize_blocks
  inpaint.py       글자/워터마크 마스크, LaMa → OpenCV 폴백
  segment.py       단순 분리(윤곽) · Claude 상자 + SAM 2 분리(가려진 부분 채움) · PDF 내장 이미지 조각
  fonts.py         글꼴 이름 읽기(fontTools), 보통/굵게 자동 배정, 글꼴 번들 복사
  builder.py       PPTX 조립: 숨김 원본, 배경, 그림 그룹, 텍스트 그룹, 제목 자리표시자, 테마 글꼴·강조색, 글머리, 형광펜, 대체 텍스트
  cache.py         슬라이드별 캐시(analysis.json + background.png + pieces/), 텍스트 수정 반영
  pipeline.py      convert / estimate / rebuild / bundle_zip. 장별 실패 격리, Claude 병렬 호출, manifest·리포트 생성
  report.py        전후 비교 미리보기 이미지 + HTML 리포트
  cli.py           unflatten convert | estimate | rebuild | ui | doctor
  app.py           Gradio UI (변환 / 텍스트 수정 / 재조립 탭, ko/en)
tests/             합성 슬라이드로 전 경로 회귀 테스트. GPU·API 없이 실행 (13개)
notebooks/unflatten_colab.ipynb   Colab 폼 기반 노트북
samples/make_samples.py           데모 입력 생성
.github/workflows/tests.yml       CI
```

## 5. 실행

```bash
pip install -e ".[claude,ui,dev]"       # CPU 환경 (PDF 텍스트층·OpenCV 경로)
pip install -e ".[all]"; pip install --no-deps simple-lama-inpainting   # GPU 전체 경로
export ANTHROPIC_API_KEY=sk-ant-...
unflatten doctor
python samples/make_samples.py
unflatten convert samples/out/text.pdf --no-claude --lang en --mode objects
unflatten ui
pytest -q
```

## 6. 검증된 것 / 안 된 것

**검증됨 (이 환경: CPU, GPU 없음, API 키 없음)**
- `pytest` 13개 통과: 입력 판별, PDF 텍스트층, 통이미지 경로(가짜 OCR), 캐시·재조립, PPTX 혼합 덱 보존, 슬라이드 범위, 실패 격리, 단위 함수.
- CLI convert/rebuild/estimate/doctor 동작. 결과 PPTX를 LibreOffice로 렌더링해 위치·글머리·그림 분리·제목 자리표시자·테마 글꼴 XML 확인.
- Gradio 6.30에서 UI 구성(build_demo) 성공.

**검증 안 됨 → 인수자가 먼저 확인할 것**
1. **GPU 경로 실제 실행**: EasyOCR, SAM 2(ultralytics), LaMa(simple-lama-inpainting). 코드는 원본 노트북을 옮긴 것이지만 이 환경에서 설치·실행하지 않았다. Colab T4에서 실제 덱(예: Polar_Frontiers.pptx)으로 `--mode full` 실행.
2. **Claude 실제 호출**: SDK 2.3.0 시그니처만 확인. `client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default", output_format=SlideOut, output_config={"effort": "low"})`. 400이 나면 `_fallbacks_ok=False`로 자동 폴백해 일반 `messages.parse`를 쓴다.
3. **Gradio 브라우저 조작**: 업로드·진행률·다운로드 UX 실제 테스트 안 함.
4. **PowerPoint/Google Slides에서 열기**: 제목 자리표시자(`p:ph type="title"`)를 빈 레이아웃 슬라이드에 넣은 것, 그룹 내 텍스트 상자, 형광펜 XML이 실제 PowerPoint·Google Slides에서 어떻게 보이는지 확인. 문제가 있으면 `Config.title_as_placeholder=False`, `group_shapes=False`로 끌 수 있다.

## 7. 주요 설계 결정

- 원본이 PPTX이면 **원본 파일을 그대로 열어** 통이미지 슬라이드만 바꾼다. 편집 가능한 슬라이드·노트·크기·테마가 그대로 남는다. 이 경우 저자 테마를 건드리지 않으려 테마 글꼴 대신 글꼴 이름을 직접 쓴다.
- PDF에 텍스트층이 있으면 OCR·Claude를 쓰지 않는다. PyMuPDF redaction으로 텍스트만 지운 배경을 렌더링하고 내장 이미지를 조각으로 뺀다.
- 캐시 키 = 슬라이드 이미지 + 인식에 영향을 주는 설정(mode, language, model, ocr, dpi, watermark). 글꼴·그룹·숨김 같은 조립 옵션은 키에 없어서 `rebuild`가 캐시를 그대로 쓴다.
- 글자 크기는 측정용 글꼴 파일로 실제 렌더링해 폭과 높이에 동시에 맞춘다(한글 0.92 비율 가정은 폴백).
- Claude 응답에 `background_type`을 받아 사진 배경이면 그림 분리를 건너뛴다. `watermarks` 박스는 인페인팅으로 지운다.
- 슬라이드 한 장이 실패하면 원본 이미지를 넣고 계속 진행한다. 리포트에 빨간색으로 표시.

## 8. 남은 작업 제안 (우선순위순)

1. GPU에서 실제 덱으로 full 모드 실행 후 `MIN_OBJECT_AREA`, 마스크 팽창 크기, 제목 임계값(0.05×높이) 튜닝.
2. Claude 프롬프트 다국어화(현재 프롬프트는 한국어 지시문, 출력 언어는 슬라이드 언어를 따르도록 지시됨).
3. `pyproject.toml` 의존성 상한 핀 확정(ultralytics, easyocr 버전은 Colab 이미지 변경에 취약).
4. Real-ESRGAN 같은 초해상 옵션(현재 `upscale`은 Lanczos).
5. Hugging Face Spaces 배포(app.py 그대로 사용 가능, ZeroGPU 권장).
6. 회귀용 실제 덱 샘플과 정답 텍스트를 `tests/fixtures`에 추가해 텍스트 정확도·위치 IoU 측정.

## 9. 참고

- 원본 노트북(출발점): `slide_unflatten_1.ipynb` (Gemini Notebook·NotebookLM 슬라이드용 단일 노트북).
- Claude 모델·가격(2026-10 기준): Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 (백만 토큰당 입력/출력). 장당 대략 Opus $0.045.
