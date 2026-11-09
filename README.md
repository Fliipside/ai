# Flipside ai

플립사이드 AI 추론 서비스입니다. backend가 `AI_SERVICE_URL`로 호출하며, 음성 인식·화자 분리 등 모델 호출과 결과 조합을 담당합니다. 인증과 데이터 저장은 backend가 담당하고, 이 서비스는 저장소를 갖지 않습니다.

## 코드 경계

```text
app/
├── main.py              # 앱 생성, 공통 오류 처리, 상태 확인
├── api/v1/              # 기능별 HTTP 라우터
├── schemas/             # 요청·응답과 기능 간 데이터 계약
├── features/            # 기능별 처리 흐름 (기능 하나당 폴더 하나)
│   └── voice/
│       ├── pipeline.py      # STT + 화자 분리 + "나" 식별 + 합치기
│       ├── merge.py         # STT 문장과 화자 구간을 시간 기준으로 합치기
│       └── speaker_id.py    # 사용자 목소리와 비교해 나/고객 라벨링
├── providers/           # 교체할 수 있는 모델·외부 API (base.py = 인터페이스)
│   ├── stt/                 # groq.py
│   ├── diarization/         # pyannote.py, sherpa_onnx.py
│   └── speaker_embedding/   # pyannote.py, sherpa_onnx.py
├── core/                # 환경 설정과 공통 오류
└── utils/               # 오디오 변환 등 공통 도구
scripts/                 # 벤치마크 스크립트
tests/
```

요청은 `api/v1/` → `features/<기능>/` → `providers/<종류>/` 순서로 흐릅니다. 새 AI 기능은 `features/`, `api/v1/`, `schemas/`에 기능 이름으로 추가하고, 같은 역할의 엔진이 여러 개면 `providers/<종류>/`에 구현체를 나란히 둡니다.

## 음성 분석 구성

```text
녹음 파일 (최대 60분)
  ① Groq Whisper API (large-v3) ──▶ 문장 + 시작/끝 시간
  ② 화자 분리 (pyannote 또는 sherpa-onnx, CPU)
         + 사용자 목소리 비교 ──▶ 화자 구간 (me / A / B …)
                    ↓ 시간 기준으로 합치기
               화자별 대화록
```

분석 결과는 사람별로 나눕니다. 사용자 목소리와 가장 비슷한 화자(유사도가 `VOICE_MATCH_THRESHOLD` 이상)는 `me`, 나머지는 대화록에서 처음 말한 순서대로 `A`, `B`, `C`…(26명 넘으면 `AA`, `AB`…)입니다. 목소리 정보가 없거나 기준을 넘는 화자가 없으면 `me` 없이 모두 `A`, `B`…이고 역할은 `unknown`입니다. 역할(`role`)은 `me` / `customer` / `unknown`이고, `speakers`에 사람별 역할과 발화 시간(초)이 들어갑니다. 화자 수는 정하지 않고 엔진이 추정합니다.

처리 규칙:

- STT를 먼저 하고 화자 분리를 합니다. Groq 한도(429)에 걸리면 몇 분짜리 화자 분리를 하기 전에 실패하고, Groq의 `Retry-After`를 그대로 전달합니다.
- 분석은 한 번에 하나만 합니다. 분석 중에 다시 요청하면 503 `VOICE_BUSY`와 `Retry-After`(`VOICE_BUSY_RETRY_SECONDS`, 기본 60초)를 돌려줍니다. 목소리 등록은 별도 lock을 써서 분석 중에도 바로 처리됩니다.
- 녹음은 `VOICE_MAX_AUDIO_SECONDS`(기본 3600초)까지 받습니다. 넘으면 모델을 돌리기 전에 400 `AUDIO_TOO_LONG`입니다. 등록 음성은 `VOICE_ENROLL_MIN_SECONDS`(기본 5초) 이상이어야 합니다.
- 서버 시작 시 모델을 미리 불러오려면 `VOICE_PRELOAD_MODELS=true`로 둡니다(운영 권장).

화자 분리 엔진은 `DIARIZATION_ENGINE`(`pyannote` | `sherpa_onnx`)으로 고릅니다. 사용자 목소리 임베딩은 같은 엔진으로 만든 것끼리만 비교할 수 있습니다. sherpa-onnx는 임베딩 모델을 바꿀 수 있으므로 엔진 이름에 모델명이 들어갑니다(예: `sherpa_onnx:wespeaker_en_voxceleb_resnet34_LM`).

sherpa-onnx 모델은 [k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)에서 받아 `SHERPA_SEGMENTATION_MODEL`(`speaker-segmentation-models`의 `sherpa-onnx-pyannote-segmentation-3-0/model.onnx`)과 `SHERPA_EMBEDDING_MODEL`(`speaker-recongition-models`의 `.onnx`)에 경로를 적습니다. 화자 수를 모를 때는 `SHERPA_CLUSTER_THRESHOLD`로 화자를 묶습니다.

### 엔진 비교

```bash
python scripts/voice_benchmark.py data/voice/*.m4a \
    --engines pyannote sherpa_onnx \
    --sherpa-embedding <임베딩 모델.onnx> ... \
    --num-speakers auto 2 --enroll <내 목소리.wav> --out result.json
```

정답 라벨 없이 속도(실시간 대비 배율), 찾은 화자 수, 엔진 간 구간 일치율, 등록 목소리와의 유사도를 출력합니다.

## 로컬 실행

Python **3.11.x 고정**입니다 (3.12 이상은 지원하지 않습니다).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
# 화자 분리 엔진 (필요한 것만)
python -m pip install -e ".[voice-pyannote]"
python -m pip install -e ".[voice-sherpa]"
uvicorn app.main:app --reload
```

`.env.example`을 참고해 `.env`를 만듭니다. `.env`는 실행 위치에서 상위 폴더 방향으로 찾으므로, 저장소 루트에 두면 그 아래 모든 worktree가 함께 씁니다. pyannote는 Hugging Face에서 `pyannote/speaker-diarization-community-1` 약관에 동의한 뒤 `HF_TOKEN`을 설정해야 합니다. 모델은 처음 실행할 때 Hugging Face 캐시(`HF_HUB_CACHE`)에 받으며, 받은 뒤에는 토큰 없이도 로드됩니다. pyannote의 사용 통계 전송은 기본으로 끕니다(`PYANNOTE_METRICS_ENABLED=false`). 분석 창(10초)을 옮기는 간격 `PYANNOTE_SEGMENTATION_STEP`은 기본 0.5입니다. pyannote 기본값 0.1보다 약 5배 빠르고, 실제 회의 녹음에서 결과가 98% 같았습니다. 오디오 변환에는 `ffmpeg`가 필요합니다.

## 현재 API 상태

| 경로                         | 입력                  | 현재 동작 |
| ---------------------------- | --------------------- | --------- |
| `GET /health`                | 없음                  | 200       |
| `POST /api/v1/voice/analyze` | `file` 오디오 + 선택 `voiceprint`(JSON 문자열) | 200 (몇 분 걸림) |
| `POST /api/v1/voice/enroll`  | `file` 사용자 혼자 말한 오디오(5초 이상) | 200       |

구현 상태:

| 부분 | 상태 |
| ---- | ---- |
| `utils/audio.py` (16kHz 모노 변환, 10분 단위 분할) | 구현 |
| `providers/stt/groq.py` (Groq Whisper) | 구현 |
| `features/voice/` (합치기, "나" 식별, 처리 흐름) | 구현 |
| `providers/diarization/pyannote.py`, `providers/speaker_embedding/pyannote.py` (community-1) | 구현 |
| `providers/diarization/sherpa_onnx.py`, `providers/speaker_embedding/sherpa_onnx.py` | 구현 |
| `scripts/voice_benchmark.py` (엔진 비교) | 구현 |

"나" 판정 기준값 `VOICE_MATCH_THRESHOLD`(코사인 유사도)는 기본 0.5입니다. 실제 회의 녹음에서 사용자는 약 0.84, 다른 사람은 약 0.2였습니다(녹음 2개, 한 사람 기준).
