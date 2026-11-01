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
녹음 파일
  ├─▶ Groq Whisper API (large-v3) ──▶ 문장 + 시작/끝 시간
  └─▶ 화자 분리 (pyannote 또는 sherpa-onnx, CPU)
         + 사용자 목소리 비교 ──▶ 화자 구간 (나 / 고객)
                    ↓ 시간 기준으로 합치기
               화자별 대화록
```

화자 분리 엔진은 `DIARIZATION_ENGINE`(`pyannote` | `sherpa_onnx`)으로 고릅니다. 사용자 목소리 임베딩은 같은 엔진으로 만든 것끼리만 비교할 수 있습니다.

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

`.env.example`을 참고해 `.env`를 만듭니다. pyannote는 Hugging Face에서 `pyannote/speaker-diarization-community-1` 약관에 동의한 뒤 `HF_TOKEN`을 설정해야 합니다. 오디오 변환에는 `ffmpeg`가 필요합니다.

## 현재 API 상태

| 경로                         | 입력                  | 현재 동작 |
| ---------------------------- | --------------------- | --------- |
| `GET /health`                | 없음                  | 200       |
| `POST /api/v1/voice/analyze` | `file` 오디오 업로드  | 501       |
| `POST /api/v1/voice/enroll`  | `file` 오디오 업로드  | 501       |

구현 상태:

| 부분 | 상태 |
| ---- | ---- |
| `utils/audio.py` (16kHz 모노 변환, 10분 단위 분할) | 구현 |
| `providers/stt/groq.py` (Groq Whisper) | 구현 |
| `features/voice/` (합치기, "나" 식별, 처리 흐름) | 구현 |
| `providers/diarization/`, `providers/speaker_embedding/` | 미구현 |

화자 분리 엔진이 구현될 때까지 voice API는 501을 반환합니다. "나" 판정 기준값 `VOICE_MATCH_THRESHOLD`(코사인 유사도)는 엔진 비교 테스트 후 정하며, 설정되지 않은 상태에서 사용자 목소리 정보와 함께 요청하면 503(`VOICE_NOT_CONFIGURED`)을 반환합니다.
