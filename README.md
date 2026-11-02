# Flipside ai

플립사이드 AI 추론 서비스입니다. backend가 `AI_SERVICE_URL`로 호출하며, 음성 인식·화자 분리 등 모델 호출과 결과 조합을 담당합니다. 인증과 데이터 저장은 backend가 담당하고, 이 서비스는 저장소를 갖지 않습니다.

## 코드 경계

```text
app/
├── main.py              # 앱 생성, 공통 오류 처리, 상태 확인
├── api/v1/              # 기능별 HTTP 라우터
├── schemas/             # 요청·응답과 기능 간 데이터 계약
├── features/            # 기능별 처리 흐름 (기능 하나당 폴더 하나)
│   ├── voice/
│   │   ├── pipeline.py      # STT + 화자 분리 + "나" 식별 + 합치기
│   │   ├── merge.py         # STT 문장과 화자 구간을 시간 기준으로 합치기
│   │   └── speaker_id.py    # 사용자 목소리와 비교해 나/고객 라벨링
│   └── memory/
│       └── chunking.py      # 대화록을 검색 단위(청크)로 묶기
├── providers/           # 교체할 수 있는 모델·외부 API (base.py = 인터페이스)
│   ├── stt/                 # groq.py
│   ├── diarization/         # pyannote.py, sherpa_onnx.py
│   ├── speaker_embedding/   # pyannote.py, sherpa_onnx.py
│   └── text_embedding/      # sentence_transformers.py (bge-m3 등)
├── core/                # 환경 설정과 공통 오류
└── utils/               # 오디오 변환 등 공통 도구
scripts/                 # 벤치마크·평가 스크립트
fixtures/
└── sales/               # 가상 영업 대화 시나리오와 정답셋 (검색·추출 평가용)
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

## 대화 기억 검색 구성

```text
화자별 대화록 ──▶ 청크 (발화 여러 개를 약 300자로 묶고 앞 청크와 발화 1개 겹침, 시작/끝 시간 유지)
                    ↓ 텍스트 임베딩 (기본 BAAI/bge-m3, 1024차원, 정규화)
                 pgvector 저장 (backend)
질문 ──▶ 임베딩 ──▶ 고객사·권한 필터 후 코사인 유사도 상위 k개 ──▶ 근거 발화와 시점
```

임베딩 모델은 `TEXT_EMBEDDING_MODEL`로 바꿀 수 있으며, 서로 다른 모델로 만든 벡터는 비교할 수 없습니다. 벡터 저장과 SQL 검색은 backend가 담당하고, 이 서비스는 청크 생성과 임베딩을 맡습니다.

### 검색 품질 평가

```bash
python scripts/retrieval_eval.py --model BAAI/bge-m3 --k 5 --max-chars 300 --overlap 1 \
    --out data/eval/result.json
```

`fixtures/sales/`의 대화를 청크로 나누고 임베딩한 뒤, 정답 질문마다 pgvector 검색과 같은 방식(고객사·권한 필터 → 코사인 유사도 순)으로 상위 k개를 골라 hit@k, recall@k, MRR, 권한 위반 건수를 출력합니다. 청크 앞에는 고객사·날짜·대화 제목 머리말을 붙여 임베딩합니다(`--no-context`로 끔). `--rerank BAAI/bge-reranker-v2-m3`는 벡터 검색 상위 `--pool`개를 재정렬 모델로 다시 정렬하고, `--no-company-filter`는 사용자가 볼 수 있는 모든 고객사에서 검색합니다.

### pgvector sandbox

실제 SQL 검색은 별도 스키마 `memory_sandbox`에서 실험합니다. 운영 테이블은 backend 마이그레이션으로 만듭니다.

1. DBeaver에서 `scripts/sql/00_check.sql`(확인용, 읽기만 함)과 `scripts/sql/01_sandbox_schema.sql`(스키마 생성)을 실행합니다.
2. SSH 터널을 열고 `.env`에 `MEMORY_DB_URL`을 적습니다 (`.env.example` 참고).
3. 적재하고 SQL로 평가합니다.

```bash
python -m pip install -e ".[memory,memory-db]"
python scripts/load_memory_sandbox.py              # memory_sandbox 테이블을 비우고 다시 채움
python scripts/retrieval_eval.py --backend pg --k 3
```

`scripts/sql/02_inspect.sql`로 적재 결과, 고객별 타임라인, 미완료 약속을 DBeaver에서 확인합니다. `DROP SCHEMA memory_sandbox CASCADE`로 전부 지울 수 있습니다.

## 로컬 실행

Python **3.11.x 고정**입니다 (3.12 이상은 지원하지 않습니다).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
# 화자 분리 엔진 (필요한 것만)
python -m pip install -e ".[voice-pyannote]"
python -m pip install -e ".[voice-sherpa]"
# 대화 기억 검색 (텍스트 임베딩)
python -m pip install -e ".[memory]"
uvicorn app.main:app --reload
```

`.env.example`을 참고해 `.env`를 만듭니다. `.env`는 실행 위치에서 상위 폴더 방향으로 찾으므로, 저장소 루트에 두면 그 아래 모든 worktree가 함께 씁니다. pyannote는 Hugging Face에서 `pyannote/speaker-diarization-community-1` 약관에 동의한 뒤 `HF_TOKEN`을 설정해야 합니다. 모델은 처음 실행할 때 Hugging Face 캐시(`HF_HUB_CACHE`)에 받으며, 받은 뒤에는 토큰 없이도 로드됩니다. pyannote의 사용 통계 전송은 기본으로 끕니다(`PYANNOTE_METRICS_ENABLED=false`). 오디오 변환에는 `ffmpeg`가 필요합니다.

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
| `providers/diarization/pyannote.py`, `providers/speaker_embedding/pyannote.py` (community-1) | 구현 |
| `providers/diarization/sherpa_onnx.py`, `providers/speaker_embedding/sherpa_onnx.py` | 구현 |
| `scripts/voice_benchmark.py` (엔진 비교) | 구현 |
| `fixtures/sales/` (가상 영업 시나리오 10개와 정답셋, [형식](fixtures/sales/README.md)) | 구현 |
| `features/memory/chunking.py`, `providers/text_embedding/` (청크 생성, 텍스트 임베딩) | 구현 |
| `scripts/retrieval_eval.py` (검색 품질 평가), `scripts/load_memory_sandbox.py`, `scripts/sql/` (pgvector sandbox) | 구현 |
| 임베딩 API, pgvector 저장·검색 (backend) | 미구현 |

API 연결 전까지 voice API는 501을 반환합니다. "나" 판정 기준값 `VOICE_MATCH_THRESHOLD`(코사인 유사도)는 엔진 비교 테스트 후 정하며, 설정되지 않은 상태에서 사용자 목소리 정보와 함께 요청하면 503(`VOICE_NOT_CONFIGURED`)을 반환합니다.
