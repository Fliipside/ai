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
│       ├── chunking.py      # 대화록을 검색 단위(청크)로 묶기
│       ├── service.py       # 청크·임베딩, 질문 임베딩, 재정렬
│       ├── extraction.py    # LLM으로 핵심 정보·사람 추출, 기억에 반영
│       └── relations.py     # 남은 약속 조회, 관계도
├── providers/           # 교체할 수 있는 모델·외부 API (base.py = 인터페이스)
│   ├── stt/                 # groq.py
│   ├── diarization/         # pyannote.py, sherpa_onnx.py
│   ├── speaker_embedding/   # pyannote.py, sherpa_onnx.py
│   ├── text_embedding/      # sentence_transformers.py (bge-m3 등)
│   ├── rerank/              # cross_encoder.py (bge-reranker-v2-m3)
│   └── llm/                 # gateway.py (대회 LLM 게이트웨이)
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
화자별 대화록 ──▶ 청크 (발화 여러 개를 약 500자로 묶고 앞 청크와 발화 1개 겹침, 시작/끝 시간 유지)
                    ↓ 텍스트 임베딩 (기본 BAAI/bge-m3, 1024차원, 정규화)
                 pgvector 저장 (backend)
질문 ──▶ 임베딩 ──▶ 고객사·권한 필터 후 코사인 유사도 상위 10개 ──▶ 재정렬 ──▶ 근거 발화와 시점
```

임베딩 모델은 `TEXT_EMBEDDING_MODEL`, 재정렬 모델은 `RERANK_MODEL`(기본 `BAAI/bge-reranker-v2-m3`)로 바꿀 수 있으며, 서로 다른 모델로 만든 벡터는 비교할 수 없습니다. 벡터 저장과 SQL 검색은 backend가 담당하고, 이 서비스는 청크 생성·임베딩·재정렬을 맡습니다. 두 모델은 첫 요청 때 한 번 불러오므로 첫 응답만 10초 안팎 걸립니다.

### 대화 기억 API

| 경로 | 요청 | 응답 `data` |
| ---- | ---- | ---- |
| `POST /api/v1/memory/chunks` | `utterances[]`(대화 1건, `id`·`start`·`end`·`speaker`·`role`·`text`), `context`(선택: `company`, `kind` meeting\|call, `occurred_at`, `title`) | `model`, `chunks[]`: `text`, `embed_text`, `start`, `end`, `utterance_ids`, `vector` |
| `POST /api/v1/memory/embed-query` | `text` (1~1000자) | `model`, `vector` |
| `POST /api/v1/memory/rerank` | `query`, `candidates[]`(`id`, `text`, 최대 50개), `top_k`(선택) | `model`, `results[]`: `id`, `score` (높은 순) |

backend 흐름:

1. 대화가 저장되면 `/chunks`를 호출해 받은 청크의 `text`(화면에 보여줄 원문), `embed_text`, `utterance_ids`, 시간, `vector`, `model`을 pgvector에 저장합니다. `context`를 보내면 청크마다 `회사 | 날짜 미팅/통화 | 제목` 머리말을 붙여 임베딩하며, 검색 정확도가 올라가므로 보내는 것을 권장합니다. 날짜는 `occurred_at`에 적힌 시간대 기준입니다(한국 시간은 `+09:00`).
2. 질문이 오면 `/embed-query`로 벡터를 받아, 고객사·권한으로 거른 뒤 같은 `model`의 청크 중 코사인 거리(`<=>`) 상위 10개를 고릅니다.
3. 그 10개의 `embed_text`를 `/rerank`에 보내 다시 정렬하고 상위 결과를 사용합니다.

정답셋 1,467개 질문(시나리오 100개, 바꿔 말한 질문 978개 포함) 기준으로 1위 정답률은 벡터 검색만 80.9%, 재정렬 후 87.5%입니다. 권한 위반과 다른 고객사 결과는 0건입니다. 로컬 CPU(Apple Silicon)에서 모델을 불러온 뒤 응답 시간은 `/embed-query` 약 50ms, `/rerank`(후보 10개) 약 300ms, `/chunks`(대화 1건) 약 140ms입니다.

### 검색 품질 평가

```bash
python scripts/retrieval_eval.py --model BAAI/bge-m3 --k 5 --max-chars 500 --overlap 1 \
    --out data/eval/result.json
```

`fixtures/sales/`의 대화를 청크로 나누고 임베딩한 뒤, 정답 질문마다 pgvector 검색과 같은 방식(고객사·권한 필터 → 코사인 유사도 순)으로 상위 k개를 골라 hit@k, recall@k, MRR, 권한 위반 건수를 출력합니다. 청크 앞에는 고객사·날짜·대화 제목 머리말을 붙여 임베딩합니다(`--no-context`로 끔). `--rerank BAAI/bge-reranker-v2-m3`는 벡터 검색 상위 `--pool`개를 재정렬 모델로 다시 정렬하고, `--no-company-filter`는 사용자가 볼 수 있는 모든 고객사에서 검색합니다.

### 임베딩 모델·청킹 비교

```bash
python scripts/compare_embeddings.py models                             # 후보 모델 7개 (500자, 겹침 1)
python scripts/compare_embeddings.py chunking --model BAAI/bge-m3         # 크기·겹침·머리말 24가지
```

같은 정답셋으로 hit@1·hit@3·recall@3·MRR과 질문 1개 임베딩 시간을 비교합니다. `open@1`은 고객사·권한 필터 없이 전체 청크에서 찾은 점수로, 모델 간 차이를 더 크게 드러냅니다. 청크가 클수록 후보 수가 줄어 무작위로 골라도 맞힐 확률(`random@1`)이 올라가므로, 청크 크기는 `hit@1 - random@1`과 `open@1`로 비교합니다. 작을수록 정확한 위치를 찾고 클수록 상위 3개에 근거가 빠짐없이 들어가며, 상위 결과를 LLM에 넘기는 구조라 기본값을 500자로 둡니다.

### pgvector 스키마 `memory`

대화 기억 검색 데이터는 DB의 별도 스키마 `memory`에 둡니다. backend의 `public` 테이블과 섞이지 않으며, 최종 구조는 backend와 합의한 뒤 마이그레이션으로 옮깁니다.

| 파일 | 하는 일 |
| ---- | ---- |
| `scripts/sql/00_check.sql` | 서버 확인 (읽기만 함): 버전, pgvector, 기존 테이블, 권한 |
| `scripts/sql/01_memory_schema.sql` | 스키마 `memory`와 테이블·인덱스 생성 (다시 실행해도 안전) |
| `scripts/load_memory_fixtures.py` | `fixtures/sales`를 청크·임베딩해 적재 SQL 파일 생성. 실행하면 더미 데이터 행만 지우고 다시 넣음 |
| `scripts/sql/02_inspect.sql` | 적재 결과, 고객별 타임라인, 미완료 약속, 벡터 유사도 조회 |
| `scripts/sql/03_clear.sql` | 테이블 구조는 두고 `memory`의 데이터만 모두 비움 |

```bash
python -m pip install -e ".[memory]"
python scripts/load_memory_fixtures.py --out data/memory/load_fixtures.sql
```

만든 SQL 파일은 DBeaver에서 열어 스크립트 전체 실행(`⌥ + X`)하거나, 서버의 DB 컨테이너 안에서 `psql`로 실행합니다. 적재에는 DB 비밀번호가 필요 없습니다. SQL 검색 평가(`retrieval_eval.py --backend pg`)만 SSH 터널과 `.env`의 `MEMORY_DB_URL`이 필요합니다(`python -m pip install -e ".[memory-db]"`).

## 핵심 정보 추출·관계 기억 구성

```text
고객사의 대화들 (날짜 순)
   ↓ 대화 1건마다 LLM 1번 호출: 대화 + 지금까지의 기억(알려진 사람, 남은 약속, 기존 정보)
추출 결과: 사람(이름·성·직함·역할), 핵심 정보 6종(요구사항·가격 반응·약속·구매 시기·구매 신호·결정),
          지켜진 약속, 바뀐 정보(이전 정보를 대체)
   ↓ 근거 발화·화자가 실제로 있는지 검사하고 기억에 반영
고객사 기억: 사람, 정보(근거 발화, 약속 상태·기한), 변경 연결 → 관계도(노드·연결선)
```

| 파일 | 하는 일 |
| ---- | ---- |
| `app/features/memory/extraction.py` | LLM 지시문, 대화별 추출, 사람 연결(이름 → 성+직함 → 문맥), 기억 반영 |
| `app/features/memory/relations.py` | 남은 약속 조회, 관계도 노드·연결선 만들기 |
| `app/providers/llm/gateway.py` | 대회 LLM 게이트웨이(OpenAI 호환) 호출, 결과 형식 검사·1회 재요청, 토큰 사용량 집계 |
| `scripts/memory_eval.py` | 더미 데이터로 추출 품질 평가, 고객사별 기억 JSON과 관계도 HTML 저장 |

기본 모델은 `claude-sonnet-5-5`에 사고량 `low`입니다(`LLM_MODEL`, `LLM_REASONING_EFFORT`). Opus는 답보다 숨은 사고 토큰이 10배 넘게 나와 비용의 약 90%를 차지했고, Sonnet low는 비용이 약 15분의 1이면서 정보 찾기 점수는 비슷하거나 높았습니다. `LLM_API_KEY`는 팀 공유 키라 `.env`에만 넣고 커밋하지 않습니다.

```bash
python -m pip install -e ".[llm]"
python scripts/memory_eval.py --gold          # LLM 없이 정답으로 기억·관계도 만들기 (채점 확인용)
python scripts/memory_eval.py                 # 전체 평가 (대화 310건, 약 9천 원)
python scripts/memory_eval.py fixtures/sales/s01*.json   # 일부만 평가
```

LLM 답은 `data/memory/extractions/`에 저장해 같은 요청은 다시 호출하지 않으며, 실행이 끝나면 토큰 수와 예상 비용을 출력합니다. 시나리오 번호 끝자리 8·9·0은 지시문 개선에 쓰지 않는 확인용으로 따로 채점합니다.

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
| `POST /api/v1/memory/chunks` | 대화 1건의 발화 목록 | 200       |
| `POST /api/v1/memory/embed-query` | 질문 | 200       |
| `POST /api/v1/memory/rerank` | 질문과 후보 청크 | 200       |

구현 상태:

| 부분 | 상태 |
| ---- | ---- |
| `utils/audio.py` (16kHz 모노 변환, 10분 단위 분할) | 구현 |
| `providers/stt/groq.py` (Groq Whisper) | 구현 |
| `features/voice/` (합치기, "나" 식별, 처리 흐름) | 구현 |
| `providers/diarization/pyannote.py`, `providers/speaker_embedding/pyannote.py` (community-1) | 구현 |
| `providers/diarization/sherpa_onnx.py`, `providers/speaker_embedding/sherpa_onnx.py` | 구현 |
| `scripts/voice_benchmark.py` (엔진 비교) | 구현 |
| `fixtures/sales/` (가상 영업 시나리오 100개와 정답셋, [형식](fixtures/sales/README.md)) | 구현 |
| `features/memory/chunking.py`, `providers/text_embedding/` (청크 생성, 텍스트 임베딩) | 구현 |
| `scripts/retrieval_eval.py` (검색 품질 평가), `scripts/compare_embeddings.py` (모델·청킹 비교), `scripts/load_memory_fixtures.py`, `scripts/sql/` (스키마 `memory`) | 구현 |
| `api/v1/memory.py`, `features/memory/service.py`, `providers/rerank/` (대화 기억 API) | 구현 |
| `features/memory/extraction.py`, `relations.py`, `providers/llm/`, `scripts/memory_eval.py` (핵심 정보 추출·관계 기억·관계도) | 구현 (로컬 더미 데이터) |
| pgvector 저장·검색 (backend) | 미구현 |

API 연결 전까지 voice API는 501을 반환합니다. "나" 판정 기준값 `VOICE_MATCH_THRESHOLD`(코사인 유사도)는 엔진 비교 테스트 후 정하며, 설정되지 않은 상태에서 사용자 목소리 정보와 함께 요청하면 503(`VOICE_NOT_CONFIGURED`)을 반환합니다.
