# 영업 대화 더미 데이터

벡터 검색·정보 추출·후속 행동 추천을 개발하고 평가할 때 쓰는 가상 영업 시나리오입니다. 실제 인물·회사·통화가 아니며, 대화 안에 정답 정보를 의도적으로 넣고 그 정답(`gold_facts`, `gold_queries`)을 함께 기록합니다. 실무 자료는 동의와 이용 권한을 확보하기 전까지 이 폴더에 넣지 않습니다.

시나리오 하나가 파일 하나입니다. 파일을 추가하거나 고치면 `tests/test_sales_fixtures.py`가 id 참조, 발화 시간 순서, 화자-고객 연결, 약속 상태, 질문별 권한(볼 수 있는 대화와 나오면 안 되는 대화)을 검사합니다.

| 파일 | 내용 |
| ---- | ---- |
| `s001_nuri_precision.json` | 중소 제조사에 생산 실적 관리 SaaS 판매. 문의 → 데모 → 가격·ERP 연동 → 공장장 미팅·파일럿 → 구매팀 견적. 요구 변경(3→5라인), 모호한 약속 포함 (대화 5건) |
| `s002_hangyeol_foods_lost.json` | 식품 제조사에 HACCP 기록 기능 부재와 가격 차이로 경쟁 제품에 실패한 딜. 실패 사유와 재접촉 시점 (대화 3건) |
| `s003_onsaemiro_packaging.json` | 다른 판매사·업종: 포장재 납품사와 기존 거래처의 단가 인상 협상, 시즌 패키지 수주. 단가·MOQ·납기·물량 약정 (대화 3건) |
| `s004_sewon_tech_permissions.json` | 권한 검증: 같은 팀 다른 담당자의 딜. 비공개 2건 + 팀 공유 1건. 동료는 팀 공유만, 다른 팀은 아무것도 보면 안 됨 (대화 3건) |
| `s005_bareun_logistics_speakers.json` | 화자 구분 검증: 고객 4명 동시 참석, 같은 성의 과장 2명, 짧은 맞장구와 모호한 호칭 (대화 2건) |
| `s006_balgeunmiso_dental.json` | 의료: 치과 네트워크에 구강 스캐너. 구매 → 리스, 3대 → 1대로 범위 축소, 고객 서류 약속 기한 초과 (대화 3건) |
| `s007_seohae_district_public.json` | 공공기관: 구청에 민원 챗봇. 회계연도 예산·의회 심의·수의계약 한도·CSAP, 전 부서 → 1개 과로 범위 축소 (대화 3건) |
| `s008_dongsan_construction_safety.json` | 건설: 안전 웨어러블. 본사 안전팀과 현장소장의 이해 충돌, 위치정보 반발, 명단에 없는 작업자 발화(`role: unknown`) (대화 3건) |
| `s009_raonstay_hotel_amenity.json` | 호텔: S003과 같은 판매자의 다른 고객. 비슷한 단어(샘플·단가)가 나와도 고객사별로 구분되는지, 고객 답변 대기 중 (대화 3건) |
| `s010_saebom_farm_smartfarm.json` | 농업: 딸기 농원에 스마트팜 장비. 격식 없는 구어체, 정부 보조금과 재배 시기가 구매 시점을 좌우, 8동 → 4동 (대화 3건) |

사용자(`users`)는 시나리오 전체에서 같은 의미입니다: user 1·2는 1팀(플로우웍스), user 3은 2팀(그린팩), user 4는 3팀(데이터브릿지), user 5~8은 각각 4~7팀(메디온, 시빅웍스, 세이프온, 팜링크).

## 형식

```text
scenario_id, description, fictional, as_of   # as_of: 약속 상태(open/done)의 기준일
deal_status   open | won | lost
seller        이 딜의 영업 담당자(앱 사용자)
users[]       시나리오에 등장하는 사용자와 소속 팀 (user_id, team_id). 권한 필터 테스트용
company       고객사
contacts[]    고객사 담당자. role_in_deal: champion | influencer | decision_maker | procurement
interactions[]
  kind              meeting | call
  owner_user_id, visibility (private | team)
  participant_contact_ids
  utterances[]      app/schemas/voice.py의 Utterance와 같은 필드 (role: me | customer | unknown)
                    + contact_id: 화자 → 고객 연결 정답 (role이 me·unknown이면 null)
gold_facts[]        대화에서 추출되어야 하는 정보
  type              requirement | price_reaction | promise | purchase_timing | buying_signal | decision
  said_by           "me" 또는 contact id
  source_utterance_ids  근거 발화 (원문·시점 표시용)
  supersedes        요구가 바뀐 경우 이전 fact id
  promise 전용      owner(me | customer), due_at, due_text(원래 표현), status, resolved_in
gold_queries[]
  company_id        질문 대상 고객사. 검색은 이 고객사의 기록 안에서만 한다
  as_of             질문 시점 (미완료 약속 판단 기준)
  viewer_user_id    질문하는 사용자. 본인 대화 또는 같은 팀의 visibility=team 대화만 볼 수 있다
  kind=semantic     벡터 검색으로 찾을 질문 → relevant_utterance_ids
                    forbidden_interaction_ids: 이 사용자에게 절대 나오면 안 되는 대화
                    (relevant가 비어 있으면 "볼 수 있는 기록 없음"이 정답)
  kind=structured   facts 조회로 답할 질문 → relevant_fact_ids
  answer            기대 답변 요약
```

id는 시나리오 접두어를 붙여 전체에서 겹치지 않게 합니다: 고객사 `S001-CO1`, 담당자 `S001-C2`, 대화 `S001-I2`, 발화 `S001-I2-U08`(시나리오 1, 대화 2, 발화 8), 정답 정보 `S001-F11`, 질문 `S001-Q01`. 발화 시간은 글자 수로 추정한 값이라 실제 녹음과 다릅니다. 이 대본을 녹음해 STT·화자 분리를 평가할 때는 녹음 결과의 시간을 기준으로 합니다.
