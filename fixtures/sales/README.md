# 영업 대화 더미 데이터

벡터 검색·정보 추출·후속 행동 추천을 개발하고 평가할 때 쓰는 가상 영업 시나리오입니다. 실제 인물·회사·통화가 아니며, 대화 안에 정답 정보를 의도적으로 넣고 그 정답(`gold_facts`, `gold_queries`)을 함께 기록합니다. 실무 자료는 동의와 이용 권한을 확보하기 전까지 이 폴더에 넣지 않습니다.

시나리오 하나가 파일 하나입니다. S011~S100은 짧은 초안을 `scripts/build_sales_fixture.py`로 변환해 만들었습니다(발화 id·화자 라벨·시간 자동 생성, 참조 검사). 파일을 추가하거나 고치면 `tests/test_sales_fixtures.py`가 id 참조, 발화 시간 순서, 화자-고객 연결, 약속 상태, 질문별 권한(볼 수 있는 대화와 나오면 안 되는 대화)을 검사합니다.

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

### S011~S100 (판매 팀 8~37)

| 파일 | 판매사 · 제품 | 고객 (업종) | 딜 | 대화 |
| ---- | ---- | ---- | ---- | ---- |
| `s011_moastyle_fashion_wms.json` | 로지큐브 · 창고관리(WMS) SaaS | 모아스타일 (온라인 패션 쇼핑몰 물류센터) | open | 3 |
| `s012_hanul_meddist_gdp_wms.json` | 로지큐브 · 창고관리(WMS) SaaS | 한울메디도매 (제약 도매 유통사) | open | 3 |
| `s013_daon_3pl_wms_lost.json` | 로지큐브 · 창고관리(WMS) SaaS | 다온로지스틱스 (3PL 물류 대행사) | lost | 3 |
| `s014_saehan_life_lms_perms.json` | 에듀링크 · 기업 교육 LMS·온라인 교육 콘텐츠 | 새한라이프 (중견 생명보험사 영업교육팀) | open | 3 |
| `s015_maru_coffee_franchise_lms.json` | 에듀링크 · 기업 교육 LMS·온라인 교육 콘텐츠 | 마루커피 (프랜차이즈 커피 본사) | open | 3 |
| `s016_hanmoe_univ_lifelong_lms.json` | 에듀링크 · 기업 교육 LMS·온라인 교육 콘텐츠 | 한뫼대학교 평생교육원 (지방 국립대 평생교육원) | open | 4 |
| `s017_dabok_hansik_buffet_dishwasher.json` | 클린웨이브 · 업소용 식기세척기 렌탈·관리 | 다복한상 (대형 한식 뷔페) | open | 3 |
| `s018_saeul_foodservice_hospital_cafeteria.json` | 클린웨이브 · 업소용 식기세척기 렌탈·관리 | 새울푸드서비스 (대학병원 직원식당 위탁 급식사) | open | 3 |
| `s019_lumiere_hotel_banquet_dishwasher.json` | 클린웨이브 · 업소용 식기세척기 렌탈·관리 | 루미에르호텔 (호텔 연회장) | won | 3 |
| `s020_milbat_bakery_pos_kiosk.json` | 페이로드 · 매장 POS·키오스크·주문 앱 | 밀밭제과 (동네 빵집 3개 매장) | open | 3 |
| `s021_hanganpark_foodcourt_kiosk.json` | 페이로드 · 매장 POS·키오스크·주문 앱 (푸드코트 공용 키오스크 + 입점사 정산) | 한간파크푸드플레이스 (복합 쇼핑몰 푸드코트 운영) | open | 4 |
| `s022_chamsae_bunsik_franchise.json` | 페이로드 · 매장 POS·키오스크·주문 앱 (가맹점 일괄 도입) | 참새분식 (대학가 분식 프랜차이즈 본사) | lost | 3 |
| `s023_loopfin_remittance_pentest.json` | 시큐어원 · 보안관제(SOC)·취약점 점검 서비스 | 루프핀 (핀테크 (소액 해외송금 전자금융업)) | open | 3 |
| `s024_jinyang_techwin_soc.json` | 시큐어원 · 보안관제(SOC)·취약점 점검 서비스 | 진양테크윈 (중견 제조사 (산업용 기계 부품)) | open | 3 |
| `s025_cheongsol_hospital_soc.json` | 시큐어원 · 보안관제(SOC)·취약점 점검 서비스 (원내 설치형 로그 수집 + 접근기록 모니터링) | 청솔종합병원 (지역 종합병원) | open | 3 |
| `s026_bukguksung_coldstorage_ppa.json` | 썬브릿지에너지 · 공장·창고 지붕 태양광 설치 (PPA 포함) | 북극성냉장 (냉동창고 운영) | won | 4 |
| `s027_dongjin_metal_rooftop_solar.json` | 썬브릿지에너지 · 공장·창고 지붕 태양광 설치 (EPC·PPA) | 동진금속 (자동차 부품 2차 협력사 (프레스·금형)) | open | 3 |
| `s028_deulsaem_rpc_rooftop.json` | 썬브릿지에너지 · 공장·창고 지붕 태양광 설치 (지붕 임대형 PPA) | 들샘농협 RPC (지역 농협 미곡종합처리장) | open | 3 |
| `s029_dalbit_retail_hr_pilot.json` | 스마트HR · 급여·근태 관리 SaaS | 달빛리테일 (프랜차이즈 편의점 가맹 본부 (직영점 운영)) | open | 3 |
| `s030_daon_construction_hr_lost.json` | 스마트HR · 급여·근태 관리 SaaS (정규직 + 일용직 공수) | 다온건설 (중견 건설사) | lost | 3 |
| `s031_hanul_nursing_hospital_hr.json` | 스마트HR · 급여·근태 관리 SaaS | 한울요양병원 (요양병원) | open | 3 |
| `s032_haetsal_care_home_fall_sensor.json` | 케어센스 · 요양시설 낙상·배회 감지 센서 | 햇살요양원 (사립 노인요양시설) | open | 3 |
| `s033_daon_senior_welfare_center_public.json` | 케어센스 · 요양시설 낙상·배회 감지 센서 | 다온시립노인복지관 (지자체 위탁 노인복지관) | open | 3 |
| `s034_neulbom_care_group_won_permissions.json` | 케어센스 · 요양시설 낙상·배회 감지 센서 | 늘봄케어그룹 (요양원 체인) | won | 4 |
| `s035_hongryong_chinese_chain_pricing.json` | 푸드체인 · 외식업 식자재 B2B 정기 납품 | 홍룡반점 (중식당 체인) | open | 3 |
| `s036_matium_food_service_handover.json` | 푸드체인 · 외식업 식자재 B2B 정기 납품 | 맛이음푸드서비스 (기업 구내식당 위탁 급식사) | open | 3 |
| `s037_burger_gongbang_lost.json` | 푸드체인 · 외식업 식자재 B2B 정기 납품 | 버거공방 (수제버거 매장) | lost | 3 |
| `s038_daesung_car_center_dialect.json` | 오토파츠원 · 정비소용 자동차 부품 공급·재고 앱 | 대성카센터 (동네 자동차 정비소) | open | 3 |
| `s039_euromotors_service_chain_pricing.json` | 오토파츠원 · 정비소용 자동차 부품 공급·재고 앱 | 유로모터스서비스 (수입차 전문 정비 체인) | open | 3 |
| `s040_hangang_taxi_garage_won.json` | 오토파츠원 · 정비소용 자동차 부품 공급·재고 앱 | 한강교통 (법인택시 회사) | won | 3 |
| `s041_modubaeum_edtech_cloud.json` | 클라우드메이트 · 클라우드 이전·운영(MSP) | 모두배움 (온라인 교육 플랫폼) | open | 3 |
| `s042_hanmaru_retail_cloud.json` | 클라우드메이트 · 클라우드 이전·운영(MSP) | 한마루유통 (중견 유통사) | open | 4 |
| `s043_mirae_policy_institute_bid.json` | 클라우드메이트 · 클라우드 이전·운영(MSP) | 미래정책연구원 (공공기관 산하 연구원) | lost | 3 |
| `s044_cheongha_binger_festival.json` | 프린트랩 · 기업 판촉물·인쇄물 제작 | 청하군 빙어축제추진위원회 (지자체 축제 추진위원회) | open | 3 |
| `s045_daon_pharma_booth_goods.json` | 프린트랩 · 기업 판촉물·인쇄물 제작 | 다온제약 (제약사 마케팅팀) | open | 3 |
| `s046_bithouse_startup_5th.json` | 프린트랩 · 기업 판촉물·인쇄물 제작 | 비트하우스 (스타트업(창립 5주년 행사)) | won | 2 |
| `s047_haetsal_homecare_rental.json` | 모빌리티원 · 법인 차량 장기렌트·차량관제 | 햇살재가간호 (방문 간호 서비스) | open | 3 |
| `s048_daedong_building_materials.json` | 모빌리티원 · 법인 차량 장기렌트·차량관제 | 대동자재상사 (지역 건설 자재 도매상) | open | 3 |
| `s049_seon_medipharm_fleet.json` | 모빌리티원 · 법인 차량 장기렌트·차량관제 | 세온메디팜 (제약 영업 조직) | open | 3 |
| `s050_saesol_apartment_pest.json` | 위생지킴이 · 방역·해충 관리 정기 서비스 | 새솔마을 3단지 관리사무소 (대형 아파트 단지) | open | 3 |
| `s051_cheongsol_foods_pest.json` | 위생지킴이 · 방역·해충 관리 정기 서비스 | 청솔푸드 (식품 제조) | open | 3 |
| `s052_basakmaru_chicken_pest_lost.json` | 위생지킴이 · 방역·해충 관리 정기 서비스 | 바삭마루치킨 (프랜차이즈 치킨 본사) | lost | 3 |
| `s053_daeun_construction_legal.json` | 리걸노트 · 계약서 검토 AI SaaS | 대운건설 (중견 건설사) | open | 3 |
| `s054_hanul_realty_franchise.json` | 리걸노트 · 계약서 검토 AI SaaS | 한울공인중개 (부동산 중개 프랜차이즈) | open | 3 |
| `s055_eden_law_firm_contract_ai.json` | 리걸노트 · 계약서 검토 AI SaaS | 법무법인 이든 (로펌) | won | 3 |
| `s056_haneon_card_callbot.json` | 보이스온 · AI 콜봇·상담 자동화 | 한온카드 (카드사 콜센터) | open | 4 |
| `s057_tuntun_vet_chain_callbot.json` | 보이스온 · AI 콜봇·상담 자동화 | 튼튼동물의료센터 (동물병원 체인) | open | 3 |
| `s058_donghae_city_gas_callbot.json` | 보이스온 · AI 콜봇·상담 자동화 | 동해도시가스 (지역 도시가스 공급) | won | 3 |
| `s059_cheongun_univ_dorm_water.json` | 아쿠아텍 · 건물 정수·급수 설비 설치·관리 | 청운대학교 생활관 (국립대 기숙사) | open | 3 |
| `s060_geunryeok_fitness_shower_lost.json` | 아쿠아텍 · 건물 정수·급수 설비 설치·관리 | 근력공장 (대형 피트니스 센터 체인) | lost | 3 |
| `s061_daon_pm_water_supply.json` | 아쿠아텍 · 건물 정수·급수 설비 설치·관리 | 다온빌딩관리 (오피스 빌딩 관리) | won | 3 |
| `s062_seirin_cosmetic_vision_inspection.json` | 비전로보틱스 · 협동로봇·비전 검사 | 세이린코스메틱 (화장품 OEM 제조) | open | 3 |
| `s063_daesung_mold_cobot.json` | 비전로보틱스 · 협동로봇·비전 검사 | 대성정밀금형 (금형 제작) | open | 3 |
| `s064_cheongwoo_eng_battery_vision.json` | 비전로보틱스 · 협동로봇·비전 검사 | 청우이엔지 (2차전지 조립 장비 제조) | open | 4 |
| `s065_pureunsoop_dental_ads_lost.json` | 마켓부스터 · 온라인 광고 대행·퍼포먼스 마케팅 | 푸른숲치과의원 (치과 의원) | lost | 3 |
| `s066_livehome_furniture_performance_ads.json` | 마켓부스터 · 온라인 광고 대행·퍼포먼스 마케팅 | 리브하임가구 (가구 제조·온라인 판매) | open | 3 |
| `s067_hanul_city_tourism_ads.json` | 마켓부스터 · 온라인 광고 대행·퍼포먼스 마케팅 | 한울시청 관광과 (지방자치단체 관광 홍보) | open | 3 |
| `s068_moasoft_it_checkup_phased.json` | 웰핏케어 · 기업 단체 건강검진·사내 건강 프로그램 | 모아소프트 (소프트웨어 개발) | open | 3 |
| `s069_eumm_fulfillment_night_checkup.json` | 웰핏케어 · 기업 단체 건강검진·사내 건강 프로그램 | 이음풀필먼트 (물류센터(풀필먼트)) | open | 3 |
| `s070_hannuri_energy_branch_checkup_won.json` | 웰핏케어 · 기업 단체 건강검진·사내 건강 프로그램 | 한누리에너지공사 중부지사 (공기업(에너지) 지사) | won | 3 |
| `s071_haneul_paper_vibration.json` | 센서플로우 · 설비 예지보전 IoT 진동 센서 | 하늘제지 (제지 공장) | open | 3 |
| `s072_daewon_injection_vibration.json` | 센서플로우 · 설비 예지보전 IoT 진동 센서 | 대원사출 (사출 성형 공장) | open | 3 |
| `s073_garam_waterworks_pump_sensor.json` | 센서플로우 · 설비 예지보전 IoT 진동 센서 | 가람시 상수도사업소 (상수도 사업소) | open | 3 |
| `s074_serim_trading_travel.json` | 트립웍스 · 기업 출장 관리·법인카드 정산 | 세림무역 (해외 출장 잦은 무역회사) | open | 3 |
| `s075_maru_consulting_expense.json` | 트립웍스 · 기업 출장 관리·법인카드 정산 | 마루컨설팅 (컨설팅 회사) | lost | 3 |
| `s076_lowell_semi_equipment_travel.json` | 트립웍스 · 기업 출장 관리·법인카드 정산 | 로웰반도체장비 (반도체 장비 회사) | open | 4 |
| `s077_haetsalgol_apple_drone.json` | 팜그로우 · 비료·농자재·드론 방제 | 햇살골사과영농조합 (사과 농가 영농조합) | open | 3 |
| `s078_solmoe_golf_turf.json` | 팜그로우 · 비료·농자재·드론 방제 | 솔뫼컨트리클럽 (골프장 잔디 관리팀) | won | 3 |
| `s079_pureundeul_tomato_organic.json` | 팜그로우 · 비료·농자재·드론 방제 | 푸른들토마토농업회사법인 (시설채소 농업회사법인) | open | 3 |
| `s080_daon_coffee_bgm.json` | 사운드라인 · 매장 배경음악 구독·음향 설비 | 다온커피 (카페 체인 본사) | open | 3 |
| `s081_powerzone_gym_music.json` | 사운드라인 · 매장 배경음악 구독·음향 설비 | 파워존 피트니스 (대형 헬스장) | lost | 3 |
| `s082_hanul_department_food_hall.json` | 사운드라인 · 매장 배경음악 구독·음향 설비 | 한울백화점 본점 (백화점 식품관) | open | 3 |
| `s083_haesol_daycare_federation_app.json` | 키즈노트북 · 어린이집·유치원 관리 앱 | 해솔구 어린이집연합회 (구립 어린이집 연합) | open | 3 |
| `s084_byeolbit_kindergarten_app.json` | 키즈노트북 · 어린이집·유치원 관리 앱 | 별빛유치원 (사립 유치원) | open | 3 |
| `s085_todaktodak_kidsworld_app.json` | 키즈노트북 · 어린이집·유치원 관리 앱 | 토닥토닥 키즈월드 (키즈카페 겸 놀이학교) | lost | 3 |
| `s086_cheonghae_hospital_fm.json` | 빌딩케어플러스 · 빌딩 시설관리(FM) 아웃소싱 | 청해병원 (지방 중소 병원) | open | 3 |
| `s087_gasan_hanbit_knowledge_center.json` | 빌딩케어플러스 · 빌딩 시설관리(FM) 아웃소싱 | 가산한빛 지식산업센터 관리단 (지식산업센터 관리단) | open | 4 |
| `s088_hangyeol_university_facility_won.json` | 빌딩케어플러스 · 빌딩 시설관리(FM) 아웃소싱 | 한결대학교 (대학 캠퍼스 시설과(국립대)) | won | 3 |
| `s089_roadeye_mobility_labeling.json` | 라벨웍스 · 데이터 라벨링 외주 | 로드아이 모빌리티 (자율주행 스타트업) | open | 3 |
| `s090_modumarket_search_labeling.json` | 라벨웍스 · 데이터 라벨링 외주 | 모두마켓 (대형 이커머스 검색팀) | open | 3 |
| `s091_medivision_lab_xray_labeling.json` | 라벨웍스 · 데이터 라벨링 외주 | 메디비전랩 (의료 AI (흉부 영상 진단 보조)) | open | 3 |
| `s092_purun_internal_clinic_emr.json` | 메디차트 · 의원급 전자차트(EMR) | 푸른내과의원 (동네 내과 의원) | won | 3 |
| `s093_teunteun_ortho_emr.json` | 메디차트 · 의원급 전자차트(EMR) | 튼튼정형외과의원 (정형외과 의원 (원장 2명)) | open | 3 |
| `s094_haeon_hanbang_chain_emr_lost.json` | 메디차트 · 의원급 전자차트(EMR) | 해온한방네트워크 (한의원 체인) | lost | 3 |
| `s095_saebyeok_road_logistics_uniform.json` | 유니폼하우스 · 기업 유니폼·작업복 제작 | 새벽로드물류 (물류 회사 (현장직 유니폼)) | open | 3 |
| `s096_yeoul_hotel_resort_uniform.json` | 유니폼하우스 · 기업 유니폼·작업복 제작 | 여울호텔앤리조트 (호텔 체인 (객실·식음팀)) | open | 3 |
| `s097_cheongsol_city_cleaners_uniform.json` | 유니폼하우스 · 기업 유니폼·작업복 제작 | 청솔시청 청소행정과 (지자체 (환경미화원 근무복)) | open | 3 |
| `s098_pado_seafood_wholesale_coldchain.json` | 프레시로드 · 냉장·냉동 화물 운송 | 파도수산유통 (수산물 도매 법인) | open | 3 |
| `s099_hankki_baguni_mealkit_coldchain.json` | 프레시로드 · 냉장·냉동 화물 운송 | 한끼바구니 (밀키트 스타트업) | open | 3 |
| `s100_eden_bio_pharma_vaccine_logistics.json` | 프레시로드 · 냉장·냉동 화물 운송 | 이든바이오제약 (제약사 (백신 물류)) | won | 4 |

사용자(`users`)는 시나리오 전체에서 같은 의미입니다: user 1·2는 1팀(플로우웍스), user 3은 2팀(그린팩), user 4는 3팀(데이터브릿지), user 5~8은 각각 4~7팀(메디온, 시빅웍스, 세이프온, 팜링크). S011~S100은 판매 팀 8~37팀(30개 판매사)이 고객사 3곳씩 맡습니다. 팀 n의 영업 담당자는 user n+1, 같은 팀 동료는 user n+100입니다.

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
  paraphrase_of     같은 질문을 대화에 없는 단어로 바꿔 말한 변형이면 원래 질문 id
                    (정답·질문자·권한은 원래 질문과 같음. 의미 검색 질문마다 2개씩)
  answer            기대 답변 요약
```

id는 시나리오 접두어를 붙여 전체에서 겹치지 않게 합니다: 고객사 `S001-CO1`, 담당자 `S001-C2`, 대화 `S001-I2`, 발화 `S001-I2-U08`(시나리오 1, 대화 2, 발화 8), 정답 정보 `S001-F11`, 질문 `S001-Q01`. 발화 시간은 글자 수로 추정한 값이라 실제 녹음과 다릅니다. 이 대본을 녹음해 STT·화자 분리를 평가할 때는 녹음 결과의 시간을 기준으로 합니다.
