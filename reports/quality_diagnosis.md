# RAG Quality Diagnosis (Solar LLM-as-judge)

> 측정일: 2026-04-27T21:43:59+09:00  
> Judge model: `solar-pro3`  
> 평가 traces: **250** (OK=134, ERR=0)

---

## 1. Quality Score 분포

| Score | Count | % | Bar |
|-------|-------|---|-----|
| 5 | 71 | 28.4% | `██████████████` |
| 4 | 80 | 32.0% | `████████████████` |
| 3 | 33 | 13.2% | `███████` |
| 2 | 4 | 1.6% | `█` |
| 1 | 62 | 24.8% | `████████████` |
| ERR | 0 | 0.0% | `` |

**평균 quality_score**: 3.38 / 5.00 (ERR 제외)

---

## 2. 실패 유형별 분포

| Mode | n | % | 설명 |
|------|---|---|------|
| **OK** | 134 | 53.6% | 만족 (quality_score ≥ 4) |
| **A** | 25 | 10.0% | Retrieval 실패 — 정답 doc 미검색 |
| **B** | 0 | 0.0% | Groundedness 과거부 — 검색 OK인데 judge가 거부 |
| **C** | 33 | 13.2% | Context noise — 잡음 doc이 답변을 흐림 |
| **D** | 0 | 0.0% | Generation 부실 — 질문 핵심 비켜감 (relevance ↓) |
| **E** | 58 | 23.2% | Fallback 과반응 — 부적절한 거부 |
| **ERR** | 0 | 0.0% | Judge 호출 실패 |

---

## 3. Challenge Type 별 성능

| Challenge | n | OK rate | avg score | mode 분포 |
|-----------|---|---------|-----------|------------|
| T1_conversational | 50 | 64.0% | 3.62 | A:4, C:4, E:10, OK:32 |
| T2_vague | 50 | 44.0% | 3.24 | A:7, C:12, E:9, OK:22 |
| T3_paraphrase | 50 | 78.0% | 4.12 | A:1, C:4, E:6, OK:39 |
| T4_multi_intent | 50 | 30.0% | 2.64 | A:7, C:9, E:19, OK:15 |
| T5_inference | 50 | 52.0% | 3.26 | A:6, C:4, E:14, OK:26 |

---

## 4. 실패 유형별 대표 사례 (각 5건)

### Mode A — Retrieval 실패 — 정답 doc 미검색

**권장 fix 방향**: router CAMPUS_PATTERN/synonym 보강, hybrid_cc_weight 재튜닝, query expansion 도입 검토

- **Q** (T1_conversational, score=4): 시험 서술형 어떻게 준비해야 돼?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_3_c8'] / 검색 top3: ['lec_lecture_reviews_2_c23', 'si_static_info_280', 'lec_lecture_reviews_103_c1']
  - expected: `시험 범위가 넓으니 미리 공부하고 PPT에 강조된 부분을 중심으로 암기하는 게 좋아. 키워드는 꼭 놓치지 말고 최대한 길게 작성해 봐.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
시험 서술형을 준비하기 위해서는 교수님이 미리 알려주는 시험 범위를 철저히 암기하고, 가능한 많은 내용을 적을 수 있도록 준비해야 합니다. 특히 서술형 문제는 암기해야 할 양이 많아 난이도가 높은 편이므로, 찝어준 내용을 꼼꼼히 복습하는 것이 중요합니다.

[출처: lec_lecture_reviews_2_c23, 강의평가, 전`
  - diagnosis: 검색된 출처가 정답 doc(lec_lecture_reviews_3_c8)을 포함하지 않아 retrieval_hit가 false이며, 시스템 답변이 정답과 유사하지만 핵심 키워드 강조와 PPT 중심 암기 내용이 누락되었습니다.
  - fix: router에 정답 doc_id(lec_lecture_reviews_3_c8)를 포함하도록 검색 쿼리를 개선하거나, 강의평가 카테고리에서 해당 강의 리뷰 문서를 우선적으로 검색하도록 설정 변경

- **Q** (T1_conversational, score=2): 수강신청 언제부터?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['학칙_학칙_대학_77'] / 검색 top3: ['si_static_info_267', 'si_static_info_265', 'cal_calendar_67']
  - expected: `수업시간표는 학기가 시작되기 4주 전에 총장이 정한다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
수강신청은 2026학년도 1학기 기준으로 2026-02-25(수)부터 2026-02-27(금)까지 진행됩니다.  
신·편입생과 재학생 모두 해당 기간에 수강신청을 할 수 있습니다.

[출처: cal_calendar_7, 학사일정, 성남]`
  - diagnosis: retrieval_hit=false로 정답 doc(학칙_학칙_대학_77)을 검색하지 못해 수강신청 시작일을 잘못 답변함
  - fix: router에 학칙 관련 doc_id(학칙_학칙_대학_77)를 포함하도록 패턴 추가

- **Q** (T1_conversational, score=4): 교수님 발표 많이 시키나요?
  - verdict=notSure retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_23_c0'] / 검색 top3: ['lec_lecture_reviews_92_c0', 'lec_lecture_reviews_38_c2', 'lec_lecture_reviews_38_c0']
  - expected: `네, 교수님은 학생 발표와 참여를 중시하는 방식으로 수업합니다. 매 수업 후 퀴즈 및 회화 테스트도 포함되어 있습니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
제공된 강의평가 자료를 바탕으로 "교수님 발표 많이 시키나요?"라는 질문에 답변드리겠습니다.

**답변:**
네, 여러 강의에서 교수님이 발표를 많이 시키는 것으로 나타납니다.

1. **세상과소통하는기초한자** (문서 1):  
   - "학생들에게 질문을 많이 하시며, 발표와 참여를 강조하는 수업입니다."
   - "PPT`
  - diagnosis: retrieval_hit가 false로, 정답 doc(lec_lecture_reviews_23_c0)을 검색하지 못했습니다. 대신 다른 강의평가 문서들을 인용하여 답변을 생성했으나, 정답에 포함된 '매 수업 후 퀴즈 및 회화 테스트' 정보가 누락되었습니다.
  - fix: router에 lec_lecture_reviews_23_c0을 포함하는 검색 패턴을 추가하고, 정답 doc이 검색되지 않을 경우 fallback 대신 해당 doc을 우선 검색하도록 개선하세요.

- **Q** (T1_conversational, score=4): 이 강의 지루해요?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_51_c6'] / 검색 top3: ['lec_lecture_reviews_32_c21', 'lec_lecture_reviews_70_c10', 'lec_lecture_reviews_12_c17']
  - expected: `네, 지루하다는 평가가 많아요. 암기량도 많고 철학적인 내용이 많아 이과 학생들에겐 어려울 수 있어요.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
제공된 강의평가 자료를 종합해 보면, 여러 강의에서 "지루한 강의"라는 의견이 공통적으로 언급되고 있습니다.

- 자연과학의이해 강의에서는 "PPT를 읽는 방식이 주를 이룹니다"라고 명시되어 있습니다.
- EU인문과학콘서트 강의에서는 "강의 내용이 지루하다고 느낍니다"라고 언급되었습니다.
- 한시의이해 강의에서는 "수업이 지루`
  - diagnosis: retrieval_hit가 false로 설정되어 정답 doc(lec_lecture_reviews_51_c6)을 검색하지 못했으며, 대신 다른 강의평가 문서들을 인용하여 답변을 생성했습니다.
  - fix: router의 검색 로직에서 정답 doc_id를 정확히 매칭하도록 개선하거나, 검색 결과에 정답 doc이 없을 경우 fallback 대신 해당 doc을 강제로 포함하도록 수정

- **Q** (T2_vague, score=1): 발표 준비는 어떻게 하면 되나요?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_78_c9'] / 검색 top3: ['si_static_info_280', 'si_static_info_203', 'lec_lecture_reviews_31_c8']
  - expected: `2인 1조 발표는 부담스럽지 않으므로, 간단하게 준비하되 팀원과의 협업을 잘하는 것이 중요합니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
[답변]
강의실 대여는 학과별로 관리되는 강의실이 있으므로, 대여하고자 하는 강의실의 담당학과를 확인한 후 해당 학과사무실에 연락하여 신청하시면 됩니다.

[출처: si_static_info_280, 시설대여, 성남]`
  - diagnosis: 검색된 출처가 강의실 대여 관련 시설 정보만 포함하고 있어 발표 준비 방법에 대한 정답 doc(lec_lecture_reviews_78_c9)을 놓쳤습니다.
  - fix: router에 발표 준비 관련 강의 평가 문서를 검색하도록 CAMPUS_PATTERN을 추가하고, 강의평가 카테고리에서 발표 관련 키워드를 포함하도록 개선하세요.

### Mode C — Context noise — 잡음 doc이 답변을 흐림

**권장 fix 방향**: passage_reranker 도입(GPU 가용 시), top_k_final 축소, 컨텍스트 정렬 prompt 추가

- **Q** (T1_conversational, score=3): 1학기 수업 언제 시작해요?
  - verdict=grounded retry=False retrieval_hit=True
  - 정답 doc: ['cal_calendar_22'] / 검색 top3: ['si_static_info_177', 'cal_calendar_11', 'cal_calendar_22']
  - expected: `1학기 수업개시일은 2026년 6월 15일 월요일입니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
1학기 수업은 2026년 3월 29일(일)에 시작합니다.

[출처: si_static_info_177, FAQ, 전체]`
  - diagnosis: 정답 doc(cal_calendar_22)은 검색되었으나, 시스템 답변은 다른 출처(si_static_info_177)의 정보를 인용하여 2026년 3월 29일이라는 잘못된 날짜를 제시했습니다.
  - fix: 검색된 sources 중 정답 doc(cal_calendar_22)을 우선적으로 사용하도록 answer prompt에 '정답 doc_id가 있으면 해당 정보를 우선 인용하라'는 규칙을 추가하세요.

- **Q** (T1_conversational, score=3): 시험 범위는 어떻게 정해지나요?
  - verdict=grounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_108_c1'] / 검색 top3: ['lec_lecture_reviews_100_c3', '학칙_학칙_대학_16', 'lec_lecture_reviews_86_c0']
  - expected: `시험은 수업 중 교수님이 중요하다고 언급한 부분과 연습문제에서 주로 출제됩니다. 교수님이 미리 시험 문제를 언급해 주시기도 합니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
시험 범위는 강의별로 다르게 정해집니다.  
- **수학** 강의의 경우, 시험 범위는 강의 내용에 기반하지만 구체적인 범위 공지가 부족하다는 의견이 있습니다.  
- **걸어서역사속으로** 강의의 경우, 시험 범위는 시험 1~2주 전에 PPT를 통해 공지됩니다.  
- **K-컨텐츠새로읽기** 강의의 경우, 시험 범위는 시험`
  - diagnosis: 정답 doc(lec_lecture_reviews_108_c1)이 검색되지 않아 핵심 정보가 누락되었고, 검색된 강의평가 문서들이 시험 범위 결정 방식에 대한 직접적인 답변을 제공하지 않아 답변이 흐려졌습니다.
  - fix: router에 시험 범위 관련 키워드(예: '시험 범위', '시험 문제')를 추가하고, 정답 doc을 명시적으로 포함하도록 검색 전략을 개선하세요.

- **Q** (T1_conversational, score=3): 팀플 발표 팁 알려줘
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_22_c14'] / 검색 top3: ['lec_lecture_reviews_90_c2', 'lec_lecture_reviews_44_c10', 'lec_lecture_reviews_1_c9']
  - expected: `발표 내용을 철저히 준비하고, 토론 쟁점을 잘 숙지하세요. 팀원들과 적극적으로 협력하는 게 중요합니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
팀플 발표 팁은 다음과 같습니다:

1. **팀원과의 원활한 소통**: 팀원들과의 소통을 통해 프로젝트를 성공적으로 마무리하는 것이 중요합니다.
2. **역할 분담 명확화**: 팀 프로젝트에서의 역할 분담을 명확히 하여 효율적으로 진행할 수 있습니다.
3. **참신한 아이디어 제안**: 참신한 아이디어를 제안하여 프로젝트의 질`
  - diagnosis: 정답 doc(lec_lecture_reviews_22_c14)이 검색되지 않았고, 검색된 sources(lec_lecture_reviews_38_c8 등)는 발표 팁과 관련된 내용이 포함되어 있으나 정답의 핵심인 '토론 쟁점 숙지'와 '팀원과의 적극적 협력'이 누락되어 있습니다.
  - fix: router에 팀플 발표 관련 키워드(예: '토론 쟁점', '팀원 협력')를 추가하고, 정답 doc_id를 명시적으로 매핑하여 검색 정확도를 높이세요.

- **Q** (T1_conversational, score=3): 기말고사 대체 과제 있나요?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_59_c1'] / 검색 top3: ['lec_lecture_reviews_48_c3', 'lec_lecture_reviews_106_c1', 'lec_lecture_reviews_66_c8']
  - expected: `일부 학기에는 기말고사를 대체하는 과제가 주어지기도 했습니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
기말고사 대체 과제 여부는 강의마다 다릅니다.

- **유럽지리와문화**: 기말고사는 반드시 치러야 합니다. (중간고사는 과제로 대체)
- **미디어크리에이터**: 기말고사는 서술형으로 진행되며, 중간고사는 과제로 대체됩니다.
- **호신술**: 기말고사는 실기 시험으로 대체되지 않습니다.
- **인류식문화**: 기말고사는 별`
  - diagnosis: 정답 doc(lec_lecture_reviews_59_c1)은 기말고사 대체 과제 존재 여부를 일반적으로 언급했으나, 시스템은 검색된 강의평가 문서들을 인용하여 특정 강의별 기말고사 대체 여부를 나열함으로써 핵심 정보를 흐리게 했습니다.
  - fix: router에 정답 doc을 우선 검색하도록 강의평가 문서 필터링 로직을 개선하고, answer prompt에 '일부 학기에는 기말고사를 대체하는 과제가 주어지기도 했습니다'라는 핵심 문장을 직접 포함하도록 지시하세요.

- **Q** (T2_vague, score=3): 디자인 수업에서 좋은 평가를 받으려면 어떻게 해야?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_53_c11'] / 검색 top3: ['lec_lecture_reviews_59_c7', 'lec_lecture_reviews_59_c4', 'lec_lecture_reviews_53_c10']
  - expected: `디자인 수업에서 A+를 받기 위해서는 시험 전 제공되는 용어 정리 노트를 철저히 암기하는 것이 중요합니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
디자인 수업에서 좋은 평가를 받으려면 다음과 같이 준비해야 합니다:

1. **수업 중 필기 철저히**: 수업 내용을 꼼꼼히 기록하고, 사진을 찍어 자료를 정리해 두는 것이 중요합니다.  
2. **사례 암기**: 서술형 문제에 대비하여 사례를 잘 기억하고, 필기를 정리하여 암기하는 것이 필요합니다.  
3. **시험 준비 철`
  - diagnosis: 정답 doc(lec_lecture_reviews_53_c11)이 검색 결과에 포함되지 않았고, 제공된 출처(lec_lecture_reviews_59_c4, lec_lecture_reviews_59_c8)는 시험 전 용어 정리 노트 암기에 대한 구체적인 언급이 없어 핵심 정보가 누락되었습니다.
  - fix: router에 lec_lecture_reviews_53_c11을 검색 대상으로 추가하고, answer prompt에 '시험 전 제공되는 용어 정리 노트 암기'를 명시적으로 강조하여 관련 출처를 우선적으로 추출하도록 개선하세요.

### Mode E — Fallback 과반응 — 부적절한 거부

**권장 fix 방향**: fallback 트리거 임계값 재튜닝, two-stage retry (top-10 확장 후 재시도) 도입

- **Q** (T1_conversational, score=1): 결석 3번 넘으면 어떻게 돼?
  - verdict=notGrounded retry=True retrieval_hit=True
  - 정답 doc: ['학칙_학칙_대학_295'] / 검색 top3: ['학칙_학칙_대학_319', '학칙_학칙_대학_69', 'reg_regulations_4']
  - expected: `3회(15주 기준 3주 이상) 결석하면 담당자가 학과장에게 보고하고, 학과장이 교무혁신처에 명단을 제출한다.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: 학칙_학칙_대학_319, 학칙, 전체]`
  - diagnosis: 정답 doc_id(학칙_학칙_대학_295)가 sources에 포함되어 있음에도 불구하고 시스템이 "제공된 자료에서 해당 정보를 찾을 수 없습니다"라고 응답하여 정답을 완전히 놓쳤습니다.
  - fix: router에 학칙_학칙_대학_295를 정답으로 명시적으로 매핑하고, answer prompt에 해당 doc_id가 포함된 경우 반드시 정보를 추출하도록 강제하세요.

- **Q** (T1_conversational, score=1): 조별 발표 안 할 땐 뭐 해?
  - verdict=notGrounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_22_c2'] / 검색 top3: ['lec_lecture_reviews_38_c2', 'lec_lecture_reviews_84_c2', 'lec_lecture_reviews_64_c2']
  - expected: `발표 안 한 조는 찬반 토론이나 생각 정리로 수업에 참여해요. 수업 시간은 유연하게 관리돼서 일찍 끝나는 경우가 많아요.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: lec_lecture_reviews_38_c2, 강의평가, 전체]`
  - diagnosis: 검색된 출처가 모두 강의평가 카테고리이며 정답 doc_id(lec_lecture_reviews_22_c2)가 포함되지 않아 정보가 전혀 제공되지 않았습니다.
  - fix: router에 정답 doc_id가 포함된 강의 자료 카테고리를 우선적으로 검색하도록 패턴을 추가하고, fallback 조건을 완화하여 정답이 있는 경우 '정보 없음' 응답을 방지하세요.

- **Q** (T1_conversational, score=1): 이 과목 학점 경쟁 어때?
  - verdict=notGrounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_12_c12'] / 검색 top3: ['lec_lecture_reviews_106_c4', 'lec_lecture_reviews_100_c6', 'lec_lecture_reviews_5_c7']
  - expected: `학점 경쟁은 비교적 치열하지 않지만 과제와 시험 준비가 중요하다고 되어 있습니다.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: lec_lecture_reviews_106_c4, 강의평가, 전체]`
  - diagnosis: 검색된 sources에 정답 doc_id(lec_lecture_reviews_12_c12)가 포함되지 않았고, 시스템은 '정보 없음'으로 fallback했으나 corpus에 정답이 존재하므로 부적절한 거부입니다.
  - fix: router에 해당 과목 강의평가 doc_id를 정확히 매핑하거나, 검색 쿼리에 과목명을 명시적으로 포함시켜 retrieval_hit를 보장해야 합니다.

- **Q** (T1_conversational, score=1): 이 수업 누구에게 추천?
  - verdict=notGrounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_7_c7'] / 검색 top3: ['lec_lecture_reviews_74_c0', 'lec_lecture_reviews_74_c8', 'lec_lecture_reviews_7_c9']
  - expected: `중국사에 관심 있고 자기주도 학습 가능한 학생에게 추천됩니다.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: lec_lecture_reviews_74_c0, 강의평가, 전체]`
  - diagnosis: 검색된 sources에 정답 doc(lec_lecture_reviews_7_c7)이 포함되어 있음에도 불구하고 retrieval_hit=False로 판단되어 notGrounded로 거부되었으며, 시스템 답변은 '정보 없음'이라고 잘못 응답했습니다.
  - fix: retrieval_hit 판단 로직을 개선하여 정답 doc_id가 sources에 존재하면 반드시 retrieval_hit=True로 설정하도록 수정하고, fallback 응답을 방지하세요.

- **Q** (T1_conversational, score=1): 이 수업 지루해요?
  - verdict=notSure retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_2_c21'] / 검색 top3: ['lec_lecture_reviews_57_c0', 'lec_lecture_reviews_102_c0', 'lec_lecture_reviews_28_c1']
  - expected: `지루함을 견디기 어려운 학생에게는 비추천합니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
제공된 강의평가 자료를 바탕으로 "이 수업 지루해요?"라는 질문에 답변드리겠습니다.

강의평가 문서들을 살펴보면, 각 강의의 지루함 여부에 대한 평가가 다음과 같이 기록되어 있습니다:

1. **글로벌시대스포츠이벤트의이해** (문서 1): "학생들이 수업을 지루해하지 않는다는 점에서 시간 관리를 잘 하고 있는 것으로 추측됩니다`
  - diagnosis: 시스템이 정답 doc(lec_lecture_reviews_2_c21)을 검색하지 못하고, 제공된 sources에 정답 정보가 없어 fallback으로 '정보 없음'을 반환했으나 corpus에 정답이 존재하므로 부적절 거부됨
  - fix: router에 lec_lecture_reviews_2_c21 doc_id를 포함한 강의평가 검색 패턴 추가 및 fallback 조건 완화

---

## 5. 빈도 기반 Top Fix 제안 (Solar 자동 추출)

| 빈도 | 제안 |
|------|------|
| 36 | 없음 |
| 8 | 현재 시스템이 이미 잘 작동하고 있으므로 특별한 수정이 필요하지 않습니다. |
| 4 | 현재 답변이 이미 정확하므로 추가 수정이 필요하지 않습니다. |
| 4 | 현재 시스템이 이미 정확한 답변을 제공하고 있으므로 추가 수정이 필요하지 않습니다. |
| 3 | 현재 답변이 이미 완벽하므로 추가 수정이 필요 없습니다. |
| 3 | 현재 답변이 이미 완벽하므로 추가 수정이 필요하지 않습니다. |
| 2 | 현재 시스템이 이미 잘 작동하고 있으므로 추가 수정이 필요하지 않습니다. |
| 1 | 현재 답변은 이미 완벽하므로 추가 수정이 필요 없습니다. |
| 1 | router에 학칙_학칙_대학_295를 정답으로 명시적으로 매핑하고, answer prompt에 해당 doc_id가 포함된 경우 반드시 정보를 추출하도록 강제하세요. |
| 1 | router에 정답 doc_id가 포함된 강의 자료 카테고리를 우선적으로 검색하도록 패턴을 추가하고, fallback 조건을 완화하여 정답이 있는 경우 '정보 없음' 응답을 방지하세요. |

---

## 6. 산출물

- `reports/quality_diagnosis.md` — 이 보고서
- `reports/quality_diagnosis.json` — 기계 판독용 통계
- `reports/quality_judgments.jsonl` — row-level 판정
- `logs/quality_traces.jsonl` — row-level trace
- `data/qa_adversarial.parquet` — adversarial 평가셋
- `data/qa_bad_cases.parquet` — quality_score < 3 행 (회귀 테스트용)