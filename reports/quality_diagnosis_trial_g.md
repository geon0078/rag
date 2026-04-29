# RAG Quality Diagnosis (Solar LLM-as-judge)

> 측정일: 2026-04-27T22:14:26+09:00  
> Judge model: `solar-pro3`  
> 평가 traces: **250** (OK=151, ERR=0)

---

## 1. Quality Score 분포

| Score | Count | % | Bar |
|-------|-------|---|-----|
| 5 | 68 | 27.2% | `██████████████` |
| 4 | 98 | 39.2% | `████████████████████` |
| 3 | 40 | 16.0% | `████████` |
| 2 | 6 | 2.4% | `█` |
| 1 | 38 | 15.2% | `████████` |
| ERR | 0 | 0.0% | `` |

**평균 quality_score**: 3.61 / 5.00 (ERR 제외)

---

## 2. 실패 유형별 분포

| Mode | n | % | 설명 |
|------|---|---|------|
| **OK** | 151 | 60.4% | 만족 (quality_score ≥ 4) |
| **A** | 26 | 10.4% | Retrieval 실패 — 정답 doc 미검색 |
| **B** | 0 | 0.0% | Groundedness 과거부 — 검색 OK인데 judge가 거부 |
| **C** | 40 | 16.0% | Context noise — 잡음 doc이 답변을 흐림 |
| **D** | 0 | 0.0% | Generation 부실 — 질문 핵심 비켜감 (relevance ↓) |
| **E** | 33 | 13.2% | Fallback 과반응 — 부적절한 거부 |
| **ERR** | 0 | 0.0% | Judge 호출 실패 |

---

## 3. Challenge Type 별 성능

| Challenge | n | OK rate | avg score | mode 분포 |
|-----------|---|---------|-----------|------------|
| T1_conversational | 50 | 58.0% | 3.60 | A:6, C:13, E:2, OK:29 |
| T2_vague | 50 | 56.0% | 3.68 | A:10, C:10, E:2, OK:28 |
| T3_paraphrase | 50 | 82.0% | 4.18 | A:1, C:5, E:3, OK:41 |
| T4_multi_intent | 50 | 38.0% | 2.74 | A:4, C:7, E:20, OK:19 |
| T5_inference | 50 | 68.0% | 3.84 | A:5, C:5, E:6, OK:34 |

---

## 4. 실패 유형별 대표 사례 (각 5건)

### Mode A — Retrieval 실패 — 정답 doc 미검색

**권장 fix 방향**: router CAMPUS_PATTERN/synonym 보강, hybrid_cc_weight 재튜닝, query expansion 도입 검토

- **Q** (T1_conversational, score=2): 이 과목 학점 경쟁 어때?
  - verdict=grounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_12_c12'] / 검색 top3: ['lec_lecture_reviews_45_c8', 'lec_lecture_reviews_91_c4', 'lec_lecture_reviews_75_c17']
  - expected: `학점 경쟁은 비교적 치열하지 않지만 과제와 시험 준비가 중요하다고 되어 있습니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
이 과목 학점 경쟁은 매우 치열합니다.  
생활일본어 강의는 일본어 기초가 없는 학생들에게는 다소 어려울 수 있으며, 고인물들이 많아 학점 경쟁이 치열합니다. 또한 학점은 상대적으로 깐깐한 편이며, A+를 받기 위해서는 시험에서 높은 점수를 받아야 하고 과제와 출석도 철저히 관리해야 합니다.

[출처: lec_lecture_r`
  - diagnosis: retrieval_hit=False로 정답 doc(lec_lecture_reviews_12_c12)을 검색하지 못했고, 대신 성남캠퍼스 관련 다른 강의평가 문서들을 인용하여 학점 경쟁이 '매우 치열하다'고 잘못 답변했습니다.
  - fix: router에 해당 과목명 또는 캠퍼스별 강의평가 패턴을 추가하여 정확한 doc_id(lec_lecture_reviews_12_c12)를 검색하도록 개선하세요.

- **Q** (T1_conversational, score=4): 시험 서술형 어떻게 준비해야 돼?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_3_c8'] / 검색 top3: ['lec_lecture_reviews_2_c23', 'si_static_info_280', 'lec_lecture_reviews_103_c1']
  - expected: `시험 범위가 넓으니 미리 공부하고 PPT에 강조된 부분을 중심으로 암기하는 게 좋아. 키워드는 꼭 놓치지 말고 최대한 길게 작성해 봐.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
시험 서술형은 미리 범위를 알려주기 때문에 해당 범위를 중심으로 철저히 암기하고, 가능한 많은 내용을 적을 수 있도록 준비하는 것이 좋습니다. 특히 서술형 문제는 암기해야 할 양이 많아 난이도가 높은 편이므로, 찝어준 내용을 꼼꼼히 복습하는 것이 중요합니다.

[출처: doc_id=lec_lecture_reviews_2_c23`
  - diagnosis: 검색된 출처가 정답 doc(lec_lecture_reviews_3_c8)을 포함하지 않아 retrieval_hit가 false이며, 시스템 답변은 정답과 유사하지만 핵심 키워드 강조와 'PPT에 강조된 부분' 같은 구체적 정보가 누락되었습니다.
  - fix: router에 정답 doc_id를 정확히 매핑하고, 검색 시 정답 문서 포함 여부를 확인하는 로직을 강화하세요. 또한 답변 생성 시 정답 문서의 핵심 내용을 명시적으로 인용하도록 answer prompt를 개선하세요.

- **Q** (T1_conversational, score=1): 수강신청 언제부터?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['학칙_학칙_대학_77'] / 검색 top3: ['si_static_info_267', 'si_static_info_265', 'cal_calendar_67']
  - expected: `수업시간표는 학기가 시작되기 4주 전에 총장이 정한다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
수강신청은 2026학년도 1학기 기준으로 2026-02-25(수)부터 2026-02-27(금)까지 진행됩니다.  
신·편입생과 재학생 모두 해당 기간에 수강신청을 할 수 있으며, 선착순 방식으로 진행됩니다.  
동시 접속이 불가한 경우가 많아 기기 1개로 접속하는 것을 권장합니다.

[출처: doc_id=cal_calendar`
  - diagnosis: retrieval_hit가 false이므로 정답 doc(학칙_학칙_대학_77)을 검색하지 못했고, 시스템 답변은 성남캠퍼스 학사일정만 인용하여 정답과 무관합니다.
  - fix: router에 학칙 관련 doc_id(학칙_학칙_대학_77)를 포함하도록 개선하고, 질문 키워드('수강신청 언제부터')에 대한 정확한 학칙 내용을 검색하도록 prompt를 수정하세요.

- **Q** (T1_conversational, score=1): 이 수업 누구에게 추천?
  - verdict=grounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_7_c7'] / 검색 top3: ['lec_lecture_reviews_74_c0', 'lec_lecture_reviews_24_c0', 'lec_lecture_reviews_0_c0']
  - expected: `중국사에 관심 있고 자기주도 학습 가능한 학생에게 추천됩니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
데이터분석의기초 강의는 데이터 분석에 관심이 있고 R 프로그래밍에 대한 기본적인 이해가 있는 학생들에게 적합합니다. 반면, 코딩에 전혀 관심이 없거나 컴퓨터에 익숙하지 않은 학생들에게는 비추천됩니다.

[출처: lec_lecture_reviews_74_c0, 강의평가, 전체]
[출처: lec_lecture_reviews_74_`
  - diagnosis: retrieval_hit=False로 정답 doc(lec_lecture_reviews_7_c7)을 검색하지 못했고, 시스템 답변은 데이터분석 강의에 대한 내용으로 질문과 무관합니다.
  - fix: router에 중국사 강의 관련 doc_id를 정확히 매핑하고, 질문 키워드 '중국사'를 포함한 강의 추천 패턴을 추가하세요.

- **Q** (T1_conversational, score=1): 학점 평가는 어떻게 돼?
  - verdict=grounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_18_c16'] / 검색 top3: ['학칙_학칙_대학_306', '학칙_학칙_대학_135', 'lec_lecture_reviews_8_c4']
  - expected: `이 강의는 절대평가로 진행되고 학점 부여가 비교적 후한 편이에요.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
학점은 상대평가를 원칙으로 하며 절대평가와 환산평가도 할 수 있고 성적등급의 분포는 가능한 정규분포가 이루어지도록 합니다. 다만, 성적평가비율을 조정하고자 하는 경우 구성원의 동의를 받아 교무혁신처에 제출하여 승인 후 처리합니다.

학업성적은 각 교과목별로 시험성적, 출석성적, 과제물 및 수업 참여도를 종합평가하여 100점 만`
  - diagnosis: 검색된 출처가 정답 doc(lec_lecture_reviews_18_c16)을 포함하지 않아 절대평가 및 후한 학점 부여라는 핵심 정보를 놓쳤습니다.
  - fix: router에 lec_lecture_reviews_18_c16을 검색 대상으로 추가하고, 강의평가 관련 질의 시 해당 doc_id를 우선 매칭하도록 규칙을 수정하세요.

### Mode C — Context noise — 잡음 doc이 답변을 흐림

**권장 fix 방향**: passage_reranker 도입(GPU 가용 시), top_k_final 축소, 컨텍스트 정렬 prompt 추가

- **Q** (T1_conversational, score=3): 조별 발표 안 할 땐 뭐 해?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_22_c2'] / 검색 top3: ['si_static_info_121', 'lec_lecture_reviews_53_c7', 'lec_lecture_reviews_63_c10']
  - expected: `발표 안 한 조는 찬반 토론이나 생각 정리로 수업에 참여해요. 수업 시간은 유연하게 관리돼서 일찍 끝나는 경우가 많아요.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
조별 발표가 없는 경우, 해당 강의에서는 조별 과제나 발표 대신 개인별로 발표를 진행하거나, 조별 과제 없이 다른 형태의 평가가 이루어집니다. 예를 들어, '디자인의이해' 강의는 조별 과제/발표가 없으며, '세상과소통하는기초한자' 강의는 발표는 개인별로 진행되고 조별 과제는 없습니다.

[출처: si_static_info_12`
  - diagnosis: 정답 doc(lec_lecture_reviews_22_c2)이 검색되지 않아 핵심 정보(찬반 토론, 생각 정리, 유연한 시간 관리)를 놓쳤고, 대신 다른 강의의 평가 정보를 인용하여 답변이 흐려졌습니다.
  - fix: router에 정답 doc_id(lec_lecture_reviews_22_c2)를 포함하도록 CAMPUS_PATTERN을 수정하고, fallback 시 정답 doc이 존재하면 강제로 사용하도록 fallback 조건을 조정하세요.

- **Q** (T1_conversational, score=3): 팀플에서 잘하려면 어떻게 해야 해?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_114_c8'] / 검색 top3: ['si_static_info_280', 'si_static_info_121', 'si_static_info_295']
  - expected: `팀플에서 적극적으로 참여하고, 교수님의 피드백을 잘 반영해서 과제를 수행하면 좋은 성적을 받을 수 있습니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
팀플에서 잘하려면 팀원들과의 원활한 소통을 통해 역할을 명확히 분담하고, 참신한 아이디어를 제안하며 프로젝트를 성공적으로 이끌어가는 것이 중요합니다. 발표 시 Q&A에 대비하여 충분한 준비를 하는 것도 필요합니다.

[출처: doc_id=lec_lecture_reviews_38_c8, 카테고리=강의평가, 캠퍼스=전체]`
  - diagnosis: 정답 doc(lec_lecture_reviews_114_c8)은 교수님의 피드백 반영과 성적 향상을 강조하지만, 시스템은 강의평가(doc_id=lec_lecture_reviews_38_c8)에서 팀플 소통 및 역할 분담에 관한 내용을 인용하여 핵심 정보가 누락되었습니다.
  - fix: router에 정답 doc_id(lec_lecture_reviews_114_c8)를 명시적으로 매핑하고, answer prompt에 '교수님 피드백 반영'과 '성적' 키워드를 강조하여 핵심 정보를 정확히 추출하도록 개선하세요.

- **Q** (T1_conversational, score=3): 1학기 수업 언제 시작해요?
  - verdict=grounded retry=False retrieval_hit=True
  - 정답 doc: ['cal_calendar_22'] / 검색 top3: ['si_static_info_177', 'cal_calendar_11', 'cal_calendar_22']
  - expected: `1학기 수업개시일은 2026년 6월 15일 월요일입니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
1학기 수업은 2026년 3월 29일(일)에 시작합니다.

[출처: si_static_info_177, FAQ, 전체]`
  - diagnosis: 정답 doc(cal_calendar_22)은 검색되었으나, 시스템 답변은 다른 출처(si_static_info_177)의 정보를 인용하여 2026년 3월 29일이라는 잘못된 날짜를 제시했습니다.
  - fix: 검색된 sources 중 정답 doc(cal_calendar_22)을 우선적으로 사용하도록 answer prompt에 '정답 doc이 있을 경우 해당 정보를 우선 인용하라'는 규칙을 추가하세요.

- **Q** (T1_conversational, score=3): 시험 범위는 어떻게 정해지나요?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_108_c1'] / 검색 top3: ['lec_lecture_reviews_81_c1', 'lec_lecture_reviews_112_c1', 'lec_lecture_reviews_100_c3']
  - expected: `시험은 수업 중 교수님이 중요하다고 언급한 부분과 연습문제에서 주로 출제됩니다. 교수님이 미리 시험 문제를 언급해 주시기도 합니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
시험 범위는 강의마다 다르게 정해집니다.  
- **영어청취및독해Ⅰ**의 경우, 배운 내용에서만 출제되며 시험 범위가 명확히 정해져 있습니다.  
- **K-컨텐츠새로읽기**는 시험 1~2주 전에 PPT를 통해 시험 범위를 공지합니다.  
- **수학**은 강의 내용에 기반하지만 구체적인 범위 공지가 부족하다는 의견이 있습니다`
  - diagnosis: 검색된 출처들은 강의평가 내용으로 시험 범위 결정 방식을 직접 설명하지 않아 핵심 정보가 누락되었습니다.
  - fix: router에 시험 범위 관련 키워드(예: '시험 범위', '출제')를 추가하고, 강의계획서나 교수 공지 문서를 우선적으로 검색하도록 개선하세요.

- **Q** (T1_conversational, score=3): 이 수업 지루해요?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_2_c21'] / 검색 top3: ['lec_lecture_reviews_12_c15', 'lec_lecture_reviews_110_c6', 'lec_lecture_reviews_12_c17']
  - expected: `지루함을 견디기 어려운 학생에게는 비추천합니다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
이 수업은 지루하다는 의견이 다수 있습니다.  
한시의이해 강의는 수업 내용이 지루하다는 평가가 많으며, 중국문학의이해 강의는 말이 빠르고 사투리가 있어 이해가 어려울 수 있고, 자연과학의이해 강의는 PPT를 읽는 방식이 주를 이루어 지루하다는 의견이 있습니다.  
창의적사고와코딩 강의 역시 지루하고 전달력이 부족하다는 점이 `
  - diagnosis: 정답 doc(lec_lecture_reviews_2_c21)이 검색되지 않아 핵심 정보가 누락되었고, 검색된 출처들이 다른 강의에 대한 평가만 인용하여 답변이 흐려졌습니다.
  - fix: router에 lec_lecture_reviews_2_c21을 포함한 강의평가 문서를 정확히 검색하도록 패턴을 추가하고, answer prompt에 질문의 핵심 키워드('지루해요')에 직접 응답하도록 지시하세요.

### Mode E — Fallback 과반응 — 부적절한 거부

**권장 fix 방향**: fallback 트리거 임계값 재튜닝, two-stage retry (top-10 확장 후 재시도) 도입

- **Q** (T1_conversational, score=1): 결석 3번 넘으면 어떻게 돼?
  - verdict=grounded retry=False retrieval_hit=False
  - 정답 doc: ['학칙_학칙_대학_295'] / 검색 top3: ['si_static_info_126', 'si_static_info_68', 'si_static_info_270']
  - expected: `3회(15주 기준 3주 이상) 결석하면 담당자가 학과장에게 보고하고, 학과장이 교무혁신처에 명단을 제출한다.`
  - answer:   `[성남캠퍼스 기준 답변입니다]
결석 3번만으로는 F 처리 여부를 판단할 수 없습니다. F 처리는 전체 수업 시간의 1/4 이상 결석 시 적용되며, 3번 결석이 이 기준에 해당하는지 여부는 수업 시간 및 학점 수에 따라 달라집니다. 정확한 기준은 강의계획서를 확인하세요.

[출처: si_static_info_126, FAQ, 전체]  
[출처: si_stat`
  - diagnosis: 검색된 출처가 정답 doc_id(학칙_학칙_대학_295)를 포함하지 않아 정답을 찾지 못했으나, 시스템은 grounded verdict를 내려 fallback을 부적절하게 거부했습니다.
  - fix: retrieval_hit가 false일 때 verdict를 notGrounded로 강제하고, 정답 doc_id가 포함된 출처를 우선 검색하도록 router의 검색 로직을 개선하세요.

- **Q** (T1_conversational, score=1): ㅇㅎㅅ 교수님은 그림 잘 그리세요?
  - verdict=notGrounded retry=True retrieval_hit=True
  - 정답 doc: ['lec_lecture_reviews_111_c2'] / 검색 top3: ['lec_lecture_reviews_115_c0', 'lec_lecture_reviews_111_c2', 'lec_lecture_reviews_38_c0']
  - expected: `네, 그림을 잘 그리시며 PPT와 판서를 병행해 수업을 재미있고 쉽게 진행하십니다.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: lec_lecture_reviews_115_c0, 강의평가, 전체]`
  - diagnosis: 검색된 sources에 정답 doc(lec_lecture_reviews_111_c2)이 포함되어 있음에도 불구하고, 시스템은 '제공된 자료에서 해당 정보를 찾을 수 없습니다'라고 응답하여 정답을 놓쳤습니다.
  - fix: router가 정답 doc을 올바르게 선택하도록 강의평가 카테고리에서 교수별 그림 실력 관련 키워드를 명시적으로 매핑하고, fallback 조건을 완화하여 정답이 존재할 경우 반드시 응답하도록 개선하세요.

- **Q** (T2_vague, score=1): A+ 받으려면 시험에서 몇 개까지 틀려도 돼?
  - verdict=notGrounded retry=True retrieval_hit=True
  - 정답 doc: ['lec_lecture_reviews_18_c17'] / 검색 top3: ['lec_lecture_reviews_18_c17', 'lec_lecture_reviews_40_c4', 'lec_lecture_reviews_17_c4']
  - expected: `A+를 받기 위해서는 시험에서 5개 이하의 문제를 틀려야 한다는 의견이 많습니다.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: lec_lecture_reviews_18_c17, 강의평가, 전체]`
  - diagnosis: 정답 doc(lec_lecture_reviews_18_c17)에 A+를 받기 위해 5개 이하 틀려도 된다는 정보가 있음에도 불구하고, 시스템은 '제공된 자료에서 해당 정보를 찾을 수 없습니다'라고 잘못 응답했습니다.
  - fix: router에 lec_lecture_reviews_18_c17을 정답 doc으로 명시적으로 매핑하고, answer prompt에 'A+를 받기 위한 최대 오답 수' 같은 구체적 질문 키워드를 강조하여 검색 정확도를 높이세요.

- **Q** (T2_vague, score=1): 교수님이 학생들을 위해 어떤 배려를 보여주시는지 궁금해요
  - verdict=notGrounded retry=True retrieval_hit=False
  - 정답 doc: ['lec_lecture_reviews_73_c1'] / 검색 top3: ['lec_lecture_reviews_37_c0', 'lec_lecture_reviews_3_c0', 'lec_lecture_reviews_14_c0']
  - expected: `교수님은 학생들의 편의를 많이 고려하며, 친절하게 수업을 진행하십니다. 카메라를 켜라고 하시지만, 강하게 요구하지는 않으십니다.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: lec_lecture_reviews_37_c0, 강의평가, 전체]`
  - diagnosis: 검색된 sources에 정답 doc_id(lec_lecture_reviews_73_c1)가 포함되지 않았고, 시스템이 '정보 없음'으로 fallback했으나 corpus에 정답이 존재해 부적절 거부되었습니다.
  - fix: router에 정답 doc_id를 포함하도록 검색 범위를 확장하거나, 정답 doc_id를 명시적으로 매핑하는 fallback 규칙을 추가하세요.

- **Q** (T3_paraphrase, score=1): ㅇㅇㄱ 교수님의 중간고사는 어떤 형식으로 진행되며, 어떤 것을 평가하나요?
  - verdict=notGrounded retry=True retrieval_hit=True
  - 정답 doc: ['lec_lecture_reviews_47_c7'] / 검색 top3: ['lec_lecture_reviews_74_c1', 'lec_lecture_reviews_47_c7', 'lec_lecture_reviews_47_c6']
  - expected: `중간고사는 오픈북 서술형으로 진행되며, 학생들의 사고력을 평가합니다.`
  - answer:   `제공된 자료에서 해당 정보를 찾을 수 없습니다.

[출처: lec_lecture_reviews_74_c1, 강의평가, 전체]`
  - diagnosis: 검색된 sources에 정답 doc(lec_lecture_reviews_47_c7)이 포함되어 있음에도 불구하고, 시스템이 '제공된 자료에서 해당 정보를 찾을 수 없습니다'라고 응답하여 정답을 제공하지 못했습니다.
  - fix: router가 정답 doc을 정확히 선택하도록 강의 제목이나 교수명을 포함한 키워드 매칭을 강화하고, answer prompt에서 '해당 정보가 있으면 반드시 인용하라'는 지시를 추가하여 fallback을 방지하세요.

---

## 5. 빈도 기반 Top Fix 제안 (Solar 자동 추출)

| 빈도 | 제안 |
|------|------|
| 41 | 없음 |
| 4 | 현재 시스템이 이미 잘 작동하고 있으므로 특별한 수정이 필요하지 않습니다. |
| 3 | 현재 답변이 이미 정확하므로 추가 수정이 필요하지 않습니다. |
| 3 | 현재 시스템이 이미 정확하게 동작하고 있으므로 추가 수정이 필요하지 않습니다. |
| 2 | 현재 답변이 이미 충분히 정확하므로 추가 수정이 필요하지 않습니다. |
| 2 | 현재 답변이 이미 완벽하므로 추가 수정이 필요 없습니다. |
| 2 | 현재 시스템이 이미 잘 작동하고 있으므로 추가 수정이 필요하지 않습니다. |
| 2 | 현재 답변은 이미 정확하므로 추가 수정이 필요하지 않습니다. |
| 2 | 현재 시스템이 이미 정답을 정확히 제공하고 있으므로 추가 수정이 필요하지 않습니다. |
| 1 | 정답 doc에 명시된 동아리 목록을 정확히 반영하고, 불필요한 '스포츠 동아리' 분류를 제거하세요. |

---

## 6. 산출물

- `reports/quality_diagnosis.md` — 이 보고서
- `reports/quality_diagnosis.json` — 기계 판독용 통계
- `reports/quality_judgments.jsonl` — row-level 판정
- `logs/quality_traces.jsonl` — row-level trace
- `data/qa_adversarial.parquet` — adversarial 평가셋
- `data/qa_bad_cases.parquet` — quality_score < 3 행 (회귀 테스트용)