# Query Skeleton Gap Report

## 1. 목적

이 문서는 deterministic query skeleton에서 GPT-4.1 teacher query와 query-only LoRA query를
row-level로 비교한다. 앞선 `docs/teacher_query_skeleton_report.md`는 aggregate metric을
정리했다. 여기서는 다음 질문에 답한다.

```text
Teacher max-3 query는 reference를 찾는데 query-only skeleton은 놓친 row가 얼마나 있고,
그 row들은 어떤 query pattern 차이를 보이는가?
```

비교 기준은 `dataset_index` join이다. Teacher 쪽은 search budget을 최대 3회로 제한한
`teacher_query_skeleton_dmr_approved500_max3` 결과를 사용했고, student 쪽은 raw 500-row
`query_skeleton_dmr_evidence_only500` 결과를 사용했다.

## 2. Aggregate Gap

### 2.1 Approved 398 subset

| Category | Retrieval count | Retrieval rate | Containment count | Containment rate |
| --- | ---: | ---: | ---: | ---: |
| Both teacher and student correct | `74` | `0.186` | `63` | `0.158` |
| Teacher only correct | `72` | `0.181` | `73` | `0.183` |
| Student only correct | `36` | `0.090` | `36` | `0.090` |
| Neither correct | `216` | `0.543` | `226` | `0.568` |

| Metric | Teacher max-3 | Query-only r16 |
| --- | ---: | ---: |
| Retrieved-reference rate | `0.367` | `0.276` |
| Containment | `0.342` | `0.249` |
| Mean retrieved messages | `3.25` | `3.76` |

### 2.2 Teacher-search 302 subset

| Category | Retrieval count | Retrieval rate | Containment count | Containment rate |
| --- | ---: | ---: | ---: | ---: |
| Both teacher and student correct | `74` | `0.245` | `63` | `0.209` |
| Teacher only correct | `72` | `0.238` | `73` | `0.242` |
| Student only correct | `8` | `0.026` | `9` | `0.030` |
| Neither correct | `148` | `0.490` | `157` | `0.520` |

| Metric | Teacher max-3 | Query-only r16 |
| --- | ---: | ---: |
| Retrieved-reference rate | `0.483` | `0.272` |
| Containment | `0.450` | `0.238` |
| Mean retrieved messages | `4.28` | `3.76` |

Teacher-search subset이 더 중요한 비교다. Teacher가 search를 수행한 row에서 student-only
retrieval success는 `8/302`뿐이다. 반면 teacher-only retrieval success는 `72/302`다. 즉 student가
teacher를 자주 대체하는 것이 아니라, teacher가 잡는 answer-bearing query를 student가 자주 놓친다.

## 3. Qualitative Pattern

Teacher-only 사례를 보면 세 가지 패턴이 반복된다.

### 3.1 Probe phrase copying

Student는 probe의 긴 표현을 그대로 query로 사용하는 경향이 있다.

| Row | Reference | Teacher queries | Student queries |
| ---: | --- | --- | --- |
| `33` | `California!` | `studio`, `California`, `Santa Barbara` | `audio studio`, `audio studio location`, `your audio studio` |
| `77` | `National parks!` | `national parks` | `what we do for fun`, `that place`, `like to visit` |

긴 probe phrase는 질문에는 등장하지만 과거 memory utterance에는 그대로 등장하지 않는 경우가 많다.
Substring recall에서는 이런 query가 특히 불리하다.

### 3.2 Plausible but wrong semantic guess

Student가 정답을 찾기 전에 plausible한 오답 후보를 query로 고르는 경우가 있다.

| Row | Reference | Teacher queries | Student queries | Student answer |
| ---: | --- | --- | --- | --- |
| `5` | `I have a cat.` | `pets`, `cat`, `cats` | `pet`, `dogs`, `dogs` | `I have a dog.` |
| `19` | `Lawn mowing!` | `hobbies`, `outdoors`, `lawn` | `outdoors`, `grilling`, `food` | `I enjoy grilling my food outdoors.` |
| `65` | `The Navy!` | `military`, `Navy` | `military service`, `branch`, `army` | `UNKNOWN` |

이 패턴은 query generation이 retrieval problem이 아니라 answer prior problem과 섞여 있음을
보여 준다. 모델이 memory를 찾기 전에 그럴듯한 answer candidate를 상상하고, 그 candidate를
검색 query에 넣는다.

### 3.3 Missing discriminative literal

Teacher는 broad query 뒤에 discriminative literal로 좁히는 경우가 많다.

| Row | Reference | Teacher queries | Student queries |
| ---: | --- | --- | --- |
| `10` | `Construction, like my dad.` | `job`, `work`, `construction` | `living`, `jobs`, `work` |
| `36` | `A Twitter contest!` | `win`, `contest`, `prize` | `what did you win`, `award`, `prize` |
| `81` | `Soda!` | `stopped drinking`, `drink`, `soda` | `stopped drinking`, `coffee`, `drinking` |

Student도 broad keyword를 어느 정도 찾지만, 마지막 discriminative literal이 틀리거나 빠진다.
따라서 학습 target은 단순히 teacher 첫 query가 아니라 query chain에서 broad-to-specific
refinement를 배워야 한다.

## 4. 해석

이 분석은 query-only SFT의 위치를 더 정확히 정한다.

1. Query-only skeleton은 random이 아니다. Approved subset에서 retrieved-reference `0.276`,
   containment `0.249`를 기록했고, student-only success도 `36`개 있다.
2. 그러나 teacher max-3와의 gap은 크다. Teacher-search subset에서 teacher-only retrieval success
   `72`개에 비해 student-only success는 `8`개다.
3. Query-only는 평균 retrieved message 수가 작지 않다. Approved subset에서는 teacher max-3보다
   더 많이 retrieve한다. 문제는 recall volume이 아니라 specificity다.
4. 다음 학습은 query imitation보다 retrieval-supervised query learning이 더 직접적이다.

## 5. 다음 학습 설계

가장 자연스러운 다음 objective는 다음과 같다.

```text
Input:
  DMR probe + search instruction

Positive query:
  reference-containing message를 retrieve하는 teacher query 또는 mined query

Negative query:
  no result, distractor-only result, 또는 plausible-but-wrong answer prior query
```

가능한 구현 방향은 세 가지다.

| Direction | Description | Why useful |
| --- | --- | --- |
| Pairwise query ranking | 같은 row에서 positive query와 negative query를 비교 | teacher-only 실패 row를 직접 사용 가능 |
| Hard-negative SFT | student wrong query를 negative example로 설명하고 teacher query를 target으로 학습 | 현재 LoRA pipeline과 가장 가까움 |
| Retrieval-aware reranker | 여러 candidate query를 생성한 뒤 local substring hit proxy로 rerank | small model이 후보를 여러 개 낼 수 있는 경우 적합 |

우선순위는 `teacher_only` retrieval row 72개다. 이 row들은 teacher query가 실제로 reference를
찾았고, query-only가 놓쳤기 때문에 query objective의 가장 깨끗한 hard set이다.

## 6. Artifacts

```text
scripts/analyze_query_skeleton_gap.py
data/analysis/query_skeleton_gap/query_skeleton_gap_all.csv
data/analysis/query_skeleton_gap/query_skeleton_gap_all.jsonl
data/analysis/query_skeleton_gap/query_skeleton_gap_all.summary.json
data/analysis/query_skeleton_gap/query_skeleton_gap_all.md
data/analysis/query_skeleton_gap/query_skeleton_gap_teacher_search.csv
data/analysis/query_skeleton_gap/query_skeleton_gap_teacher_search.jsonl
data/analysis/query_skeleton_gap/query_skeleton_gap_teacher_search.summary.json
data/analysis/query_skeleton_gap/query_skeleton_gap_teacher_search.md
```
