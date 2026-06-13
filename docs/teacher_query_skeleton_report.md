# Teacher Query Skeleton Replay 보고서

## 1. 목적

`docs/query_skeleton_dmr_report.md`에서는 query-only r16 adapter가 deterministic query
skeleton 안에서 full trajectory r16보다 좋은 query generator가 된다는 것을 확인했다. 하지만
그 `0.23` retrieved-reference rate가 충분한지 판단하려면 teacher query 자체의 upper bound가
필요하다.

이번 실험은 GPT-4.1 teacher가 실제 DMR에서 사용한 `conversation_search` query string을 같은
local substring skeleton으로 replay한다. 즉 Letta agent loop나 OpenAI tool-call transport를
제거하고, query string 자체가 reference-bearing utterance를 얼마나 잘 retrieve하는지 측정한다.

## 2. 실험 조건

추가 스크립트:

```text
scripts/eval_teacher_query_skeleton_dmr.py
```

입력 trace:

```text
data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl
```

Answer model은 이전 skeleton 실험과 동일하게 `nano-memgpt-llama3-r16`을 사용했다. Answer prompt는
retrieved evidence only 조건이다. Teacher가 search를 하지 않은 approved row는 retrieved
evidence가 없으므로 `UNKNOWN`으로 끝난다. 따라서 결과는 반드시 `all approved`와
`teacher search subset`으로 나누어 해석해야 한다.

## 3. 0-100 구간 결과

Offset `0`, limit `100` 안에는 approved teacher row가 `75`개 있었다. 그중 teacher가 실제
`conversation_search`를 한 row는 `46`개였다.

| Condition | Subset | Rows | Containment | Retrieved-reference | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher query skeleton | approved | `75` | `0.333` | `0.360` | `2.97` |
| Query-only r16 skeleton | same approved | `75` | `0.267` | `0.280` | `3.57` |
| Full r16 skeleton | same approved | `75` | `0.213` | `0.187` | `2.77` |
| Teacher query skeleton | teacher-search only | `46` | `0.543` | `0.587` | `4.85` |
| Query-only r16 skeleton | teacher-search only | `46` | `0.217` | `0.239` | `3.17` |
| Full r16 skeleton | teacher-search only | `46` | `0.174` | `0.174` | `2.87` |

이 결과는 두 가지를 보여 준다.

1. Query-only skeleton은 full r16 skeleton보다 강하다.
2. 하지만 teacher search subset에서는 teacher query와의 gap이 크다.

## 4. Scaled approved 398 결과

승인된 teacher row 전체 `398`개로 확장했다.

### 4.1 Teacher full query chain

먼저 teacher가 실제 실행한 query chain 전체를 replay했다. 이 조건은 search budget이 query-only
skeleton보다 크다. 일부 row는 teacher가 8회, 10회, 15회까지 검색했다.

| Subset | Rows | Containment | Retrieved-reference | Mean searches | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| All approved | `398` | `0.405` | `0.442` | `2.95` | `4.36` |
| Teacher search subset | `302` | `0.533` | `0.583` | `3.89` | `5.74` |
| Teacher no-search subset | `96` | `0.000` | `0.000` | `0.00` | `0.00` |

Teacher no-search subset이 `0`인 것은 teacher가 틀렸다는 뜻이 아니다. 이 replay는 evidence-only
answer condition이므로, teacher가 original run에서 core/persona/immediate context로 바로 답한
row는 evidence가 비어 있어 `UNKNOWN`이 된다.

### 4.2 Teacher max-3 query chain

공정 비교를 위해 teacher query도 query-only skeleton과 같은 max-search `3` budget으로 잘랐다.

| Subset | Rows | Containment | Retrieved-reference | Mean searches | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| All approved | `398` | `0.342` | `0.367` | `1.80` | `3.25` |
| Teacher search subset | `302` | `0.450` | `0.483` | `2.37` | `4.28` |
| Teacher no-search subset | `96` | `0.000` | `0.000` | `0.00` | `0.00` |

### 4.3 Query-only r16 skeleton 500-row comparison

Query-only r16 skeleton도 raw 500 row 전체로 확장했다. 전체 결과:

| Rows | Containment | Retrieved-reference | Mean searches | Mean retrieved |
| ---: | ---: | ---: | ---: | ---: |
| `500` | `0.216` | `0.244` | `2.98` | `3.74` |

Teacher approved subset에 맞춰 비교하면 다음과 같다.

| Condition | Subset | Rows | Containment | Retrieved-reference | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher max-3 query | approved | `398` | `0.342` | `0.367` | `3.25` |
| Query-only r16 skeleton | same approved | `398` | `0.249` | `0.276` | `3.76` |
| Teacher max-3 query | teacher-search only | `302` | `0.450` | `0.483` | `4.28` |
| Query-only r16 skeleton | teacher-search only | `302` | `0.238` | `0.272` | `3.76` |

즉 search budget을 맞춰도 teacher query는 query-only skeleton보다 높다. 하지만 gap이 무한히 큰
것은 아니다. Query-only r16은 approved subset에서 teacher max-3의 약 73% 수준의 containment
(`0.249 / 0.342`)을 달성했고, teacher-search subset에서는 약 53% 수준
(`0.238 / 0.450`)을 달성했다.

## 5. 해석

### 5.1 Query-only SFT에는 실제 query-policy signal이 있다

Query-only skeleton은 full r16 skeleton보다 좋고, raw 500 전체에서도 containment `0.216`,
retrieved-reference `0.244`를 달성했다. 이는 query-only SFT가 단순히 tool-call 형식만 외운 것이
아니라, 일부 유효한 literal query selection behavior를 학습했다는 뜻이다.

### 5.2 하지만 teacher query와는 아직 큰 gap이 있다

Teacher max-3 query는 teacher-search subset에서 retrieved-reference `0.483`이다. Query-only
skeleton은 같은 subset에서 `0.272`다. 즉 현재 query-only adapter는 teacher query의 약 절반
수준만 answer-bearing evidence를 찾는다.

### 5.3 Query-only는 no-search teacher row도 일부 해결한다

Teacher no-search subset `96`개에서는 teacher replay가 evidence-only 조건상 모두 `UNKNOWN`이
된다. 반면 query-only skeleton은 같은 subset에서 containment `0.281`, retrieved-reference
`0.292`를 보였다. 이는 teacher가 original run에서 core/persona로 답한 row에도, local history
검색으로 답을 찾을 수 있는 경우가 있음을 뜻한다.

### 5.4 다음 병목은 query candidate와 distractor control이다

Query-only skeleton은 teacher보다 mean retrieved가 approved subset에서 더 높다
(`3.76` vs teacher max-3 `3.25`)인데 retrieved-reference는 낮다. 이는 query-only가 더 넓거나
덜 특정적인 query를 생성해 distractor를 많이 가져온다는 뜻이다.

## 6. 다음 실험

1. Teacher max-3 query와 query-only skeleton query를 row-level로 비교한다.
2. Query-only가 틀리고 teacher가 맞힌 row를 positive training candidates로 만든다.
3. Query-only가 teacher보다 넓게 검색한 row에서 distractor query pattern을 분류한다.
4. Retrieval-supervised objective를 만든다.

예시 objective:

```text
positive query: reference-containing message를 retrieve함
hard negative query: 많은 message를 retrieve하지만 reference-containing message는 못 찾음
easy negative query: result가 없음
```

## 7. Artifacts

```text
scripts/eval_teacher_query_skeleton_dmr.py
data/evaluation/teacher_query_skeleton_dmr_approved100/
data/evaluation/teacher_query_skeleton_dmr_approved500/
data/evaluation/teacher_query_skeleton_dmr_approved500_max3/
data/evaluation/query_skeleton_dmr_evidence_only500/
data/evaluation/teacher_query_skeleton_dmr_approved500/teacher-query-to-nano-memgpt-llama3-r16-skeleton-offset-0-limit-500.jsonl
data/evaluation/teacher_query_skeleton_dmr_approved500/teacher-query-to-nano-memgpt-llama3-r16-skeleton-offset-0-limit-500.summary.json
data/evaluation/teacher_query_skeleton_dmr_approved500_max3/teacher-query-to-nano-memgpt-llama3-r16-skeleton-offset-0-limit-500.jsonl
data/evaluation/teacher_query_skeleton_dmr_approved500_max3/teacher-query-to-nano-memgpt-llama3-r16-skeleton-offset-0-limit-500.summary.json
data/evaluation/query_skeleton_dmr_evidence_only500/nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-skeleton-searches-3-offset-0-limit-500.jsonl
data/evaluation/query_skeleton_dmr_evidence_only500/nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-skeleton-searches-3-offset-0-limit-500.summary.json
```
