# Phase-Routed DMR 진단 보고서

## 1. 목적

Query-only LoRA는 end-to-end Letta agent로는 실패했다. Endpoint proxy가 assistant
`content` 안의 JSON을 `tool_calls`로 70회 구제했는데도 20-row smoke는 `2/20` completion,
containment `0.0`에 머물렀다. 이 결과만 보면 query-only adapter가 쓸모없는 것처럼 보일 수
있지만, 실제로는 두 문제가 섞여 있었다.

| 문제 | 의미 |
| --- | --- |
| stop-and-answer failure | 모델이 검색을 멈추고 최종 답변으로 전환하지 못함 |
| query-quality failure | 검색은 하지만 answer-bearing utterance를 찾는 query를 못 고름 |

이번 phase-routed 진단은 이 둘을 분리한다. Letta multi-turn loop 대신 deterministic
controller가 search phase와 answer phase를 나누고, search phase는 query-only adapter,
answer phase는 full trajectory LoRA r16 adapter를 사용한다.

## 2. 실험 조건

| 항목 | 값 |
| --- | --- |
| Protocol | `msc_dmr_phase_routed_query_only_search_answer_model_v1` |
| Search model | `nano-memgpt-llama3-query-only-r16` |
| Answer model | `nano-memgpt-llama3-r16` |
| Dataset | MSC DMR offset `0`, limit `100` |
| Search execution | local case-insensitive substring recall |
| Max searches | `3` |
| Answer prompt | retrieved evidence only |
| Persona/context leakage | 제거 |

Search phase에서는 OpenAI tool choice를 `conversation_search`로 강제했다. 모델은 query string을
포함한 tool call을 생성하고, controller는 그 query를 local DMR history에 대해 substring
검색으로 실행한다. Answer phase에는 probe, search query, retrieved evidence만 제공한다.
Evidence가 비어 있거나 부족하면 `UNKNOWN`을 출력하도록 지시했다.

이 조건은 Letta agent performance가 아니라 diagnostic이다. 목적은 query-only adapter가
검색 phase 전용 모듈로 쓸 수 있는지, 그리고 검색어 자체가 정답 근거를 얼마나 자주 가져오는지
측정하는 것이다.

## 3. 결과

### 3.1 100-row summary

| Metric | Value |
| --- | ---: |
| rows | `100` |
| completed | `100/100` |
| errors | `0` |
| mean searches | `2.22` |
| mean retrieved messages | `1.10` |
| rows with any retrieved message | `36/100` |
| rows with zero retrieved messages | `64/100` |
| retrieved evidence contains reference | `9/100` |
| answer contains reference | `11/100` |
| mean ROUGE-L recall | `0.2010` |
| answers beginning with `UNKNOWN` | `67/100` |

Search count distribution:

| Search count | Rows |
| ---: | ---: |
| `2` | `78` |
| `3` | `22` |

Retrieved-message distribution:

| Retrieved messages | Rows |
| ---: | ---: |
| `0` | `64` |
| `1` | `15` |
| `2` | `8` |
| `3+` | `13` |

### 3.2 20-row evidence-only smoke

동일한 evidence-only 설정으로 먼저 20-row smoke를 실행했다.

| Metric | Value |
| --- | ---: |
| rows | `20` |
| completed | `20/20` |
| answer containment | `0.10` |
| retrieved evidence contains reference | `0.10` |
| mean searches | `2.30` |
| mean retrieved messages | `1.60` |

100-row 결과도 같은 방향으로 나왔다. 따라서 low score는 작은 smoke의 우연이라기보다
현재 query-only search policy의 구조적 한계로 보는 것이 안전하다.

## 4. 성공 예시

성공한 row들은 대부분 probe의 단서와 answer-bearing utterance가 같은 literal phrase를
공유한다.

| Index | Query chain | Reference | Answer |
| ---: | --- | --- | --- |
| `2` | `work in`, `work in the movies`, `work in the movies` | `I used to work in the human services field.` | `I used to work in the human services field.` |
| `35` | `favorite food`, `favorite food` | `Cheeseburgers!` | `My favorite food is cheeseburgers.` |
| `50` | `german shepherd`, `german shepherd` | `Barnaby!` | `Your German Shepherd's name is Barnaby.` |
| `54` | `banana`, `banana` | `That the banana only stretches once.` | `The banana only stretches once.` |
| `58` | `pizza`, `pepperoni`, `pepperoni` | `Pepperoni!` | `Pepperoni.` |
| `90` | `music`, `death metal music`, `death metal music` | `Death metal music!` | `I said I enjoyed death metal music the most.` |

이 예시들은 query-only adapter가 완전히 무작위인 것은 아니라는 점을 보여 준다. Literal
anchor가 probe에 충분히 드러나면 적절한 검색어를 만들 수 있다.

## 5. 실패 예시

### 5.1 너무 좁거나 어긋난 query

Probe:

```text
Hey, remember that time we talked about music? What was the artist you mentioned you could get into?
```

Reference:

```text
Taylor Swift!
```

Query-only search:

```text
artist
artist
```

Substring recall 결과는 `No results found`였다. Answer phase는 evidence-only 조건이므로
`UNKNOWN`을 출력했다. 이 row는 hallucination을 줄인다는 점에서는 바람직하지만, query가
answer-bearing utterance를 찾지 못했다.

### 5.2 관련 evidence를 찾았지만 정답 evidence는 아님

Probe:

```text
Hey, remember that time we talked about our jobs and expenses?
What was that one thing you said you did to save money?
```

Reference:

```text
I eat a fresh and raw diet to save on groceries.
```

Query-only search:

```text
save money
save money
```

검색 결과에는 `save money` 관련 다른 메시지가 들어왔지만, 정답인 raw diet/groceries
utterance는 없었다. Answer model은 retrieved evidence에 있는 coupon 내용을 답했다. 이
row는 검색 결과가 0개는 아니어도 정답 근거가 아닐 수 있음을 보여 준다.

## 6. 해석

Phase-routed controller는 stop-and-answer failure를 제거했다. 100-row 평가에서 behavioral
failure는 `0`이고 모든 row가 완료되었다. 따라서 endpoint proxy-rescue에서 보였던 `2/20`
completion 문제는 query-only adapter를 end-to-end Letta agent로 쓸 때의 phase-control
문제였음이 확인된다.

하지만 phase-control을 제거해도 answer containment는 `0.11`, retrieved-reference rate는
`0.09`에 머문다. 즉 남은 병목은 query content다. Query-only adapter는 tool call 모양과
검색 반복 행동은 학습했지만, DMR probe에서 정답 utterance에 도달하는 literal query를 충분히
잘 고르지 못한다.

이 결과는 기존 ablation들과 잘 맞는다.

| Condition | 핵심 결과 | 해석 |
| --- | ---: | --- |
| LoRA r16 end-to-end | judge `0.4809` | loop는 안정화됐지만 query/evidence gap 존재 |
| Teacher-query hint | judge `0.6294` | 좋은 query는 성능을 크게 올림 |
| Teacher evidence replay | judge `0.8668` | evidence가 있으면 answer model은 잘 답함 |
| Query-only phase-routed evidence-only | containment `0.11`, retrieved-hit `0.09` | query-only search policy 자체는 아직 약함 |

따라서 query-only adapter는 현재 상태로는 바로 성능 개선 모듈이 아니라 실패 원인을
분리하기 위한 diagnostic이다.

## 7. 후속 결과: Deterministic query skeleton

이 보고서 이후 tool-call shell을 더 강하게 제거한 deterministic query skeleton을 실행했다.
모델은 `conversation_search` JSON을 만들지 않고 query string만 생성하며, wrapper가 tool shell을
고정한다. 자세한 내용은 `docs/query_skeleton_dmr_report.md`에 정리했다.

| Query generator | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `0.09` | `0.11` | `1.10` |
| Query-only r16, skeleton | `100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `0.16` | `0.17` | `2.86` |

이 결과는 query-only adapter 안에 유효한 query-policy signal이 있음을 보여 준다. 다만 그
signal은 전체 tool-call JSON을 생성하게 할 때 약해지고, query string만 생성하게 할 때 더 잘
드러난다.

## 8. 다음 실험

### 8.1 Teacher query skeleton replay

Teacher query를 같은 local skeleton recall로 실행해 query upper bound를 측정한다. 이 조건은
teacher-query hint의 Letta execution과 달리 query string 자체의 retrieval power를 직접 본다.

### 8.2 Teacher-query imitation의 평가 방식 변경

현재 query-only SFT target은 teacher의 tool-call JSON 전체다. 다음에는 `query` 값만 target으로
두고, 후보 query 여러 개를 beam 또는 self-consistency 방식으로 생성한 뒤 substring recall hit
rate를 평가한다.

### 8.3 Retrieval-supervised objective

Token-level query imitation만으로는 answer-bearing evidence hit가 낮다. 다음 objective는
teacher query와 동일한 문자열을 맞히는 것보다, reference가 포함된 utterance를 실제로
retrieve하는 query를 선호하도록 설계해야 한다.

예시는 다음과 같다.

```text
positive: query가 reference-containing message를 retrieve함
negative: query가 결과 없음 또는 distractor만 retrieve함
loss: contrastive / pairwise ranking / RL-style reward
```

### 8.4 Evidence sufficiency classifier

Phase-routed controller는 현재 max-search rule만 사용한다. 실제 agent로 확장하려면 retrieved
evidence가 probe에 충분한지 판단하는 작은 classifier 또는 rule이 필요하다. 다만 이번 결과상
우선순위는 sufficiency classifier보다 query hit rate 개선이다.

## 9. Artifacts

```text
scripts/eval_phase_routed_dmr.py
scripts/eval_query_skeleton_dmr.py
data/evaluation/phase_routed_dmr_smoke/
data/evaluation/phase_routed_dmr_forced_smoke/
data/evaluation/phase_routed_dmr_evidence_only20/
data/evaluation/phase_routed_dmr_evidence_only100/
data/evaluation/phase_routed_dmr_evidence_only100/nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-searches-3-offset-0-limit-100.jsonl
data/evaluation/phase_routed_dmr_evidence_only100/nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-searches-3-offset-0-limit-100.summary.json
data/evaluation/query_skeleton_dmr_evidence_only100/
data/evaluation/query_skeleton_dmr_r16_100/
```
