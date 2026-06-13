# Deterministic Query Skeleton DMR 보고서

## 1. 목적

Phase-routed diagnostic은 query-only adapter를 search phase 전용으로 분리하면 loop는
`100/100` 완료되지만, retrieved-reference rate가 `0.09`에 그친다는 결과를 보였다. 그러나 그
조건에서도 모델은 여전히 `conversation_search` tool-call JSON을 생성해야 했다. 즉 query
content와 tool-call shell 생성 부담이 완전히 분리되지는 않았다.

이번 실험은 더 강한 controller 조건이다. 모델은 전체 tool call을 만들지 않고 query string만
생성한다. Wrapper가 다음 skeleton을 고정한다.

```json
{
  "name": "conversation_search",
  "arguments": {
    "query": "<MODEL_GENERATED_QUERY>",
    "roles": ["assistant", "user"],
    "limit": 10,
    "request_heartbeat": true
  }
}
```

따라서 이 실험은 다음 질문에 답한다.

```text
tool-call JSON shell을 제거하면 query-only adapter의 실제 검색어 품질은 얼마나 회복되는가?
```

## 2. 구현

추가 스크립트:

```text
scripts/eval_query_skeleton_dmr.py
```

Search phase는 tool schema를 모델에 전달하지 않는다. 모델에는 DMR probe와 이전 검색 결과만
보여 주고, "짧은 literal substring query 한 줄만 출력하라"고 지시한다. 모델이 습관적으로
JSON을 출력할 수 있으므로 parser는 다음 형태를 모두 query string으로 정규화한다.

```text
Taylor Swift
query: Taylor Swift
{"query": "Taylor Swift"}
{"arguments": {"query": "Taylor Swift"}}
```

검색 실행은 기존 phase-routed 실험과 동일한 local case-insensitive substring recall이다.
Answer phase도 동일하게 `nano-memgpt-llama3-r16`을 사용하며, retrieved evidence만 보고 답한다.
Evidence가 비어 있거나 부족하면 `UNKNOWN`을 출력하게 했다.

## 3. 20-row smoke

먼저 세 query generator를 20-row smoke로 비교했다.

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `20` | `20/20` | `0.10` | `0.10` | `1.60` |
| Query-only r16, skeleton | `20` | `20/20` | `0.35` | `0.35` | `4.20` |
| Full trajectory r16, skeleton | `20` | `20/20` | `0.25` | `0.30` | `3.75` |
| Base Llama-3-8B, skeleton | `20` | `20/20` | `0.20` | `0.20` | `1.75` |

20-row에서는 skeleton이 큰 차이를 만들었다. Query-only r16은 tool-call JSON을 생성할 때
retrieved-reference `0.10`에 그쳤지만, query string만 생성하면 `0.35`까지 올랐다.

## 4. 100-row 결과

100-row에서는 query-only r16 skeleton과 full trajectory r16 skeleton을 비교했다.

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | ROUGE-L recall | Mean searches | Mean retrieved | `UNKNOWN` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `100/100` | `0.09` | `0.11` | `0.2010` | `2.22` | `1.10` | `67/100` |
| Query-only r16, skeleton | `100` | `100/100` | `0.23` | `0.23` | `0.4486` | `2.99` | `3.70` | `24/100` |
| Full trajectory r16, skeleton | `100` | `100/100` | `0.16` | `0.17` | `0.3339` | `2.92` | `2.86` | `38/100` |

Query-only r16 skeleton이 full trajectory r16 skeleton보다 더 좋다. Query-only는 100개 중
23개에서 reference-containing evidence를 가져왔고, answer containment도 23개였다. Full r16은
각각 16개, 17개에 머물렀다.

## 5. 추가 통계

### 5.1 Query-only skeleton

| Metric | Value |
| --- | ---: |
| rows with any retrieved message | `78/100` |
| rows with zero retrieved messages | `22/100` |
| retrieved-reference rows | `23/100` |
| answer containment rows | `23/100` |
| retrieved-reference but wrong answer | `3/100` |
| answers beginning with `UNKNOWN` | `24/100` |

Retrieved-message distribution:

| Retrieved messages | Rows |
| ---: | ---: |
| `0` | `22` |
| `1` | `20` |
| `2` | `9` |
| `3` | `6` |
| `4` | `6` |
| `5+` | `37` |

Skeleton은 reference hit를 늘렸지만 distractor도 늘렸다. Mean retrieved가 `1.10`에서 `3.70`으로
증가했고, 정답 evidence가 없는 관련 메시지를 읽고 그럴듯하지만 틀린 답을 낸 row도 많았다.

### 5.2 Correct overlap

Query-only skeleton과 full r16 skeleton의 containment correct row를 비교하면 다음과 같다.

| Category | Count |
| --- | ---: |
| both correct | `15` |
| query-only skeleton only | `8` |
| full r16 skeleton only | `2` |

Query-only skeleton이 추가로 맞힌 row index:

```text
3, 17, 18, 51, 58, 65, 74, 96
```

Full r16 skeleton만 맞힌 row index:

```text
4, 57
```

## 6. 성공 예시

Query-only skeleton은 probe의 indirect clue에서 literal query 후보를 여러 개 뽑는 능력이
tool-call 조건보다 좋아졌다.

| Index | Query chain | Reference | Answer |
| ---: | --- | --- | --- |
| `3` | `part-time job`, `McDonald's`, `working at` | `Burger King!` | `I was working at Burger King.` |
| `17` | `dog`, `walk`, `parents` | `Trooper!` | `Trooper.` |
| `18` | `sons`, `favorite`, `play` | `The attic!` | `My fifth son's favorite play area is the attic.` |
| `51` | `your hobbies`, `dish`, `Paella` | `Paella!` | `Paella is my favorite dish to cook.` |
| `65` | `military service`, `army`, `navy` | `The Navy!` | `I was in the Navy.` |
| `96` | `taco`, `fish tacos`, `fish` | `Fish tacos with cilantro lime slaw.` | `My fiance makes killer fish tacos with cilantro lime slaw.` |

흥미로운 점은 일부 query가 정확한 답 문자열이 아니어도, 주변 literal anchor를 맞히면 recall이
정답 evidence를 가져온다는 것이다. 예를 들어 row `3`에서는 `Burger King`을 직접 생성하지
않았지만 `working at` query가 해당 utterance를 찾아냈다.

## 7. 실패 해석

Skeleton은 query content 능력을 회복하지만 충분하지는 않다.

첫째, 여전히 `77/100` row는 containment 기준으로 틀렸다. Query-only skeleton은 phase-routed
tool-call 조건보다 낫지만 teacher-query hint의 judge `0.6294`나 teacher evidence replay의
`0.8668`과는 거리가 멀다.

둘째, 검색 결과가 너무 넓어질 수 있다. Mean retrieved가 `3.70`이고, 5개 이상 message를
가져온 row가 37개다. 이 경우 answer model은 evidence-only 조건에서도 distractor를 따라갈 수
있다.

셋째, answer model이 evidence를 찾았는데도 틀리는 경우가 남아 있다. Query-only skeleton에서
retrieved-reference but wrong answer가 `3/100`이다. 이 수는 크지 않지만, retrieval 개선 후에는
evidence sufficiency와 answer grounding 문제가 다시 중요해질 것이다.

## 8. 연구적 결론

이번 결과는 매우 중요하다.

```text
Query-only SFT는 실패한 것이 아니라, 잘못된 interface에서 평가되고 있었다.
```

End-to-end Letta agent에서는 query-only adapter가 stop-and-answer policy를 갖지 못해 실패했다.
Phase-routed tool-call 조건에서는 loop는 완료됐지만 tool-call JSON shell 부담 때문에
retrieved-reference rate가 `0.09`에 머물렀다. Deterministic skeleton으로 query string만
생성하게 하자 retrieved-reference rate가 `0.23`까지 올라갔다.

따라서 query-only adapter에는 실제 query-policy signal이 들어 있다. 다만 small model에게
동시에 맡기면 안 되는 것이 세 가지다.

| Surface | 현재 결론 |
| --- | --- |
| tool-call transport | controller/parser가 고정해야 함 |
| stop-and-answer phase transition | controller 또는 별도 policy가 맡아야 함 |
| query content | query-only adapter가 일부 개선하지만 아직 부족함 |

다음 단계는 query string generator를 더 직접적으로 강화하는 것이다. 단순 teacher-query
token imitation보다, 실제 reference-bearing message를 retrieve하는지에 대한 retrieval-supervised
objective가 더 적합하다.

## 9. 다음 실험

1. Teacher query skeleton replay는 완료했다. 자세한 결과는
   `docs/teacher_query_skeleton_report.md`에 정리했다.
2. Query-only skeleton과 teacher skeleton의 query chain을 row-level로 비교한다.
3. Query candidate를 여러 개 생성하고, local recall 결과를 rerank하는 self-consistency 또는
   retrieval-reward 방식을 실험한다.
4. 검색 결과가 너무 넓은 row에서 distractor filtering 또는 evidence sufficiency classifier를
   추가한다.

## 10. Artifacts

```text
scripts/eval_query_skeleton_dmr.py
scripts/eval_teacher_query_skeleton_dmr.py
data/evaluation/query_skeleton_dmr_smoke20/
data/evaluation/query_skeleton_dmr_evidence_only100/
data/evaluation/query_skeleton_dmr_evidence_only500/
data/evaluation/query_skeleton_dmr_r16_smoke20/
data/evaluation/query_skeleton_dmr_r16_100/
data/evaluation/query_skeleton_dmr_base_smoke20/
data/evaluation/teacher_query_skeleton_dmr_approved500_max3/
data/evaluation/query_skeleton_dmr_evidence_only100/nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-skeleton-searches-3-offset-0-limit-100.jsonl
data/evaluation/query_skeleton_dmr_evidence_only100/nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-skeleton-searches-3-offset-0-limit-100.summary.json
data/evaluation/query_skeleton_dmr_r16_100/nano-memgpt-llama3-r16-to-nano-memgpt-llama3-r16-skeleton-searches-3-offset-0-limit-100.jsonl
data/evaluation/query_skeleton_dmr_r16_100/nano-memgpt-llama3-r16-to-nano-memgpt-llama3-r16-skeleton-searches-3-offset-0-limit-100.summary.json
```
