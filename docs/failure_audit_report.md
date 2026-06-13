# Failure Audit Report: Why LoRA Accuracy Stalls

## 1. 목적

Post-LoRA DMR 결과는 두 가지를 동시에 보여 준다.

| 관찰 | 의미 |
| --- | --- |
| r8/r16 모두 GPT-4.1 semantic judge가 약 `0.48` 근처 | LoRA가 최종 답변 능력을 일정 수준 회복함 |
| Oracle replay Llama는 `0.7337` | teacher evidence가 주어지면 훨씬 잘 답함 |

따라서 핵심 질문은 다음이다.

> LoRA를 했는데도 end-to-end 정확도가 낮은 이유는 무엇인가?

이 문서는 r8/r16 post-training 결과를 자동 audit해서, 실패가 query/search 단계에서
나는지, evidence를 찾았는데도 final answer에서 놓치는지, 혹은 tool-call format에서
나는지 분해한다.

## 2. Audit 방법

사용한 artifact는 다음과 같다.

```text
data/evaluation/post_lora_dmr_r8_lenient_v3/
data/evaluation/post_lora_dmr_r16_lenient_v3/
```

분석 스크립트:

```bash
python3 scripts/audit_post_lora_failures.py \
  --run r8:data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.jsonl:data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.judged.gpt41.jsonl \
  --run r16:data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.jsonl:data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.judged.gpt41.jsonl \
  --output-dir data/analysis/post_lora_failure_audit \
  --examples-per-type 10
```

자동 분류는 `raw_messages`의 `conversation_search` call과 `tool_return`을 사용한다.
각 row에 대해 다음을 확인한다.

| Feature | 의미 |
| --- | --- |
| `searched` | `conversation_search`를 호출했는가 |
| `queries` | 실제 검색 query |
| `evidence_contains_reference` | tool output 안에 reference string이 포함되는가 |
| `semantic_correct` | GPT-4.1 judge가 정답으로 인정했는가 |
| `surface_issue` | final answer에 `Thinking`, `Let me search` 같은 실행 의도가 새었는가 |

중요한 caveat가 있다. `evidence_contains_reference`는 exact normalized substring 기준이므로
보수적인 lower bound다. paraphrase evidence나 reference와 의미가 같은 표현은 놓칠 수
있다. 따라서 이 audit은 절대적인 ground-truth classification이 아니라, 다음 실험 방향을
정하기 위한 diagnostic lower-bound다.

## 3. 핵심 결과

| Run | Rows | OK | Judge acc | Lexical acc | Search rate | Evidence-hit rate | Evidence-hit among wrong | Mean searches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r8 | 500 | 489 | 0.4765 | 0.2474 | 0.7280 | 0.1534 | 0.0352 | 1.3701 |
| r16 | 500 | 497 | 0.4809 | 0.2656 | 0.7948 | 0.1489 | 0.0194 | 1.7022 |

해석:

1. r16은 r8보다 search rate와 평균 검색 횟수가 높다.
2. 그런데 exact evidence-hit rate는 둘 다 약 `15%`로 낮다.
3. 틀린 row 중 reference가 검색 output에 exact match로 들어 있는 경우는 매우 적다.

즉, 남은 실패의 큰 축은 **검색 결과를 찾았는데 final answer에서 놓치는 것**보다는
**정답 evidence가 애초에 검색 결과에 들어오지 않는 것**에 가깝다.

## 4. Failure Type 분포

### 4.1 전체 row 기준

| Failure type | r8 | r16 | 의미 |
| --- | ---: | ---: | --- |
| `correct_semantic` | `233` | `239` | GPT-4.1 judge가 정답으로 인정 |
| `no_search` | `77` | `60` | 검색 없이 답하거나 검색 의도만 말함 |
| `searched_wrong_or_insufficient_evidence` | `170` | `193` | 검색했지만 exact reference evidence가 없음 |
| `evidence_found_but_not_used` | `9` | `5` | reference evidence가 있었지만 judge 기준 오답 |
| `tool_call_format_failure` | `11` | `3` | tool-call loop 실패 |

### 4.2 GPT-4.1 judge 오답 row 기준

| Failure type | r8 | r16 |
| --- | ---: | ---: |
| `no_search` | `77` | `60` |
| `searched_wrong_or_insufficient_evidence` | `170` | `193` |
| `evidence_found_but_not_used` | `9` | `5` |

비율로 보면 r16의 judge 오답 `258`개 중 `193`개, 약 `74.8%`가
`searched_wrong_or_insufficient_evidence`다. 이는 r16이 검색은 더 자주 하지만, query가
정답 문자열 또는 정답이 포함된 과거 utterance를 안정적으로 끌어오지 못한다는 뜻이다.

## 5. 예시 해석

### 5.1 r16: 검색했지만 evidence가 부족한 경우

Probe:

```text
Hey, remember that time we talked about music? What was the artist you mentioned you could get into?
```

Reference:

```text
Taylor Swift!
```

r16 query:

```text
music
```

검색 결과에는 probe 자체와 `country music` 관련 문장만 들어왔고, `Taylor Swift`는 들어오지
않았다. 모델은 이후 Chris Stapleton/Jason Aldean을 답했다.

이 row는 final answer generation 문제가 아니라, query가 너무 일반적이어서 정답 memory를
못 가져온 문제에 가깝다.

### 5.2 r16: evidence가 있었지만 못 쓴 경우

Reference:

```text
The drums!
```

r16 query:

```text
What instrument
play
```

검색 결과 중 하나에 `I play the drums`가 있었지만, 모델은 guitar를 답했다. 이런 경우는
`evidence_found_but_not_used`다.

이 유형은 존재하지만 r16 기준 `5`개로 적다. 따라서 현재 가장 큰 병목으로 보기는 어렵다.

### 5.3 r8/r16 공통: 실행 의도 누출

일부 row에서는 final answer가 실제 답변이 아니라 다음과 같은 형태로 끝난다.

```text
Let me check our past conversations...
```

또는

```text
Thinking aloud: ...
```

이런 `reasoning_or_search_intent_leak`는 r8에서 `31`, r16에서 `39`개 관찰되었다.
이는 학습 데이터의 tool-use chain을 imitation하면서 final answer boundary가 아직 완전히
분리되지 않았다는 신호다.

## 6. 연구적 결론

현재까지의 가장 강한 해석은 다음이다.

> LoRA distillation은 MemGPT tool-call contract와 loop stability를 상당히 회복하지만,
> end-to-end DMR 정확도는 query selection 병목 때문에 Oracle replay 수준까지 올라가지
> 못한다.

Oracle replay에서 Llama는 `0.7337` judge accuracy를 냈다. 이는 teacher가 고른 evidence가
주어지면 small model도 최종 답변을 꽤 잘 만들 수 있음을 뜻한다. 반면 end-to-end LoRA는
`0.4765`-`0.4809`에 머문다. 이 차이는 주로 student가 직접 적절한 query를 만들고,
그 query로 exact recall memory를 가져와야 하는 단계에서 생기는 것으로 보인다.

따라서 지금 연구의 다음 단계는 단순히 LoRA rank를 키우는 것이 아니라, query/evidence
단계를 분리해서 학습하고 평가하는 것이다.

## 7. 다음 실험 제안

### 7.1 Teacher-query ablation

Student에게 teacher가 사용한 search query를 주고, 나머지 retrieval/result reading/final
answer를 수행하게 한다.

목적:

```text
query만 좋아지면 end-to-end 성능이 얼마나 회복되는가?
```

이 결과가 크게 오르면 query selection이 주 병목임이 강해진다.

### 7.2 Teacher-result ablation

Student에게 teacher가 얻은 tool result 또는 정답 evidence snippet을 직접 주고 final
answer만 생성하게 한다.

목적:

```text
정답 evidence가 있을 때 answer grounding은 얼마나 가능한가?
```

이 조건은 기존 Teacher-Trace Oracle과 비슷하지만, final answer generation만 더 깨끗하게
분리하는 역할을 한다.

### 7.3 Query SFT

현재 SFT는 full teacher action sequence imitation에 가깝다. 다음에는 query target을 별도
task로 만든다.

```text
input: probe + compact conversation/retrieval instruction
target: teacher conversation_search query
```

기대 효과는 `no_search`와 `searched_wrong_or_insufficient_evidence` 감소다.

### 7.4 Evidence-grounded answer SFT

정답 evidence snippet을 context에 넣고, 짧고 정확한 final answer를 생성하도록 별도 학습한다.

```text
input: probe + retrieved evidence
target: teacher final answer
```

기대 효과는 `evidence_found_but_not_used`와 `reasoning_or_search_intent_leak` 감소다.

## 8. 후속 실험: Query+answer SFT가 실패한 이유

위 제안 중 Query SFT와 Evidence-grounded answer SFT를 빠르게 결합한 실험을 수행했다.
결과는 negative result였다.

| 항목 | 결과 |
| --- | ---: |
| combined SFT step | `1,578` |
| 학습 후 final eval loss | `0.8583` |
| 학습 후 final token accuracy | `0.7859` |
| DMR early attempted rows | `35` |
| DMR completed rows | `1` |
| DMR behavioral failures | `34` |
| 주된 에러 | `No tool calls found in response` |

이 결과는 failure audit의 해석을 더 정교하게 만든다. 기존 audit은 r16 LoRA의 남은 오답이
주로 `searched_wrong_or_insufficient_evidence`라고 보였고, 따라서 query supervision이
필요하다는 결론은 여전히 타당하다. 하지만 query supervision과 answer supervision을 같은
adapter에 단순히 섞는 방식은 실패했다.

가장 그럴듯한 원인은 output surface mismatch다.

| Target type | Letta loop에서 요구되는 출력 |
| --- | --- |
| query step | 반드시 schema-valid `conversation_search` tool call |
| answer step | `send_message` 또는 assistant final answer |

combined SFT는 이 두 target을 같은 adapter에 섞었다. token-level loss는 낮아졌지만,
실제 inference에서는 모델이 tool-call이 필요한 step에서도 자연어 답변 또는 계획 문장을
생성할 가능성이 커졌고, Letta는 이를 `No tool calls found in response`로 거절했다.

따라서 다음 실험은 query selection 병목을 겨냥하되, tool-call shell은 더 강하게 고정해야
한다.

### 8.1 Query-only SFT

`conversation_search` target만 포함한다. answer text는 넣지 않는다. 목적은 다음 둘이다.

```text
1. tool-call contract 안정성을 유지한다.
2. query string 선택만 teacher에 가깝게 만든다.
```

평가 지표는 full DMR 전에 반드시 `20`-row smoke에서 format failure rate를 본다.
format failure가 기존 r16의 `3/500`에 비해 크게 늘면 풀 평가는 진행하지 않는다.

이 실험은 실제로 수행했다. Query-only r16은 `1,178` query-call record 중 `1,125`개를
train, `53`개를 eval로 사용했고, 최종 eval loss `0.8244`, token accuracy `0.7703`을
기록했다. 하지만 DMR smoke에서는 첫 `5/5` row가 모두 behavioral failure로 종료되어 full
evaluation을 중단했다.

겉으로는 query-only도 `tool_call_format_failure`지만, raw provider response는 더 세밀한
그림을 보여 준다. 5개 실패 row 모두 초반에는 정상 `tool_calls` 채널로
`conversation_search`를 여러 번 생성했다. 각 row의 valid tool-call message 수는 `9`, `21`,
`10`, `10`, `7`개였다. 그러나 이후 같은 tool-call JSON을 assistant `content` 문자열로
출력했고, provider는 이를 `tool_calls=[]`, `finish_reason="stop"`으로 반환했다. Letta는
tool call이 필요한 step에서 이 응답을 다음 오류로 거절했다.

```text
No tool calls found in response, model must make a tool call
```

따라서 query-only 실험은 기존 해석을 약간 수정한다. Query-only SFT는 tool-call intent를
일부 학습시켰지만, multi-turn MemGPT loop 전체에서 tool-call channel contract를 안정적으로
유지하지 못했다. 다음 실험은 query-only full DMR이 아니라, JSON-as-content를 tool call로
구제하는 parser-rescue 진단 또는 모델이 query string만 생성하는 deterministic skeleton
조건이 더 적절하다.

추가로 vLLM parser-rescue를 구현해 smoke를 실행했다. `nano_rescue_llama`는 explicit JSON
call에서 schema 밖 noise field를 버리고 단순 타입을 정규화한다. 그러나 6-row smoke에서
`1`개만 완료되고 `5`개가 다시 `tool_call_format_failure`로 끝났다. 즉 vLLM parser layer만으로는
`finish_reason="stop"` content JSON을 충분히 구제하지 못했다. 다음 rescue는 Letta agent
직전 또는 OpenAI-compatible proxy에서 수행해야 한다.

Endpoint proxy-rescue도 수행했다. Proxy는 vLLM과 Letta 사이에서 assistant `content` JSON을
`tool_calls`로 변환했고, 로그상 rescue event가 `70`회 발생했다. 하지만 20-row smoke 결과는
`2/20` loop completion, `18/20` tool-call format failure, containment `0.0`이었다. 완료된
두 row도 정답이 아니라 "I'll search..."류 planning leakage였다.

따라서 query-only의 핵심 병목은 단순 channel failure를 넘어선다. 모델은 계속 검색하는
행동은 강화했지만, 검색을 멈추고 final answer를 보내는 policy를 배우지 못했다. Query-only
adapter는 end-to-end agent가 아니라 search phase 전용 모듈로 해석해야 한다.

### 8.2 Phase-routed query-only diagnostic

위 결론을 확인하기 위해 search phase와 answer phase를 deterministic controller로 분리했다.
Search phase에는 query-only r16 adapter를 쓰고, answer phase에는 full trajectory r16 adapter를
사용했다. Answer prompt에서는 persona와 full history를 제거하고 retrieved evidence만
제공했으며, evidence가 없거나 부족하면 `UNKNOWN`을 답하게 했다.

| Metric | Value |
| --- | ---: |
| rows | `100` |
| completed | `100/100` |
| answer containment | `0.11` |
| retrieved-reference rate | `0.09` |
| mean searches | `2.22` |
| rows with zero retrieved messages | `64/100` |
| `UNKNOWN` answers | `67/100` |

이 결과는 failure 원인을 더 좁힌다. Controller가 stop-and-answer transition을 대신하면
behavioral failure는 사라진다. 하지만 정답 evidence를 검색해 온 비율이 `9%`에 그치므로,
query-only adapter의 남은 병목은 channel보다 query content다. 즉 모델은 `conversation_search`
형태의 행동을 반복할 수 있지만, substring recall에서 정답 utterance를 맞히는 literal query를
아직 충분히 잘 고르지 못한다.

따라서 다음 연구는 query imitation loss를 더 낮추는 것보다, query가 실제 reference-bearing
message를 retrieve하는지 직접 최적화하거나 평가하는 방향이 더 타당하다.

### 8.3 Deterministic query skeleton 결과

위 방향을 구현해 모델이 전체 JSON tool call이 아니라 query string만 생성하도록 만들었다.
Wrapper는 `conversation_search` shell, roles, limit, heartbeat를 고정한다.

| Query generator | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `0.09` | `0.11` | `1.10` |
| Query-only r16, skeleton | `100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `0.16` | `0.17` | `2.86` |

이 결과는 query-only SFT가 완전히 실패한 것이 아니라, end-to-end Letta interface와
tool-call JSON shell에서 성능이 가려졌음을 보여 준다. Query-only adapter를 constrained query
generator로 쓰면 retrieved-reference rate가 `0.09`에서 `0.23`으로 오른다.

동시에 이 결과는 남은 실패도 보여 준다. Query-only skeleton은 더 많은 message를 retrieve하지만,
mean retrieved가 `3.70`으로 늘어 distractor도 증가한다. 따라서 다음 failure audit은 query hit
rate뿐 아니라 distractor filtering과 evidence sufficiency를 함께 봐야 한다.

### 8.4 Teacher query skeleton 비교

Teacher query도 같은 deterministic skeleton으로 replay했다. Search budget을 최대 3회로 맞춘
teacher query는 approved subset과 teacher-search subset 모두에서 query-only skeleton보다 높다.

| Query source | Subset | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher max-3 query | approved | `398` | `0.367` | `0.342` | `3.25` |
| Teacher max-3 query | teacher-search only | `302` | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | same approved | `398` | `0.276` | `0.249` | `3.76` |
| Query-only r16 skeleton | teacher-search subset | `302` | `0.272` | `0.238` | `3.76` |

이 비교는 failure audit의 결론을 더 선명하게 만든다. Query-only adapter는 단순히 검색을 적게
해서 실패하는 것이 아니다. 오히려 teacher max-3보다 평균 retrieved message가 더 많은데도
reference hit가 낮다. 따라서 남은 병목은 recall volume이 아니라 query specificity와 distractor
control이다.

### 8.5 Constrained query generation

모델이 전체 JSON tool call을 생성하지 않고, query string만 생성하도록 만든다. 예를 들어
wrapper가 다음 skeleton을 고정한다.

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

이 조건은 parser/prompt adapter에 가깝고 학습 모듈을 추가하는 것은 아니다. 연구적으로는
"small model이 search query content를 고르는 능력"과 "tool-call JSON shell을 유지하는
능력"을 분리해서 볼 수 있다.

### 8.6 Answer-only evidence grounding

teacher evidence 또는 oracle evidence가 context에 주어진 뒤 final answer만 생성하도록
별도 adapter/prompt를 만든다. 이 조건은 end-to-end query 문제를 제거하고,
`evidence_found_but_not_used`와 final-answer leakage만 본다.

현재 evidence replay가 r16에서 `345/398` judge accuracy를 보였으므로, answer-only
adapter는 큰 성능 향상보다는 답변을 짧고 정확하게 만들고 leakage를 줄이는 보조 역할일
가능성이 높다.

## 9. Artifact

```text
data/analysis/post_lora_failure_audit/summary.md
data/analysis/post_lora_failure_audit/summary.json
data/analysis/post_lora_failure_audit/r8.rows.csv
data/analysis/post_lora_failure_audit/r8.rows.jsonl
data/analysis/post_lora_failure_audit/r8.examples.jsonl
data/analysis/post_lora_failure_audit/r16.rows.csv
data/analysis/post_lora_failure_audit/r16.rows.jsonl
data/analysis/post_lora_failure_audit/r16.examples.jsonl
```
