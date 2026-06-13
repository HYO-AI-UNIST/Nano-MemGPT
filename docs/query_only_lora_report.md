# Query-Only LoRA 진단 보고서

## 1. 목적

Query+answer SFT는 token-level proxy metric은 좋아졌지만, Letta end-to-end loop에서는
tool-call format failure를 크게 늘렸다. 그 결과는 query target과 answer target을 같은
adapter에 섞으면 MemGPT agent가 요구하는 출력 surface가 흐려질 수 있음을 보여 주었다.

이번 실험은 그 문제를 한 단계 더 분리한다. Answer target을 모두 제거하고,
`conversation_search` call만 target으로 둔 query-only LoRA를 학습했다.

핵심 질문은 다음이다.

```text
answer target을 제거하면 tool-call shell 안정성을 보존하면서 query policy만 개선할 수 있는가?
```

## 2. 학습 설정

| 항목 | 값 |
| --- | ---: |
| Base model | `NousResearch/Meta-Llama-3-8B-Instruct` |
| Dataset | `data/trajectories/gpt41_paper_substring_scaled_query_chain_sft.jsonl` |
| Target action | `conversation_search` only |
| LoRA rank / alpha | `r=16`, `alpha=32` |
| max length | `8192` |
| loss | `chunked_nll` |
| output | `outputs/lora_query_only_r16/final_adapter/` |

학습 데이터는 query-chain SFT `1,180` step 중 `8,192` token을 넘는 2개 step을 제외한
`1,178` record로 구성되었다.

| 항목 | 값 |
| --- | ---: |
| train records | `1,125` |
| eval records | `53` |
| dropped overlength | `2` |
| optimizer step | `213/213` |
| epoch | `3` |
| train runtime | 약 `1시간 45분 59초` |
| train loss | `0.9618` |
| final eval loss | `0.8244` |
| final eval mean token accuracy | `0.7703` |
| adapter size | 약 `53M` |

Proxy metric만 보면 query-only는 나쁘지 않다. 기존 full-trajectory r16의 eval loss
`0.8694`보다 낮고, query+answer r16의 eval loss `0.8583`보다도 낮다. 하지만 이 수치는
teacher query-call target에 대한 token-level 지표일 뿐이며, 실제 MemGPT loop 통과를
보장하지 않는다.

## 3. Serving 및 등록

vLLM은 다음 adapter id를 노출했다.

```text
nano-memgpt-llama3-query-only-r16
```

Letta provider registry에는 다음 handle로 등록했다.

```text
vllm/nano-memgpt-llama3-query-only-r16
```

## 4. DMR Smoke 결과

Full 500-row 평가 전에 20-row smoke를 시작했다. 첫 5개 row가 모두 behavioral failure로
끝났기 때문에, full evaluation은 중단했다.

| Metric | Value |
| --- | ---: |
| attempted rows | `5` |
| completed rows | `0` |
| behavioral failures | `5` |
| failure candidate | `tool_call_format_failure: 5` |
| mean provider attempts per failure | `14.4` |

대표 에러는 다음과 같다.

```text
No tool calls found in response, model must make a tool call
```

## 5. 중요한 관찰: 완전한 no-tool 실패는 아니다

겉으로 보면 query-only도 `5/5` tool-call format failure로 실패했다. 그러나 raw provider
response를 보면 query+answer 실패와는 조금 다르다.

각 실패 row에서 모델은 초반에 실제 OpenAI `tool_calls` 채널로 `conversation_search`를 여러
번 생성했다.

| row | valid `tool_calls` message | JSON-as-content message |
| ---: | ---: | ---: |
| `0` | `9` | `3` |
| `1` | `21` | `3` |
| `2` | `10` | `3` |
| `3` | `10` | `3` |
| `4` | `7` | `3` |

문제가 되는 순간은 모델이 아래와 같은 JSON을 assistant `content` 문자열로 출력할 때다.

```json
{
  "arguments": {
    "query": "artist",
    "roles": ["assistant"],
    "limit": 20
  },
  "name": "conversation_search",
  "request_heartbeat": true
}
```

의미적으로는 tool call에 가깝지만, provider response에서는 `tool_calls=[]`,
`finish_reason="stop"`으로 들어온다. Letta는 tool call이 필요한 step에서 이 출력을
받으면 즉시 `No tool calls found`로 거절한다.

따라서 query-only의 실패는 다음처럼 해석하는 것이 정확하다.

```text
모델이 tool-call 의도를 전혀 못 배운 것은 아니다.
다만 multi-turn Letta loop에서 tool call을 반드시 tool_calls 채널로 유지하는 안정성이 부족하다.
```

## 6. 해석

Query-only SFT는 query+answer SFT보다 좋은 방향으로 움직인 부분이 있다. Query+answer는
초반부터 자연어/answer surface로 무너지는 경향이 강했지만, query-only는 실제 tool call을
여러 번 만든다. 그러나 반복 검색이 길어지면 같은 JSON 구조를 일반 assistant content로
출력하기 시작한다.

이 결과는 연구 결론을 더 좁혀 준다.

1. 단순 token-level SFT loss 개선은 MemGPT loop 성공을 보장하지 않는다.
2. Query target만 남겨도 OpenAI tool-call 채널 계약은 안정적으로 고정되지 않는다.
3. 남은 문제는 query content 학습과 tool-call transport/channel control이 섞여 있다.
4. Full 500-row query-only DMR은 현재 조건에서는 의미가 낮아 중단하는 것이 맞다.

## 7. vLLM parser-rescue smoke

위 해석을 검증하기 위해 `nano_rescue_llama` parser를 추가했다. 이 parser는 모델이 이미 출력한
명시적 JSON tool-call intent에서 schema 밖 noise field를 제거하고, `roles`, `limit`,
`request_heartbeat` 같은 단순 transport-level 타입만 정규화한다. Query나 answer를 새로
고르지는 않는다.

수정 파일:

```text
vllm_plugins/nano_strict_tool_parser.py
```

vLLM은 다음 설정으로 재시작했다.

```text
STUDENT_TOOL_CALL_PARSER=nano_rescue_llama
```

그 뒤 query-only r16 adapter로 20-row smoke를 시작했지만, 처음 6개 row 중 5개가 다시
behavioral failure로 종료되어 중단했다.

| Metric | Value |
| --- | ---: |
| attempted rows | `6` |
| completed rows | `1` |
| behavioral failures | `5` |
| failure candidate | `tool_call_format_failure: 5` |
| mean provider attempts per failure | `12.4` |
| completed row containment | `1/1` |

이 결과는 vLLM parser-rescue만으로는 충분하지 않다는 뜻이다. 일부 row는 정상 완료되었지만,
대부분의 실패 row에서는 여전히 provider response가 `tool_calls=[]`, `finish_reason="stop"`인
assistant content로 Letta에 전달되었다. 즉 vLLM parser가 모든 stop-content JSON을
tool-call extraction path로 보내지는 못한다.

실패 content 예시는 다음과 같다.

```json
{
  "arguments": {
    "query": "save money",
    "roles": ["assistant", "user"],
    "limit": 10
  },
  "name": "conversation_search",
  "request_heartbeat": true
}
```

이 JSON은 의미적으로 rescue 가능하지만 Letta agent에 도달할 때는 여전히 assistant content로
남아 있었다. 따라서 다음 diagnostic은 vLLM parser가 아니라 Letta-side 또는 endpoint-proxy
rescue가 되어야 한다.

## 8. Endpoint proxy-rescue smoke

vLLM parser-rescue가 부족했기 때문에, vLLM과 Letta 사이에 OpenAI-compatible proxy를 추가로
구현했다. 이 proxy는 `/v1/chat/completions` 응답을 받은 뒤, assistant `content` 안의
schema-valid JSON 또는 bare `{"query": ...}` 객체를 OpenAI `tool_calls` 형식으로 변환한다.
모델이 만들지 않은 query를 새로 선택하지는 않는다.

수정/추가 파일:

```text
scripts/vllm_tool_rescue_proxy.py
```

실험 중 vLLM provider URL은 다음처럼 임시 변경했다.

```text
from: http://llama-vllm:8000/v1
to:   http://nano-memgpt-dev:8002/v1
```

20-row smoke 결과는 다음과 같다.

| Metric | Query-only r16 + endpoint proxy rescue |
| --- | ---: |
| attempted rows | `20` |
| completed rows | `2` |
| behavioral failures | `18` |
| tool-call format failures | `18` |
| containment | `0.0` |
| ROUGE-L recall | `0.0` |
| search rate among completed rows | `1.0` |
| mean provider attempts per behavioral failure | `24.56` |

Proxy는 실제로 작동했다. 로그 기준 `rescued tool call`이 반복적으로 기록되었고, 전체 proxy
로그에는 rescue event가 `70`회 남았다. 즉 많은 JSON-as-content 출력이 tool call로 변환되었다.
그럼에도 loop completion은 `2/20`에 머물렀다.

완료된 두 row도 정답이 아니었다. 두 답변 모두 아래와 같은 planning leakage였다.

```text
I'll search for the user's previous messages mentioning music or a favorite artist...
I've checked all recent messages for the fast food place; I'll try searching for "work"...
```

따라서 endpoint proxy-rescue 결과는 더 강한 결론을 준다.

```text
query-only SFT의 남은 문제는 단순 channel failure만이 아니다.
channel을 상당 부분 구제해도 모델은 검색을 멈추고 답하는 policy를 거의 갖지 못한다.
```

이 결과는 query-only adapter를 end-to-end agent로 쓰면 안 된다는 뜻이다. Query-only adapter는
search phase 전용 모듈로만 의미가 있다. 실제 agent에서는 search policy와 answer policy를
분리하거나, deterministic controller가 search phase를 끝낼 조건을 정해야 한다.

가장 타당한 다음 단계는 parser-rescue 또는 constrained tool skeleton 진단이다.

| 방향 | 목적 |
| --- | --- |
| Phase-routed adapter | search phase에는 query-only adapter, answer phase에는 full LoRA/base/answer adapter 사용 |
| Deterministic tool skeleton | 모델은 `query` string만 생성하고 wrapper가 `conversation_search` shell을 고정 |
| Query quality audit under rescue | channel을 복구했을 때 query가 실제 evidence를 찾는지 별도 측정 |
| Answer-only evidence grounding | retrieval이 안정화된 뒤 final answer boundary를 별도 학습 |

Rescue는 학습 모듈을 추가하는 것이 아니다. 모델 출력의 의미적 intent가 이미
`conversation_search` JSON에 가까운지 확인하는 diagnostic adapter다. vLLM-level rescue는
불충분했고, endpoint proxy rescue도 end-to-end 성능을 회복하지 못했다. 따라서 이제는
channel보다 `when to stop searching and answer`를 분리해서 다루는 것이 더 중요하다.

## 9. 다음 실험 방향

### 9.1 Phase-routed diagnostic 결과

Query-only adapter를 search phase 전용 모듈로 해석할 수 있는지 보기 위해, Letta loop를
우회한 phase-routed diagnostic을 실행했다. Controller가 search phase와 answer phase를 나누고,
query-only r16은 `conversation_search` query만 생성한다. 검색은 local substring recall로
실행하고, answer phase에는 full trajectory r16 adapter를 사용했다. Answer prompt에는 retrieved
evidence만 넣었고, evidence가 부족하면 `UNKNOWN`을 답하게 했다.

| Metric | Value |
| --- | ---: |
| rows | `100` |
| completed | `100/100` |
| answer containment | `0.11` |
| retrieved evidence contains reference | `0.09` |
| mean searches | `2.22` |
| mean retrieved messages | `1.10` |
| rows with zero retrieved messages | `64/100` |
| `UNKNOWN` answers | `67/100` |

이 결과는 두 가지를 동시에 보여 준다.

첫째, query-only adapter의 stop-and-answer failure는 controller로 제거할 수 있다. Endpoint
proxy-rescue에서는 `2/20`만 완료됐지만, phase-routed 조건에서는 `100/100`이 완료됐다.

둘째, loop를 안정화해도 query-only search policy의 evidence hit rate는 낮다. Retrieved
evidence가 reference를 포함한 row는 `9/100`뿐이었다. 따라서 query-only adapter는 지금
상태로는 성능 개선 모듈이라기보다 query-quality diagnostic으로 보는 것이 맞다.

자세한 row-level 분석은 `docs/phase_routed_dmr_report.md`에 기록했다.

### 9.2 다음 방향

이제 가장 중요한 후속 조건은 deterministic tool skeleton이다. 모델은 query string만 만들고,
wrapper가 `conversation_search` shell을 고정한다. 그 뒤 teacher query, query-only LoRA query,
그리고 skeleton query가 각각 얼마나 reference-bearing message를 retrieve하는지 직접 비교한다.

### 9.3 Deterministic query skeleton 결과

위 조건을 구현해 실행했다. 자세한 분석은 `docs/query_skeleton_dmr_report.md`에 정리했다.

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `100/100` | `0.09` | `0.11` | `1.10` |
| Query-only r16, skeleton | `100` | `100/100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `100/100` | `0.16` | `0.17` | `2.86` |
| Base Llama skeleton smoke | `20` | `20/20` | `0.20` | `0.20` | `1.75` |

이 결과는 query-only SFT가 완전한 실패가 아니라는 점을 보여 준다. End-to-end Letta loop에서는
stop-and-answer policy가 없어서 실패했고, tool-call phase-routed 조건에서는 JSON shell 부담
때문에 query 품질이 낮았다. 하지만 skeleton으로 query string만 생성하게 하자
retrieved-reference rate가 `0.23`까지 올랐다.

따라서 query-only adapter는 agent 전체가 아니라 constrained query generator로 쓰는 것이 맞다.
이후 teacher query skeleton replay를 실행해 query upper reference와 row-level gap을 확인했다.

### 9.4 Teacher query skeleton 비교

같은 deterministic skeleton에서 GPT-4.1 teacher query를 replay하면, search budget을 최대 3회로
제한해도 query-only skeleton보다 명확히 높은 retrieval hit를 보인다.

| Query source | Subset | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher max-3 query | approved | `398` | `0.367` | `0.342` | `3.25` |
| Teacher max-3 query | teacher-search only | `302` | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | same approved | `398` | `0.276` | `0.249` | `3.76` |
| Query-only r16 skeleton | teacher-search subset | `302` | `0.272` | `0.238` | `3.76` |

이 비교에서 query-only r16은 approved subset 기준으로 teacher max-3 containment의 약 `73%`를
따라오지만, teacher-search subset에서는 약 `53%`에 머문다. 즉 query-only SFT는 query-policy
signal을 일부 학습했지만, indirect DMR probe에서 teacher처럼 answer-bearing literal을 고르는
수준에는 아직 부족하다.

자세한 분석은 `docs/teacher_query_skeleton_report.md`에 정리했다.

## 10. Artifacts

```text
outputs/lora_query_only_r16/final_adapter/
logs/lora_query_only_r16.log
scripts/eval_phase_routed_dmr.py
scripts/eval_query_skeleton_dmr.py
scripts/eval_teacher_query_skeleton_dmr.py
data/evaluation/post_lora_dmr_query_only_r16_smoke/vllm-nano-memgpt-llama3-query-only-r16-offset-0-limit-20.jsonl
data/evaluation/post_lora_dmr_query_only_r16_smoke/vllm-nano-memgpt-llama3-query-only-r16-offset-0-limit-20.summary.json
data/evaluation/post_lora_dmr_query_only_r16_rescue_smoke/vllm-nano-memgpt-llama3-query-only-r16-offset-0-limit-20.jsonl
data/evaluation/post_lora_dmr_query_only_r16_rescue_smoke/vllm-nano-memgpt-llama3-query-only-r16-offset-0-limit-20.summary.json
data/evaluation/post_lora_dmr_query_only_r16_proxy_rescue_smoke/vllm-nano-memgpt-llama3-query-only-r16-offset-0-limit-20.jsonl
data/evaluation/post_lora_dmr_query_only_r16_proxy_rescue_smoke/vllm-nano-memgpt-llama3-query-only-r16-offset-0-limit-20.summary.json
data/evaluation/phase_routed_dmr_evidence_only100/
data/evaluation/query_skeleton_dmr_evidence_only100/
data/evaluation/query_skeleton_dmr_r16_100/
logs/vllm_tool_rescue_proxy.log
```
