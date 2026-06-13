# Post-LoRA DMR Evaluation Report

## 1. 목적

이 문서는 GPT-4.1 teacher trajectory로 학습한 Llama-3-8B LoRA adapter가 Vanilla
MemGPT DMR 환경에서 실제로 control loop를 복구하는지 평가한다. 학습 proxy metric은
`docs/lora_training.md`에 기록되어 있으며, 여기서는 end-to-end Letta loop 결과만 다룬다.

핵심 질문은 두 가지다.

| 질문 | 관측 지표 |
| --- | --- |
| 모델이 MemGPT tool-call loop를 안정적으로 통과하는가? | 정상 loop 완료 수, behavioral failure 수 |
| 검색된 기억을 최종 답으로 잘 사용하는가? | deterministic containment, ROUGE-L, GPT-4.1 judge accuracy |

## 2. 평가 조건

| 항목 | 값 |
| --- | --- |
| Base model | `NousResearch/Meta-Llama-3-8B-Instruct` |
| Adapter | `outputs/lora_student_r8/final_adapter/`, `outputs/lora_student_r16/final_adapter/` |
| vLLM exposed id | `nano-memgpt-llama3-r8`, `nano-memgpt-llama3-r16` |
| Letta model handle | `vllm/nano-memgpt-llama3-r8`, `vllm/nano-memgpt-llama3-r16` |
| Parser condition | `parser_lenient_v3` |
| Dataset | MSC DMR offset `0`, limit `500` |
| Judge model | `gpt-4.1-2025-04-14` |

`parser_lenient_v3`는 모델 출력의 top-level `name`, `arguments`, `thinking`,
`request_heartbeat`, `message` 필드를 OpenAI `tool_calls` 형식으로 정규화한다. 이
adapter는 query 내용이나 final answer 내용을 새로 선택하지 않는다.

## 3. 결과

### 3.1 Raw end-to-end summary

| Condition | 정상 완료 | behavioral failure | ROUGE-L recall | containment | search rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA `r=8` | `489/500` | `11` | `0.5489` | `0.2474` | `0.7280` |
| LoRA `r=16` | `497/500` | `3` | `0.5515` | `0.2656` | `0.7948` |

### 3.2 Failure candidate distribution

| Candidate | r8 | r16 | Interpretation |
| --- | ---: | ---: | --- |
| `incorrect_answer` | `368` | `365` | lexical containment 기준 실패 |
| `retrieval_miss_candidate` | `133` | `102` | 검색 호출이 없거나 부족한 후보 |
| `retrieval_hallucination_candidate` | `10` | `9` | 검색 query 또는 tool-call 구조가 의심되는 후보 |
| `tool_call_format_failure` | `11` | `3` | Letta loop를 끝까지 통과하지 못한 후보 |

### 3.3 Semantic judge

| Condition | Judge | Judged rows | Accuracy |
| --- | --- | ---: | ---: |
| LoRA `r=8` | `gpt-4.1-2025-04-14` | `489` | `0.4765` |
| LoRA `r=16` | `gpt-4.1-2025-04-14` | `497` | `0.4809` |
| LoRA `r=16` sanity check | `gpt-4-turbo` | `497` | `0.4789` |

GPT-4.1을 primary judge로 사용한다. `gpt-4-turbo` 결과는 Docker 컨테이너의 stale
environment 값으로 먼저 실행된 sanity check이며, 두 judge가 거의 같은 판정을 냈다는
보조 근거로만 사용한다.

## 4. Baseline과 비교

| Condition | Loop completion | Format failure | Search rate | Containment | Semantic judge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw Vanilla Llama pilot | `0/5` | `5` | n/a | n/a | n/a |
| Strict-template Llama scaled | `481/500` | `19` | `0.8462` | `0.1954` | not run |
| LoRA r8 post-training | `489/500` | `11` | `0.7280` | `0.2474` | `0.4765` |
| LoRA r16 post-training | `497/500` | `3` | `0.7948` | `0.2656` | `0.4809` |

이 비교에서 가장 확실한 개선은 control-loop stability다. 학습 전 Llama는 raw 조건에서
tool-call contract를 통과하지 못했고, strict-template adapter를 붙여도 500행 중 19개가
format failure로 남았다. r8 LoRA는 format failure를 11개로 줄였고, r16 LoRA는 같은
규모에서 3개까지 줄였다.

정답 정확도는 더 조심스럽게 해석해야 한다. deterministic containment는 exact/reference
string 포함 여부를 보기 때문에 보수적이고, GPT-4.1 judge는 paraphrase와 불필요하지만
해롭지 않은 추가 문장을 허용한다. 따라서 `0.2656`과 `0.4809`는 서로 다른 질문에 대한
지표다.

## 5. 해석

이번 결과는 proposal의 가설을 부분적으로 지지한다.

첫째, small model의 주요 병목 중 하나는 단순 지식 부족이 아니라 MemGPT tool-use
policy와 output contract였다. LoRA distillation 이후 loop failure가 크게 줄었다.
`r=16`은 `r=8`보다 `conversation_search` 호출률이 높고 format failure가 적어,
adapter capacity가 operational stability에 영향을 준다는 근거를 제공한다.

둘째, tool-use policy를 복구해도 DMR 정답률이 자동으로 Oracle 수준까지 올라가지는
않았다. GPT-4.1 teacher trajectory Oracle replay에서 Llama는 `292/398` (`0.7337`)
judge accuracy를 기록했지만, 실제 LoRA end-to-end 조건은 `0.4765`-`0.4809`에 머물렀다. 이
차이는 teacher evidence가 직접 주어지는 replay와, student가 직접 query를 만들고 검색
결과를 선택해야 하는 end-to-end 조건의 차이다.

셋째, teacher evidence ablation은 이 해석을 더 강하게 만든다. 같은 LoRA adapter라도
teacher trace evidence를 직접 replay하면 r8은 `343/398` (`0.8618`), r16은 `345/398`
(`0.8668`) GPT-4.1 judge accuracy까지 회복된다. 따라서 post-LoRA end-to-end 실패는
answer-from-evidence 능력 부족보다 query/evidence acquisition 병목에 더 가깝다.

넷째, 다음 병목은 retrieval query supervision과 evidence grounding일 가능성이 높다.
많은 실패가 format failure가 아니라 incorrect answer 또는 retrieval miss 후보로
분류되기 때문이다.

## 6. Artifact

```text
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.jsonl
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.summary.json
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.judged.gpt41.jsonl
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.judged.gpt41.summary.json
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.jsonl
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.summary.json
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.judged.gpt41.jsonl
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.judged.gpt41.summary.json
logs/post_lora_dmr_r16_lenient_v3.log
logs/post_lora_dmr_r16_lenient_v3_judge_gpt41.log
logs/post_lora_dmr_r8_lenient_v3.log
logs/post_lora_dmr_r8_lenient_v3_judge_gpt41.log
```

## 7. Teacher Evidence Ablation

자세한 결과는 `docs/teacher_evidence_ablation_report.md`에 정리했다. 여기서는 end-to-end
LoRA와 teacher evidence replay를 한 표로 비교한다.

| Condition | Evidence source | Student controls search? | Rows | containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: |
| LoRA r8 end-to-end | student retrieval | Yes | `489` completed | `0.2474` | `0.4765` |
| LoRA r8 + teacher trace | teacher evidence | No | `398` approved | `0.4874` | `0.8618` |
| LoRA r16 end-to-end | student retrieval | Yes | `497` completed | `0.2656` | `0.4809` |
| LoRA r16 + teacher trace | teacher evidence | No | `398` approved | `0.4899` | `0.8668` |

이 ablation은 pure Teacher-query condition은 아니다. Teacher가 실제로 얻은 tool output까지
student에게 직접 제공하므로, Letta loop의 query generation과 recall execution을 제거한
answer-from-evidence 진단 조건이다. 그럼에도 end-to-end와의 차이가 매우 크기 때문에,
다음 실험은 query selection과 evidence acquisition을 더 직접적으로 분리해야 한다.

## 8. 다음 단계

1. `docs/failure_audit_report.md`의 자동 audit 결과를 바탕으로 failure sample을 수동
   검산한다.
2. Teacher-query ablation을 만들어 query string selection만 해결했을 때의 회복 폭을
   측정한다.
3. query generation step을 별도 supervised target으로 강화하거나, retrieved evidence를
   final answer에 더 강하게 ground하는 loss/prompt를 실험한다.

## 9. Query+answer SFT negative result

위 3번을 직접 시험하기 위해 query-chain SFT와 evidence-grounded answer SFT를 결합한
`r=16` adapter를 추가로 학습했다. 자세한 학습 설정은 `docs/lora_training.md`의
`Query-chain + evidence-answer SFT 실패 실험` 절에 기록했다.

핵심 결과는 다음과 같다.

| Condition | Proxy metric | End-to-end DMR outcome |
| --- | --- | --- |
| Full trajectory LoRA r16 | eval loss `0.8694`, token acc `0.7599` | `497/500` loop 완료, judge `0.4809` |
| Query+answer LoRA r16 | eval loss `0.8583`, token acc `0.7859` | early DMR `35` rows 중 `34` tool-call format failure |

즉, token-level SFT metric은 좋아졌지만 실제 MemGPT agent loop는 크게 나빠졌다. 실패
에러는 거의 모두 다음 형태였다.

```text
No tool calls found in response, model must make a tool call
```

이 결과는 기존 결론을 더 세밀하게 수정한다. 문제는 단순히 teacher query를 더 많이
학습시키면 해결되는 것이 아니다. Letta/MemGPT에서는 매 step의 출력 surface가
`conversation_search` 같은 tool call이어야 하는데, query target과 answer text target을
같은 adapter에 섞으면 모델이 어떤 step에서 tool-call shell을 유지해야 하는지 혼동할 수
있다.

따라서 다음 실험은 다음 중 하나로 가야 한다.

| 방향 | 기대 효과 |
| --- | --- |
| Query-only adapter | search step에서 JSON tool-call contract를 유지한 채 query string만 개선 |
| Answer-only adapter | teacher evidence가 주어진 조건에서 final answer grounding만 개선 |
| Deterministic tool skeleton | `name`, `arguments` wrapper는 고정하고 모델은 `query` 값만 생성 |
| Adapter routing | search phase와 answer phase에 서로 다른 adapter를 적용 |

이 negative result는 연구적으로 유용하다. small model의 병목이 query selection이라는
기존 해석은 유지되지만, query supervision을 넣는 방법은 agent-loop contract와 분리해서
설계해야 한다는 제약이 추가된다.

## 10. Query-only SFT smoke 결과

Query+answer negative result 이후 answer target을 제거하고 `conversation_search` target만
남긴 query-only LoRA를 학습했다. 학습 결과와 raw response 분석은
`docs/query_only_lora_report.md`에 정리했다.

| Condition | Proxy metric | DMR smoke outcome |
| --- | --- | --- |
| Query+answer LoRA r16 | eval loss `0.8583`, token acc `0.7859` | early DMR `35` rows 중 `34` tool-call format failure |
| Query-only LoRA r16 | eval loss `0.8244`, token acc `0.7703` | first `5/5` rows behavioral failure |

겉으로는 query-only도 실패다. 그러나 failure surface는 query+answer보다 미묘하다. Query-only
adapter는 각 row 초반에 실제 `tool_calls` 채널로 `conversation_search`를 여러 번 생성했다.
5개 실패 row에서 valid tool-call message는 각각 `9`, `21`, `10`, `10`, `7`개였다. 하지만
나중 턴에서 같은 tool-call JSON을 assistant `content` 문자열로 출력했고, provider response는
`tool_calls=[]`, `finish_reason="stop"`이 되었다. Letta는 이 응답을 다음 오류로 거절했다.

```text
No tool calls found in response, model must make a tool call
```

따라서 query-only 결과는 "모델이 도구 호출 의도를 전혀 못 배웠다"가 아니라,
"multi-turn loop에서 tool-call intent를 OpenAI `tool_calls` 채널에 안정적으로 실어 보내지
못한다"로 해석해야 한다.

이 때문에 query-only full 500-row DMR은 진행하지 않는다. 다음 평가는 parser-rescue 또는
deterministic tool skeleton 조건으로 가는 것이 더 정보량이 높다. 특히 assistant `content`
안의 schema-valid JSON을 deterministic하게 `tool_calls`로 변환했을 때 성능이 회복되는지 보면,
channel/transport failure와 query-quality failure를 분리할 수 있다.

### 10.1 vLLM parser-rescue smoke

위 진단을 위해 `vllm_plugins/nano_strict_tool_parser.py`에 `nano_rescue_llama` parser를
추가했다. 이 parser는 모델이 이미 출력한 explicit JSON call에서 schema 밖 noise field를
버리고, 단순 타입만 정규화한다. vLLM을 `STUDENT_TOOL_CALL_PARSER=nano_rescue_llama`로
재시작한 뒤 query-only r16 smoke를 다시 실행했다.

| Metric | Query-only r16 + vLLM rescue |
| --- | ---: |
| attempted rows | `6` |
| completed rows | `1` |
| behavioral failures | `5` |
| tool-call format failures | `5` |
| mean provider attempts per behavioral failure | `12.4` |

결과적으로 vLLM-level rescue만으로는 충분하지 않았다. 일부 row는 정상 완료됐지만, 대부분의
실패 row에서는 `finish_reason="stop"`인 assistant content JSON이 여전히 Letta에
`tool_calls=[]`로 전달되었다. 따라서 다음 rescue는 vLLM parser가 아니라 Letta-side 또는
OpenAI-compatible endpoint proxy에서 수행하는 것이 더 직접적이다.

### 10.2 Endpoint proxy-rescue smoke

다음으로 vLLM과 Letta 사이에 OpenAI-compatible proxy를 두었다. 이 proxy는
`/v1/chat/completions` 응답을 받은 뒤 assistant `content` 안의 schema-valid JSON 또는
bare `{"query": ...}` 객체를 `tool_calls`로 변환한다.

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

Proxy는 실제로 작동했다. 로그에는 `rescued tool call` event가 `70`회 남았다. 하지만 loop
completion은 `2/20`에 그쳤고, 완료된 두 row도 정답이 아니라 planning leakage였다.

따라서 endpoint proxy-rescue의 결론은 분명하다.

```text
query-only SFT는 search action을 계속 생성하는 능력은 강화하지만,
검색을 멈추고 user-facing final answer로 전환하는 policy는 거의 학습하지 못한다.
```

이제 query-only adapter를 end-to-end agent로 평가하는 것은 중단한다. 다음 실험은
search phase와 answer phase를 분리한 adapter routing 또는 deterministic controller 조건으로
가야 한다.

## 11. Phase-routed query-only diagnostic

Endpoint proxy-rescue 결과만으로는 두 문제가 섞여 있었다. 하나는 query-only adapter가
검색을 멈추고 final answer로 전환하지 못하는 문제이고, 다른 하나는 검색 query 자체가
정답 근거를 못 찾는 문제다. 이를 분리하기 위해 Letta loop를 제거하고 deterministic
controller가 search phase와 answer phase를 나누는 진단을 수행했다.

| 항목 | 값 |
| --- | --- |
| Search model | `nano-memgpt-llama3-query-only-r16` |
| Answer model | `nano-memgpt-llama3-r16` |
| Dataset | MSC DMR offset `0`, limit `100` |
| Search execution | local substring recall |
| Answer condition | retrieved evidence only |
| Max searches | `3` |

결과는 다음과 같다.

| Metric | Value |
| --- | ---: |
| completed | `100/100` |
| errors | `0` |
| answer containment | `0.11` |
| retrieved evidence contains reference | `0.09` |
| mean ROUGE-L recall | `0.2010` |
| mean searches | `2.22` |
| mean retrieved messages | `1.10` |
| rows with zero retrieved messages | `64/100` |
| `UNKNOWN` answers | `67/100` |

이 결과는 phase routing이 stop-and-answer failure를 제거한다는 점을 보여 준다. Endpoint
proxy-rescue에서는 `2/20`만 완료되었지만, controller가 phase를 나누면 `100/100`이 정상
완료된다. 그러나 evidence-only 조건에서 retrieved-reference rate가 `0.09`에 그치므로,
query-only adapter의 핵심 한계는 query content quality다. 즉 현재 adapter는 search action을
반복하는 습관은 배웠지만, DMR probe에서 answer-bearing utterance를 찾는 literal query를
충분히 안정적으로 고르지 못한다.

따라서 다음 단계는 query-only adapter를 그대로 agent에 붙이는 것이 아니라 deterministic
tool skeleton에서 query string만 평가하는 것이다. 자세한 분석은
`docs/phase_routed_dmr_report.md`에 정리했다.

## 12. Deterministic query skeleton

Phase-routed 조건에서도 모델은 여전히 `conversation_search` JSON을 생성해야 했다. 다음
진단에서는 모델이 query string만 생성하고, wrapper가 tool shell을 고정했다.

```text
model output: <query string>
wrapper: conversation_search(query=<query string>, roles=["assistant", "user"], limit=10)
```

결과는 다음과 같다.

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `100/100` | `0.09` | `0.11` | `1.10` |
| Query-only r16, skeleton | `100` | `100/100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `100/100` | `0.16` | `0.17` | `2.86` |
| Base Llama skeleton smoke | `20` | `20/20` | `0.20` | `0.20` | `1.75` |

이 결과는 query-only SFT를 재평가하게 만든다. Query-only adapter는 end-to-end agent로는
부적합하지만, deterministic skeleton 안에서는 full trajectory r16보다 좋은 query generator가
된다. 즉 query-only SFT에는 유효한 query-policy signal이 들어 있다. 문제는 그 signal을
tool-call transport와 stop-and-answer policy까지 동시에 맡긴 interface에서 꺼내 쓰려 했다는
점이다.

자세한 분석은 `docs/query_skeleton_dmr_report.md`에 정리했다.

## 13. Teacher query skeleton replay

Deterministic query skeleton의 의미를 더 정확히 보기 위해, GPT-4.1 teacher가 실제로 사용한
`conversation_search` query도 같은 local substring skeleton으로 replay했다. 이 조건은
teacher-query hint ablation과 다르다. Letta agent loop와 prompt hint 효과를 제거하고,
query string 자체가 reference-bearing message를 얼마나 잘 retrieve하는지만 측정한다.

| Query source | Subset | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher full query chain | approved | `398` | `0.442` | `0.405` | `4.36` |
| Teacher full query chain | teacher-search only | `302` | `0.583` | `0.533` | `5.74` |
| Teacher max-3 query | approved | `398` | `0.367` | `0.342` | `3.25` |
| Teacher max-3 query | teacher-search only | `302` | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | same approved | `398` | `0.276` | `0.249` | `3.76` |
| Query-only r16 skeleton | teacher-search subset | `302` | `0.272` | `0.238` | `3.76` |

공정한 search budget 비교에서는 `Teacher max-3 query`를 기준선으로 보는 것이 좋다.
Approved 398 subset에서 query-only skeleton은 teacher max-3 containment의 약 `73%`
(`0.249 / 0.342`)까지 따라온다. 그러나 teacher가 실제로 search를 수행한 302개 subset에서는
약 `53%` (`0.238 / 0.450`)에 머문다. 즉 query-only SFT는 유효한 query-policy signal을
학습했지만, teacher 수준의 literal query selection에는 아직 크게 부족하다.

따라서 다음 학습 방향은 단순 teacher query imitation보다 retrieval-supervised objective가 더
알맞다. Query가 reference-containing message를 retrieve하면 positive로 두고, 결과가 없거나
distractor만 retrieve하면 negative로 두는 ranking 또는 preference-style 학습이 현재 병목에 더
직접적으로 맞다. 자세한 분석은 `docs/teacher_query_skeleton_report.md`에 정리했다.
