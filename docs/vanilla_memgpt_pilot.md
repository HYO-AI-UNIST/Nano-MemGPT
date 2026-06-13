# Vanilla MemGPT DMR 호환성 파일럿

## 1. 실험 목적

이 파일럿은 제안서의 첫 진단 단계다. Oracle tool-call replay나 LoRA 학습에 앞서,
학습하지 않은 소형 instruction model이 Vanilla MemGPT 제어 루프에 들어갈 수 있는지
확인한다. 사용 데이터셋은 `MemGPT/MSC-Self-Instruct`의 DMR split이다.

각 행에서 `scripts/eval_vanilla_dmr.py`는 MSC 과거 세션 1-5를 recall storage에
적재하고 세션 6 probe만 전달한다. 따라서 모델은 과거 사실을 답하기 위해
`conversation_search`를 사용해야 한다.

## 2. 모델 조건

| 표기 | serving checkpoint | precision | quantization | context |
| --- | --- | --- | --- | ---: |
| Llama-3-8B | `NousResearch/Meta-Llama-3-8B-Instruct` | BF16 | 없음 | 8192 |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | BF16 | 없음 | 8192 |

공식 Meta Llama repository는 현재 계정에서 gated 상태이므로 public mirror를 revision
`53346005fb0ef11d3b6a83b12c895cca40156b6c`에 고정했다. Mistral service는 upstream
parallel tool template과 Hugging Face tokenizer를 사용하며 NVIDIA vLLM `0.15.1`
호환성을 위해 config와 weight loading을 자동 처리한다.

## 3. Raw Vanilla 결과

| 모델 | 평가 행 | 완료된 MemGPT loop | 행동 실패 | 행당 provider 시도 |
| --- | ---: | ---: | ---: | ---: |
| Llama-3-8B | 5 | 0 | 5 | 3 |
| Mistral-7B | 5 | 0 | 5 | 3 |

두 모델 모두 유효한 Letta tool call을 내기 전에 멈췄다. Letta는 각 행을 세 번
시도한 뒤 다음 오류로 응답을 거절했다.

```text
No tool calls found in response, model must make a tool call
```

이 결과는 검색 품질 실패가 아니다. 검색을 시도할 수 있는 제어 계약에 아직 진입하지
못한 Type-0 failure다.

### 관찰된 출력 예시

Mistral은 XML과 비슷한 문자열로 `conversation_search`를 흉내 낸 사례가 있었고,
Llama는 recall 접근 의도를 자연어로 서술한 사례가 있었다. 그러나 둘 다 Letta가
실행 가능한 구조화 호출은 아니었다.

```text
자연어 의도: "I should search the conversation history."
필요한 구조: {"name":"conversation_search","arguments":{"query":"Burger King"}}
```

따라서 raw 조건의 ROUGE-L, containment, search rate는 `null`로 남겨 둔다. 거절된
provider text를 답변처럼 채점하면 이 연구가 찾으려는 function-calling 실패를 숨기게
된다.

## 4. Strict Template Adapter 조건

strict adapter는 모델 가중치를 변경하지 않는다. 모델별 tool chat template을
추가하고, 명시적이며 schema-valid인 단일 function call만 OpenAI `tool_calls` 형식으로
변환한다.

adapter가 하지 않는 일은 다음과 같다.

- 어떤 도구를 호출할지 대신 선택하지 않는다.
- 빠진 argument를 복원하지 않는다.
- 자연어 설명에서 호출 의도를 추론하지 않는다.
- 알 수 없는 tool name을 교정하지 않는다.
- 모호한 multi-call 출력을 임의로 하나로 축약하지 않는다.

즉 adapter는 출력 표면 형식을 맞추지만 기억 검색 행동을 학습시키는 모듈은 아니다.

| 모델 | 평가 행 | 완료 loop | ROUGE-L recall | containment | search rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 5 | 5 | 0.6323 | 0.4000 | 0.8000 |
| Mistral-7B | 5 | 0 | n/a | n/a | n/a |

### Llama 해석

Llama에서는 Type-0 format 장벽이 제거되었다. 5개 중 4개 trajectory가
`conversation_search`를 호출했고 2개 답변이 기준 문자열을 포함했다. 남은 오류는
이제 검색어 선택과 답변 구성 차원에서 분석할 수 있다.

`Burger King` 사례는 capture 기반 recall lookup이 end-to-end로 동작함을 보여 준다.

```text
conversation_search(query="Burger King")
-> 과거 대화의 직접 관련 메시지 2개 회수
-> 최종 답변에 "Burger King" 포함
```

### Mistral 해석

Mistral은 단일 tool smoke test는 통과했지만 전체 Letta prompt에서는 5개 행 모두
실패했다. 관찰된 실패는 tool intent 서술, 존재하지 않는 tool name, 필수 schema field
누락, 여러 호출 혼합이다. strict adapter는 이런 출력에 의미를 부여하지 않으므로
실패로 남겨 둔다.

metadata를 교정한 v2 ingestion 규약으로 gate를 다시 실행해도 5개 행 모두
`tool_call_format_failure`였다. rendered recall count는 71-76이었다. 따라서 stale
recall-count metadata가 원인은 아니다.

## 5. 규약 업그레이드

scaled 규약 이름은 `msc_dmr_recall_capture_reset_recompile_v2`다. fixed history를
capture한 뒤 Letta의 `reset-messages` endpoint를 호출하고 prompt를 다시 compile한다.
저장한 메시지는 검색 가능하게 유지하면서 즉시 문맥에서는 제거하고, rendered recall
count는 실제 저장량과 일치시킨다.

평가기는 매 행 checkpoint하고 infrastructure-only failure만 새 agent로 재시도한다.
행동 실패는 수정하거나 숨기지 않는다.

## 6. 산출물

- `data/evaluation/vanilla_dmr/vllm-nousresearch-meta-llama-3-8b-instruct-offset-0-limit-5.jsonl`
- `data/evaluation/vanilla_dmr/vllm-mistralai-mistral-7b-instruct-v0-3-offset-0-limit-5.jsonl`
- `data/evaluation/template_aligned_dmr/vllm-nousresearch-meta-llama-3-8b-instruct-offset-0-limit-5.jsonl`
- `data/evaluation/template_aligned_dmr/vllm-mistralai-mistral-7b-instruct-v0-3-offset-0-limit-5.jsonl`
- `data/evaluation/experiment_1/dmr_mistral_strict/vllm-mistralai-mistral-7b-instruct-v0-3-offset-0-limit-5.jsonl`
- 각 디렉터리의 `.summary.json`
- `configs/vanilla_models.yaml`
- `vllm_plugins/nano_strict_tool_parser.py`

## 7. 이 파일럿으로 말할 수 있는 것

- 두 raw 모델은 현재 parser 조건에서 Vanilla MemGPT loop에 직접 진입하지 못한다.
- Llama는 제한적인 형식 adapter만으로 검색 행동 분석이 가능한 단계까지 들어간다.
- Mistral은 같은 adapter 이후에도 제어 계약 실패가 남는다.

아직 말할 수 없는 것은 다음과 같다.

- 소형 모델의 지식 자체가 부족하다고 결론 내릴 수 없다.
- Llama strict-template 조건이 paper-faithful teacher 조건과 동일하다고 볼 수 없다.
- Mistral에 LoRA가 효과가 없다고 결론 내릴 수 없다.

다음 단계에서는 GPT-4.1 teacher trajectory를 수집하고, 동일한 memory history에서
teacher trace를 두 student에 replay하여 검색 행동과 답변 구성 실패를 더 직접적으로
분리한다.
