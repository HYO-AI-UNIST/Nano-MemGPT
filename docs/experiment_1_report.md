# Experiment 1 보고서: Vanilla MemGPT Baseline

## 1. 질문과 범위

Experiment 1은 Oracle replay와 학습 이전의 기준선을 측정한다. 핵심 질문은 “작은
instruction model이 Vanilla MemGPT 환경에서 장기 기억 도구를 사용할 수 있는가”다.

평가 범위는 두 부분으로 나뉜다.

| 평가 | 목적 | 현재 지위 |
| --- | --- | --- |
| MSC Deep Memory Retrieval(DMR) | 긴 대화 기록에서 개인적 사실 검색 | 핵심 baseline |
| NaturalQuestions-Open Document-QA | 문서 검색 문맥에서 답변 구성 평가 | 로컬 context-pack proxy |

로컬 serving checkpoint는 quantization 없는 BF16이다.

| 표기 | checkpoint | context |
| --- | --- | ---: |
| Llama-3-8B | `NousResearch/Meta-Llama-3-8B-Instruct` | 8192 |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | 8192 |

공식 Meta checkpoint는 현재 환경에서 gated 상태다. Llama mirror revision은
`configs/vanilla_models.yaml`에 고정했다.

## 2. DMR 실행 규약

`scripts/eval_vanilla_dmr.py`는 MSC 행마다 새 `memgpt_agent`를 만든다. 세션 1-5를
recall storage에 capture하고, `reset-messages` endpoint를 호출하며, system prompt를
다시 compile한 다음 세션 6 probe만 전달한다.

이 방식은 과거 메시지를 검색 가능하게 유지하면서 즉시 문맥에서는 제거한다. 검증
행에서는 36개 교환을 capture한 뒤 72개의 recall message가 정상적으로 표시되었다.
평가기는 매 행 checkpoint, infrastructure-only retry, JSONL resume, 독립 행 worker를
지원한다.

세부 규약은 `docs/vanilla_dmr_protocol.md`에 기록했다.

## 3. DMR 호환성 파일럿

### 3.1 Raw Vanilla

| 모델 | 평가 행 | 완료 loop | 행동 실패 |
| --- | ---: | ---: | ---: |
| Llama-3-8B | 5 | 0 | 5 |
| Mistral-7B | 5 | 0 | 5 |

두 raw serving 조건은 유효한 Letta tool call을 만들기 전에 실패했다. 따라서 이
결과는 retrieval-quality 점수가 아니라 control-contract failure baseline이다.

### 3.2 Strict template alignment

strict adapter는 모델 가중치를 고정한 채 명시적이며 schema-valid인 단일 호출만
OpenAI `tool_calls` 형식으로 변환한다. 의도를 추론하거나 빠진 argument를 복구하지
않는다.

| 모델 | 평가 행 | 완료 loop | ROUGE-L recall | containment | search rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 5 | 5 | 0.6323 | 0.4000 | 0.8000 |
| Mistral-7B | 5 | 0 | n/a | n/a | n/a |

Mistral gate는 metadata-fixed v2 ingestion 규약으로 다시 확인했다.

| 평가 행 | 완료 loop | 행동 실패 | rendered recall count |
| ---: | ---: | ---: | --- |
| 5 | 0 | 5 | 71-76 |

5개 행 모두 `tool_call_format_failure`였다. stale recall-count metadata가 Mistral
strict-gate 실패 원인은 아니다.

산출물:
`data/evaluation/experiment_1/dmr_mistral_strict/vllm-mistralai-mistral-7b-instruct-v0-3-offset-0-limit-5.jsonl`

## 4. Scaled Strict-Template Llama DMR

이 500행 실행은 학습 없는 strict-template adapter 조건이다. raw Vanilla 조건은 이미
0/5 control-contract gate에서 멈췄으므로 별도 조건으로 보존한다.

| 평가 행 | 완료 loop | 행동 실패 | infrastructure error |
| ---: | ---: | ---: | ---: |
| 500 | 481 | 19 | 0 |

| ROUGE-L recall | containment | search rate |
| ---: | ---: | ---: |
| 0.3929 | 0.1954 | 0.8462 |

| 진단 후보 | 건수 | 의미 |
| --- | ---: | --- |
| `incorrect_answer` | 387 | deterministic containment proxy 실패 |
| `retrieval_miss_candidate` | 74 | `conversation_search` 호출 없음 |
| `retrieval_hallucination_candidate` | 8 | query 또는 tool-call 구조 이상 |
| `tool_call_format_failure` | 19 | Letta loop 진입 실패 |

rendered recall count는 63-78 범위다. 높은 search rate와 낮은 containment의 조합은
형식 정렬만으로 Type-0 장벽 일부를 제거할 수 있지만 query selection, tool chaining,
answer construction은 해결되지 않음을 보여 준다.

산출물:
`data/evaluation/experiment_1/dmr/vllm-nousresearch-meta-llama-3-8b-instruct-offset-0-limit-500.jsonl`

### 해석 시 주의점

이 scaled artifact는 현재의 `paper_substring` 감사와 teacher 수집보다 먼저 실행한
strict-template baseline이다. 이후 감사에서 Letta 도구 설명과 실제 SQL substring
실행 경로의 불일치를 발견하고 teacher 조건을 교정했다. 따라서 이 수치를 교정된
GPT-4.1 teacher 결과와 완전히 같은 조건의 정면 비교로 사용하지 않는다.

Experiment 1 artifact의 행 단위 LLM judge label은 아직 별도로 생성하지 않았다.
현재 API key는 구성되어 있으므로 teacher 수집을 방해하지 않는 시점에
`scripts/judge_dmr_answers.py`로 후처리할 수 있다.

## 5. Document-QA Context-Pack Proxy

제안서의 paper-faithful Document-QA 조건은 PostgreSQL과 pgvector HNSW index에 적재한
Wikipedia 20M passage embedding이 필요하다. 2026-06-01 기준 참조된 공개 Hugging
Face repository [`MemGPT/wikipedia_embeddings`](https://huggingface.co/datasets/MemGPT/wikipedia_embeddings)는
`.gitattributes`만 포함하며 저장된 데이터 크기가 0 byte다. 원본 MemGPT와 Letta
clone에서도 대체 artifact 경로를 확인하지 못했다.

로컬 proxy는 `MemGPT/qa_data` 앞 50개 질문을 사용한다. 이 수치는 재현 가능한 진단
결과지만 paper-faithful archival retrieval 결과로 보고하지 않는다.

### 5.1 Llama-3-8B

| mode | 요청 K | 유효 K | exact match | containment |
| --- | ---: | ---: | ---: | ---: |
| `gold_plus_dpr` | 5 | 5 | 0.42 | 0.46 |
| `gold_plus_dpr` | 10 | 10 | 0.38 | 0.46 |
| `gold_plus_dpr` | 20 | 20 | 0.36 | 0.48 |
| `gold_plus_dpr` | 40 | 30 | 0.32 | 0.50 |
| `dpr_only` | 5 | 5 | 0.04 | 0.04 |
| `dpr_only` | 10 | 10 | 0.06 | 0.06 |
| `dpr_only` | 20 | 20 | 0.08 | 0.16 |
| `dpr_only` | 40 | 29 | 0.04 | 0.08 |

산출물:
`data/evaluation/experiment_1/document_qa_proxy/nousresearch-meta-llama-3-8b-instruct-offset-0-limit-50.jsonl`

### 5.2 Mistral-7B

| mode | 요청 K | 유효 K | exact match | containment |
| --- | ---: | ---: | ---: | ---: |
| `gold_plus_dpr` | 5 | 5 | 0.28 | 0.58 |
| `gold_plus_dpr` | 10 | 10 | 0.26 | 0.58 |
| `gold_plus_dpr` | 20 | 20 | 0.38 | 0.60 |
| `gold_plus_dpr` | 40 | 30 | 0.36 | 0.54 |
| `dpr_only` | 5 | 5 | 0.02 | 0.10 |
| `dpr_only` | 10 | 10 | 0.04 | 0.12 |
| `dpr_only` | 20 | 20 | 0.00 | 0.06 |
| `dpr_only` | 40 | 29 | 0.02 | 0.04 |

산출물:
`data/evaluation/experiment_1/document_qa_proxy_mistral/mistralai-mistral-7b-instruct-v0-3-offset-0-limit-50.jsonl`

### 5.3 Proxy가 보여 주는 것

- `gold_plus_dpr`는 답변 근거가 문맥에 있을 때 answer construction 능력을 본다.
- `dpr_only`는 제한된 로컬 후보 집합에서 retrieval과 answer construction이 함께
  필요할 때 성능이 크게 낮아짐을 보여 준다.
- paper-faithful Wikipedia index가 없으므로 원 논문의 archival search 수치와 직접
  비교할 수 없다.

상세 경계와 실행 명령은 `docs/document_qa_proxy.md`에 정리했다.

## 6. 재현 정보

| 항목 | 값 |
| --- | --- |
| Proposal SHA-256 | `f32bdde0eda47ef57c3e6995fb4e27e938e708832b562b049c3602a4a5159206` |
| MemGPT/Letta source commit | `1131535716e8a31c9a437f8695e25ac98f203a24` |
| vLLM source commit | `7b546902447c695c3a555a81352719710d4f1783` |
| NVIDIA vLLM image | `nvcr.io/nvidia/vllm:26.02-py3` |

## 7. 결론

Experiment 1은 두 종류의 병목을 분리했다.

1. raw 소형 모델은 우선 tool-call 제어 계약에서 실패한다.
2. Llama는 엄격한 표면 형식 adapter로 loop에 들어가지만, 500행에서 낮은
   containment를 보인다.
3. Mistral은 같은 adapter 조건에서도 format failure가 남는다.

따라서 다음 단계는 강한 teacher의 정상 trajectory를 수집하고, 같은 memory
history에서 student에게 teacher trace를 replay하여 “검색 행동 부족”과 “최종 답변
구성 부족”을 분리하는 것이다.
