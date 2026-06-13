# 작은 오픈소스 LLM의 MemGPT Function-Calling 실패 진단과 Teacher Trajectory Distillation을 통한 복구

> Working paper draft. 이 문서는 나중에 영어 LaTeX 논문으로 옮기기 전, 현재 연구의 논리와 실험 결과를 한국어로 정리한 초안이다. 현재 포함된 결과는 DMR baseline, GPT-4.1 teacher trajectory 수집, Teacher-Trace Oracle replay, LoRA distillation, post-LoRA failure audit, LoRA teacher-evidence ablation, LoRA teacher-query hint ablation, query+answer SFT, query-only SFT negative result, phase-routed query-only diagnostic, deterministic query skeleton diagnostic, teacher query skeleton replay, hard-query SFT/DPO, candidate query reranking, evidence filtering final diagnostic이다.

## Abstract

MemGPT 계열 memory system은 LLM의 제한된 context window를 외부 memory tool로 확장한다. 중요한 점은 retrieval이 항상 고정된 retriever에 의해 자동 수행되는 것이 아니라, LLM 자체가 memory-management tool을 호출하고, 검색 query를 고르고, 검색 결과를 다시 active context로 가져올지 결정한다는 것이다. 따라서 MemGPT가 잘 작동하려면 underlying model이 단순히 답변을 잘 생성하는 것만으로는 부족하다. 모델은 유효한 structured function call을 만들고, 적절한 memory query를 선택하며, 필요하면 heartbeat를 통해 여러 tool call을 chain하고, 최종 답변을 retrieved evidence에 ground해야 한다.

본 연구는 작은 open-source instruction-tuned LLM이 MemGPT Deep Memory Retrieval(DMR) 환경에서 왜 실패하는지를 진단하고, GPT teacher trajectory distillation으로 이 실패를 얼마나 복구할 수 있는지 실험한다. 로컬 Letta/MemGPT-style DMR protocol을 구축하고, Llama-3-8B와 Mistral-7B를 vLLM BF16 serving 조건에서 평가했다. Raw vanilla 조건에서는 두 모델 모두 control-contract gate를 통과하지 못했다. Llama-3-8B와 Mistral-7B 모두 5-row pilot에서 `0/5` DMR tool-use loop completion을 기록했다. 즉 초기 실패는 단순 정답률 문제가 아니라, MemGPT tool-call loop에 진입하지 못하는 문제였다.

이후 strict deterministic parser adapter를 붙이면 Llama는 loop에 들어갈 수 있었다. 그러나 500-row DMR에서 Llama strict-template 조건은 `481/500` loop completion, containment `0.1954`에 머물렀다. 이는 tool-call 표면 형식을 고치는 것만으로는 retrieval behavior가 충분히 회복되지 않는다는 것을 보여 준다.

지식 부족과 memory-control 실패를 분리하기 위해, GPT-4.1 teacher가 같은 DMR 환경에서 생성한 trajectory를 수집하고 judge가 승인한 teacher evidence를 frozen student에게 replay했다. GPT-4.1 teacher는 500개 DMR row 중 `398/500`을 judge 기준으로 맞혔다. 이 승인된 teacher evidence를 replay하면 Llama-3-8B는 `292/398` (`0.7337`) GPT-4.1 judge accuracy를 기록했고, Mistral-7B는 `349/398` (`0.8769`)을 기록했다. 특히 Mistral은 vanilla MemGPT tool-call gate를 통과하지 못했지만 teacher evidence를 받으면 높은 정확도로 답했다. 이는 작은 모델의 실패가 단순 지식 부족이 아니라 memory-management behavior 부족에서 크게 비롯된다는 가설을 강하게 지지한다.

마지막으로 승인된 GPT-4.1 trajectory를 이용해 Llama-3-8B에 LoRA distillation을 수행했다. LoRA `r=8`과 `r=16`은 post-training DMR에서 각각 GPT-4.1 semantic judge accuracy `0.4765`, `0.4809`를 기록했다. 두 조건의 semantic accuracy는 거의 비슷했지만, `r=16`은 operational metric에서 더 안정적이었다. `r=16`은 `497/500` loop completion, `3/500` format failure, search rate `0.7948`을 기록했다. Post-LoRA failure audit 결과, 남은 오답의 대부분은 모델이 정답 evidence를 찾았는데도 무시한 경우가 아니라, 검색 query가 부정확해 정답 evidence가 검색 결과에 들어오지 않은 경우였다. r16의 judge 오답 258개 중 193개가 `searched_wrong_or_insufficient_evidence`로 분류되었다.

이 해석을 검증하기 위해 LoRA adapter에도 teacher trace evidence를 직접 replay했다. 이 조건에서 r8 LoRA는 `343/398` (`0.8618`), r16 LoRA는 `345/398` (`0.8668`) GPT-4.1 judge accuracy를 기록했다. 즉 같은 adapter가 end-to-end search를 직접 수행하면 `0.48` 근처에 머물지만, teacher evidence가 주어지면 `0.86`대까지 회복된다. 이어서 teacher query chain만 prompt hint로 제공한 r16 ablation은 search rate `0.9594`, judge accuracy `248/394` (`0.6294`)를 기록했다.

따라서 현재 결론은 다음과 같다. Teacher-trajectory LoRA는 작은 모델의 MemGPT control surface와 answer-from-evidence behavior를 상당히 복구하지만, end-to-end DMR 정확도는 query selection과 evidence retrieval 병목 때문에 Teacher-Trace Replay 수준까지 올라가지 못한다. Teacher-query hint는 end-to-end 성능을 중간 정도 회복시키지만 teacher-evidence replay에는 도달하지 못한다. 추가로 query+answer SFT와 query-only SFT를 수행한 결과, token-level proxy metric이 좋아져도 Letta multi-turn loop의 tool-call channel contract가 무너질 수 있음을 확인했다. Phase-routed diagnostic에서는 controller가 search/answer phase를 분리하면 `100/100` row가 완료되지만, evidence-only containment는 `0.11`, retrieved-reference rate는 `0.09`에 머물렀다. Deterministic query skeleton에서는 query-only r16이 100-row에서 `0.23` retrieved-reference와 `0.23` containment까지 회복되어 full trajectory r16 skeleton의 `0.16`/`0.17`보다 높았고, raw 500-row에서는 retrieved-reference `0.244`, containment `0.216`을 기록했다. 그러나 같은 skeleton에서 teacher max-3 query는 approved 398 subset에서 retrieved-reference `0.367`, containment `0.342`, teacher-search subset에서 `0.483`/`0.450`을 기록했다. 이는 query-only SFT가 실패한 것이 아니라, query policy signal이 tool-call transport와 phase transition 부담에 가려졌음을 보여 주는 동시에, teacher 수준의 literal query selection까지는 아직 큰 gap이 남아 있음을 보여 준다.

## 1. Introduction

장기 대화 memory는 보통 retrieval-augmented generation 문제로 이해된다. 외부 retriever가 관련 과거 정보를 선택하고, LLM은 그 retrieved context를 읽어 답변한다. 그러나 MemGPT는 조금 다르다. MemGPT에서는 모델이 passive하게 retrieval 결과를 받는 것이 아니라, 스스로 memory-management tool을 호출한다. 모델은 언제 검색할지, 어떤 query를 사용할지, 검색 결과가 부족하면 다시 검색할지, 언제 최종 답변을 보낼지 결정해야 한다.

이 구조에서는 underlying model이 곧 memory policy의 일부가 된다. DMR 질문 하나에 성공하려면 모델은 다음 능력을 모두 보여야 한다.

1. 유효한 structured tool call을 생성해야 한다.
2. 과거 대화 memory에 실제로 등장할 법한 짧은 literal query를 선택해야 한다.
3. 첫 검색 결과가 부족하면 heartbeat continuation을 통해 추가 검색을 이어가야 한다.
4. 최종 답변은 persona priors나 hallucination이 아니라 retrieved evidence에 기반해야 한다.

강한 proprietary function-calling model은 이 요구사항을 어느 정도 자연스럽게 만족할 수 있다. 하지만 작은 open-source instruction model은 다를 수 있다. 이 경우 낮은 DMR accuracy는 여러 원인을 가질 수 있다. 모델이 지식이나 추론 능력이 부족해서 틀릴 수도 있지만, 그보다 앞서 memory interface를 제대로 조작하지 못해 실패할 수도 있다.

본 연구는 이 차이를 실험적으로 분리한다. 중심 질문은 다음이다.

> 작은 open-source LLM이 MemGPT DMR loop에서 낮은 성능을 보일 때, 주된 원인은 final answer 능력 부족인가, 아니면 memory-management behavior 부족인가?

이를 위해 여섯 단계를 수행했다. 첫째, raw vanilla와 parser-adapted vanilla baseline을 측정했다. 둘째, full-history upper bound와 Teacher-Trace Oracle을 통해 retrieval이 제거되거나 teacher evidence가 주어질 때 frozen student가 얼마나 회복되는지 보았다. 셋째, GPT-4.1 teacher trajectory를 Llama-3-8B에 LoRA로 distill했다. 넷째, post-LoRA 실패를 audit하여 남은 병목이 search, evidence, answer 중 어디에 있는지 분석했다. 다섯째, 같은 LoRA adapter에 teacher trace evidence를 직접 replay하여 end-to-end 실패가 answer-generation 문제인지 evidence acquisition 문제인지 더 분리했다. 여섯째, teacher query chain만 hint로 제공하여 query selection과 evidence-grounded final answer 사이의 gap을 측정했다.

현재 결과는 layered diagnosis를 지지한다. 작은 모델은 처음에는 tool-call control contract에서 실패한다. LoRA distillation은 이 표면 행동을 상당히 복구한다. 그러나 end-to-end 정확도는 Teacher-Trace Oracle보다 여전히 낮고, 남은 오답의 다수는 answer-bearing evidence를 retrieval하지 못한 경우다. 따라서 다음 학습은 단순 final answer SFT가 아니라 query selection과 evidence-grounded answer construction을 분리해서 다뤄야 한다.

## 2. Background and Task

### 2.1 MemGPT-style Memory Control

MemGPT-style agent에서는 모든 관련 memory가 항상 model context에 들어가지 않는다. 과거 대화는 외부 recall storage에 저장되고, 사용자가 과거 사실을 물으면 모델이 `conversation_search` 같은 memory tool을 호출해야 한다. 모델은 검색 결과를 읽고, 필요하면 추가 검색을 수행하고, 충분한 evidence를 얻은 뒤 최종 답변을 생성한다.

핵심은 retrieval이 model-controlled라는 점이다. 모델은 retrieval result를 소비하는 것뿐 아니라 retrieval action 자체를 선택한다.

이상적인 DMR trajectory는 다음과 같다.

```text
User:
  Hey, remember that time we talked about music?
  What was the artist you mentioned you could get into?

Model:
  conversation_search(query="music", request_heartbeat=true)

Tool:
  ... prior music conversation messages ...

Model:
  conversation_search(query="Taylor Swift", request_heartbeat=true)

Tool:
  ... evidence containing "Taylor Swift" ...

Model:
  Taylor Swift.
```

작은 차이가 큰 실패로 이어질 수 있다. 모델이 valid tool call 없이 "Let me search memory"라고 말만 하면 loop가 실패한다. `music`처럼 너무 넓은 query만 검색하면 정답 utterance가 검색 결과에 들어오지 않을 수 있다. 정답 evidence를 받았더라도 persona prior로 답하면 최종 accuracy는 낮다.

### 2.2 Deep Memory Retrieval

본 연구는 Multi-Session Chat Deep Memory Retrieval(DMR)을 사용한다. 각 row는 여러 과거 conversation session과, 나중에 등장하는 probe question으로 구성된다. probe는 이전 session에서 언급된 개인적 사실을 묻는다.

로컬 protocol은 다음과 같다.

1. 각 DMR row마다 새 Letta `memgpt_agent`를 만든다.
2. session 1-5를 recall storage에 capture한다.
3. active message를 reset하여 과거 메시지가 immediate context에 남지 않게 한다.
4. system prompt를 다시 compile한다.
5. session 6 probe만 agent에게 전달한다.
6. 모델은 recall search를 통해 과거 사실을 찾아 답해야 한다.

이 방식은 과거 메시지를 recall memory에는 유지하지만 immediate context leakage는 막는다. 세부 protocol은 `docs/vanilla_dmr_protocol.md`에 정리되어 있다.

### 2.3 Paper-Substring Recall Contract

중요한 구현 이슈는 search contract다. Maintained Letta의 tool description은 `conversation_search`를 hybrid 또는 semantic search처럼 설명할 수 있지만, 본 연구의 local DMR recall path는 PostgreSQL에 저장된 대화 메시지에 대해 case-insensitive substring matching을 수행한다. 따라서 우리는 paper-era DMR에 맞춰 좁은 tool description을 mount했다.

```text
Search prior conversation history using case-insensitive substring matching.
Choose short literal words or phrases that are likely to occur verbatim.
```

이 차이는 teacher trajectory 품질에 직접적인 영향을 준다. 초기 GPT-4.1 pilot에서는 semantic-search처럼 읽히는 tool description과 실제 substring execution이 어긋나 judge accuracy가 `12/20`에 머물렀다. tool contract를 paper-substring semantics에 맞춘 후 corrected pilot은 `17/20`으로 개선되었다.

## 3. Research Questions

본 논문은 다섯 개의 연구 질문을 중심으로 구성된다.

### RQ1: 작은 open-source model은 vanilla MemGPT control loop를 실행할 수 있는가?

Llama-3-8B와 Mistral-7B가 weight update나 parser help 없이 valid tool call을 만들 수 있는지 본다.

### RQ2: tool-call surface format을 고치면 retrieval quality도 회복되는가?

Strict deterministic parser adapter를 사용한다. 이 adapter는 explicit schema-valid tool-call JSON만 OpenAI `tool_calls` 형식으로 변환하며, missing intent를 추론하거나 새 query를 선택하지 않는다.

### RQ3: teacher evidence가 주어지면 frozen student는 답할 수 있는가?

Teacher-Trace Oracle replay에서는 student가 직접 search action을 선택하지 않는다. teacher가 실제로 얻은 tool output을 evidence로 주고, student는 final answer만 생성한다. 이 조건에서 성능이 회복되면, student의 answer-generation ability는 충분하지만 memory behavior가 부족하다는 해석이 가능하다.

### RQ4: teacher trajectory LoRA distillation은 end-to-end MemGPT behavior를 복구하는가?

GPT-4.1 teacher trajectory를 이용해 LoRA adapter를 학습하고, student가 다시 직접 DMR loop를 수행하게 한다.

### RQ5: LoRA 이후에도 남는 병목은 무엇인가?

Post-LoRA failure audit을 통해 남은 실패가 no search, wrong query/evidence, evidence not used, residual tool-call format failure 중 어디에 집중되는지 분석한다.

## 4. Experimental Setup

### 4.1 Models

주요 student model은 Llama-3-8B-Instruct이며, 접근 가능한 mirror checkpoint를 사용했다.

```text
NousResearch/Meta-Llama-3-8B-Instruct
```

추가로 다음 모델을 평가했다.

```text
mistralai/Mistral-7B-Instruct-v0.3
```

두 모델 모두 vLLM을 통해 BF16으로 serving했다. 보고된 DMR 실험에서는 quantized weight를 사용하지 않았다.

Teacher model은 다음 snapshot이다.

```text
gpt-4.1-2025-04-14
```

원본 MemGPT 논문은 GPT-4 Turbo `gpt-4-1106-preview`를 사용했지만, 해당 endpoint는 더 이상 사용할 수 없다. 따라서 본 연구의 GPT-4.1 결과는 원 논문의 teacher 조건을 정확히 재현한 것이 아니라, 현대적인 strong teacher를 사용한 diagnostic experiment로 해석한다.

### 4.2 Infrastructure

로컬 stack은 다음과 같다.

| Component | Role |
| --- | --- |
| Letta server | MemGPT-style agent loop와 memory tool 실행 |
| PostgreSQL/pgvector | persisted recall memory와 storage backend |
| vLLM | BF16 student serving과 LoRA adapter serving |
| `nano-memgpt-dev` | script, dataset preparation, evaluation, judge 실행 |
| PEFT/TRL | LoRA training stack |

### 4.3 Metrics

DMR에서는 단일 metric만으로 성능을 설명하기 어렵기 때문에 여러 지표를 함께 사용했다.

| Metric | 의미 |
| --- | --- |
| loop completion | agent가 behavioral failure 없이 끝났는가 |
| format failure | invalid tool-call 또는 control-loop failure |
| search rate | completed row 중 `conversation_search`를 호출한 비율 |
| ROUGE-L recall | reference와 answer의 surface overlap |
| deterministic containment | normalized reference string이 answer에 포함되는가 |
| GPT-4.1 judge accuracy | GPT-4.1이 semantic correctness를 인정하는가 |
| failure type distribution | 남은 오류의 진단적 분포 |

Containment는 엄격한 lexical metric이다. 반면 GPT-4.1 judge는 paraphrase와 harmless extra text를 허용하는 semantic metric이다. 따라서 이 둘을 같은 accuracy로 직접 비교하지 않는다.

### 4.4 Teacher Approval and Data Export

Teacher row를 모두 신뢰하지 않는다. GPT-4.1도 retrieval miss나 ambiguous answer를 만들 수 있기 때문이다. 따라서 GPT-4.1이 DMR answer와 trajectory를 생성한 뒤, 각 answer를 judge하고 승인된 row만 Oracle replay와 LoRA training에 사용했다.

Scaled teacher collection 결과는 다음과 같다.

| Item | Value |
| --- | ---: |
| raw teacher rows | `500` |
| GPT-4.1 judge-approved rows | `398` |
| approved Oracle traces | `398` |
| context-complete SFT steps | `1,664` |

## 5. Baselines

### 5.1 Raw Vanilla MemGPT

Raw vanilla student는 parser adapter나 prompt adapter 없이 그대로 serving했다.

| Model | Rows | Completed loop | Behavioral failures |
| --- | ---: | ---: | ---: |
| Llama-3-8B | 5 | 0 | 5 |
| Mistral-7B | 5 | 0 | 5 |

두 모델 모두 meaningful retrieval-quality 평가 이전에 실패했다. 이는 첫 번째 병목이 tool-call control contract임을 보여 준다.

### 5.2 Strict Template Adapter

Strict adapter는 explicit schema-valid tool-call surface만 보정한다. malformed intent를 복구하거나 query를 새로 고르지 않는다.

Pilot 결과:

| Model | Rows | Completed loop | ROUGE-L recall | containment | search rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 5 | 5 | 0.6323 | 0.4000 | 0.8000 |
| Mistral-7B | 5 | 0 | n/a | n/a | n/a |

Scaled Llama strict-template DMR:

| Rows | Completed loop | Behavioral failure | ROUGE-L recall | containment | search rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 481 | 19 | 0.3929 | 0.1954 | 0.8462 |

이 결과는 format repair가 Llama를 loop에 진입시키지만, retrieval behavior와 final answer quality를 충분히 회복하지는 못한다는 것을 보여 준다.

### 5.3 Full-History Upper Bound

Full-history 조건은 retrieval을 제거하고 과거 session 전체를 context에 직접 넣는다. deployable MemGPT 조건은 아니지만, frozen student가 evidence만 있으면 얼마나 답할 수 있는지 보는 upper bound다.

| Model | Rows | Completed | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: |
| Llama-3-8B | 500 | 500 | 0.7850 | 0.5080 |
| Mistral-7B | 500 | 500 | 0.6711 | 0.4240 |

## 6. Teacher-Trace Oracle

### 6.1 Pilot Oracle

Paper-substring contract를 교정한 후, GPT-4.1 20-row pilot은 `17/20` judge accuracy를 기록했다. 이 17개 approved teacher trace를 frozen student에게 replay한 결과는 다음과 같다.

| Model | Approved traces | Completed | Judge accuracy | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 17 | 17 | `14/17` (`0.8235`) | 0.6590 | 0.5294 |
| Mistral-7B | 17 | 17 | `15/17` (`0.8824`) | 0.7372 | 0.4706 |

### 6.2 Scaled Oracle

Scaled teacher collection은 500개 raw row를 포함한다. GPT-4.1 judge는 이 중 398개를 승인했다.

Teacher performance:

| Metric | Value |
| --- | ---: |
| raw teacher rows | `500` |
| completed | `500` |
| GPT-4.1 judge accuracy | `398/500` (`0.7960`) |
| ROUGE-L recall | `0.7393` |
| containment | `0.4120` |
| search rate | `0.7740` |

Approved teacher evidence를 student에게 replay한 결과:

| Model | Completed | Judge accuracy | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: |
| Llama-3-8B | `398/398` | `292/398` (`0.7337`) | `0.6597` | `0.4246` |
| Mistral-7B | `398/398` | `349/398` (`0.8769`) | `0.7208` | `0.4598` |

이 결과는 behavioral-bottleneck 가설을 강하게 지지한다. Mistral은 vanilla Letta control gate를 통과하지 못했지만, teacher evidence가 주어지면 approved probe의 대부분을 맞힌다. Llama 역시 strict-template end-to-end DMR보다 크게 회복된다.

## 7. LoRA Teacher-Trajectory Distillation

### 7.1 Training Data

승인된 GPT-4.1 teacher trajectory에서 context-complete SFT step을 export했다. 각 step은 provider request context와 teacher의 다음 action 또는 response를 보존한다. Scaled approved set은 다음 규모다.

```text
1,664 context-complete SFT steps
```

### 7.2 LoRA Configurations

현재 GPU 환경에서 Llama-3-8B full fine-tuning은 현실적으로 어렵기 때문에 LoRA를 주요 학습 방법으로 사용했다.

| Condition | rank | alpha | final train loss | final eval loss | final token accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA r8 | 8 | 16 | 1.0890 | 0.9009 | 0.7546 |
| LoRA r16 | 16 | 32 | 1.0270 | 0.8694 | 0.7599 |

Token-level proxy에서는 r16이 더 좋다. 하지만 실제 결론은 end-to-end DMR에서 내려야 한다.

### 7.3 Post-Training DMR Evaluation

두 adapter는 같은 Llama-3-8B base model 위에서 vLLM으로 serving했다.

| Condition | Loop completion | Format failure | Search rate | ROUGE-L recall | containment | GPT-4.1 judge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA r8 | `489/500` | `11` | `0.7280` | `0.5489` | `0.2474` | `0.4765` |
| LoRA r16 | `497/500` | `3` | `0.7948` | `0.5515` | `0.2656` | `0.4809` |

Strict-template Llama와 비교하면 LoRA는 loop stability와 containment를 개선한다.

| Condition | Loop completion | Format failure | Search rate | containment | Semantic judge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw vanilla Llama pilot | `0/5` | `5` | n/a | n/a | n/a |
| Strict-template Llama scaled | `481/500` | `19` | `0.8462` | `0.1954` | not run |
| LoRA r8 | `489/500` | `11` | `0.7280` | `0.2474` | `0.4765` |
| LoRA r16 | `497/500` | `3` | `0.7948` | `0.2656` | `0.4809` |

r16은 r8보다 operationally 안정적이다. Format failure가 적고, search rate와 containment가 높다. 그러나 semantic judge accuracy는 거의 동일하다. 이는 LoRA rank 증가가 control-loop stability와 retrieval action frequency에는 도움을 주지만, evidence selection 자체를 충분히 해결하지는 못한다는 것을 시사한다.

### 7.4 LoRA Teacher-Trace Replay

Post-training end-to-end DMR만 보면 LoRA adapter가 최종 답변 생성 능력을 충분히 배우지 못했다고 오해할 수 있다. 이를 분리하기 위해 승인된 GPT-4.1 teacher trace를 같은 LoRA adapter에 직접 replay했다. 이 조건은 Letta end-to-end loop가 아니라 direct vLLM answer-from-evidence 조건이다. Student는 teacher가 실제로 얻은 tool output을 읽고 최종 답변만 생성한다.

| Condition | Evidence source | Student controls search? | Rows | containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: |
| LoRA r8 end-to-end | student retrieval | Yes | `489` completed | `0.2474` | `0.4765` |
| LoRA r8 + teacher trace | teacher evidence | No | `398` approved | `0.4874` | `0.8618` |
| LoRA r16 end-to-end | student retrieval | Yes | `497` completed | `0.2656` | `0.4809` |
| LoRA r16 + teacher trace | teacher evidence | No | `398` approved | `0.4899` | `0.8668` |

이 결과는 post-LoRA 병목이 answer-from-evidence 능력 부족보다는 evidence acquisition에 있다는 해석을 강하게 지지한다. 특히 LoRA r16은 end-to-end에서는 `0.4809`에 머물지만, teacher evidence가 주어지면 `0.8668`까지 회복된다. 이어지는 teacher-query hint ablation은 query hint만으로 `0.6294`까지 회복됨을 보여 주어, query selection과 evidence-grounded final answer를 모두 별도 objective로 다뤄야 한다는 해석을 강화한다.

### 7.5 LoRA Teacher-Query Hint

Teacher evidence replay는 query selection과 evidence retrieval을 모두 teacher에게 맡긴다. 이를 한 단계 더 분리하기 위해 teacher가 사용한 `conversation_search` query chain만 prompt hint로 제공하고, Letta end-to-end loop는 student가 직접 수행하게 했다. 이 조건은 Letta 내부 tool call을 강제 치환한 pure Teacher-query ablation이 아니라, query hint 기반 ablation이다.

| Condition | Search source | Evidence source | Rows judged | Search rate | containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LoRA r16 end-to-end | student | student retrieval | `497` | `0.7948` | `0.2656` | `0.4809` |
| LoRA r16 + teacher-query hint | teacher query hint + student execution | student retrieval | `394` | `0.9594` | `0.3452` | `0.6294` |
| LoRA r16 + teacher trace | teacher trace | teacher evidence replay | `398` | n/a | `0.4899` | `0.8668` |

이 결과는 query selection이 실제 병목임을 보여 준다. Teacher query hint만으로 judge accuracy가 `0.4809`에서 `0.6294`로 오른다. 그러나 teacher evidence replay의 `0.8668`과는 큰 차이가 남는다. Query hint 조건에서 planning leakage 56개, tool-output leakage 15개가 관찰되었고, tool-output leakage 중 14개는 judge 오답이었다. 따라서 query selection만 해결하는 것보다, retrieved evidence를 user-facing final answer로 바꾸는 boundary를 별도로 학습해야 한다.

### 7.6 Query+Answer SFT와 Query-Only SFT Negative Results

Teacher-query hint 결과를 보고 query supervision을 더 직접적으로 넣는 후속 학습을 수행했다.
먼저 query-chain target과 evidence-grounded answer target을 결합한 query+answer r16 adapter를
학습했다. 이 조건은 final eval loss `0.8583`, token accuracy `0.7859`로 기존 full-trajectory
r16보다 좋은 proxy metric을 보였다. 그러나 Letta DMR smoke에서는 초반 `35`개 row 중
`34`개가 tool-call format failure로 끝났다.

이 실패를 분리하기 위해 answer target을 제거하고 `conversation_search` target만 남긴
query-only r16 adapter를 추가로 학습했다. Query-only는 `1,178` query-call record 중
`1,125`개를 train, `53`개를 eval로 사용했고, final eval loss `0.8244`, token accuracy
`0.7703`을 기록했다. Proxy metric은 가장 좋았지만, 20-row smoke의 첫 `5`개 row가 모두
behavioral failure로 종료되어 full evaluation을 중단했다.

중요한 점은 query-only 실패가 완전한 no-tool failure는 아니라는 것이다. 5개 실패 row에서
모델은 초반에 정상 OpenAI `tool_calls` 채널로 `conversation_search`를 여러 번 생성했다.
각 row의 valid tool-call message 수는 `9`, `21`, `10`, `10`, `7`개였다. 하지만 이후 같은
tool-call JSON을 assistant `content` 문자열로 출력했고, provider는 이를
`tool_calls=[]`, `finish_reason="stop"`으로 반환했다. Letta는 이 응답을
`No tool calls found in response, model must make a tool call`로 거절했다.

이 negative result는 연구적으로 중요하다. Query-only SFT는 tool-call intent를 일부
강화했지만, multi-turn MemGPT loop 전체에서 tool-call을 반드시 `tool_calls` 채널에 싣는
transport/channel 안정성은 보장하지 못했다. 이를 확인하기 위해 vLLM-level
`nano_rescue_llama` parser도 추가했다. 이 parser는 explicit JSON call에서 schema 밖 noise
field를 버리고 단순 타입을 정규화한다. 그러나 6-row smoke에서 `1`개만 완료되고 `5`개가
다시 tool-call format failure로 끝났다. 이후 OpenAI-compatible endpoint proxy를 구현해
assistant `content` JSON을 `tool_calls`로 변환했다. Proxy는 rescue event를 `70`회 기록했지만,
20-row smoke는 `2/20` completion, containment `0.0`에 머물렀다. 완료된 두 row도 정답이
아니라 planning leakage였다. 따라서 query-only adapter는 end-to-end agent가 아니라 search
phase 전용 모듈로 해석해야 한다.

### 7.7 Phase-Routed Query-Only Diagnostic

Query-only adapter가 search phase 전용 모듈로는 의미가 있는지 확인하기 위해, Letta
multi-turn loop를 우회한 phase-routed diagnostic을 수행했다. Search phase는
`nano-memgpt-llama3-query-only-r16`이 담당하고, answer phase는 full trajectory LoRA인
`nano-memgpt-llama3-r16`이 담당한다. Controller는 최대 3회까지 substring recall을 실행한 뒤
retrieved evidence만 answer model에 전달한다. Persona와 full history는 answer prompt에서
제거했고, evidence가 없거나 부족하면 `UNKNOWN`을 답하게 했다.

| Condition | Rows | Completion | Retrieved-reference rate | Containment | Mean searches | `UNKNOWN` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-routed query-only search + r16 answer | `100` | `100/100` | `0.09` | `0.11` | `2.22` | `67/100` |

이 결과는 두 가지를 분리한다. Controller가 phase 전환을 맡으면 stop-and-answer failure는
사라진다. 그러나 정답 evidence를 실제로 retrieve한 row는 `9/100`에 그친다. 따라서 query-only
adapter의 남은 문제는 OpenAI tool-call channel만이 아니라, DMR probe에서 answer-bearing
utterance를 찾는 literal query를 고르는 능력이다.

### 7.8 Deterministic Query Skeleton

Phase-routed 조건에서도 query-only adapter는 여전히 `conversation_search` JSON을 생성해야
했다. 따라서 다음 diagnostic에서는 모델이 query string만 출력하고, wrapper가 tool shell을
고정했다.

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `100/100` | `0.09` | `0.11` | `1.10` |
| Query-only r16, skeleton | `100` | `100/100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `100/100` | `0.16` | `0.17` | `2.86` |
| Base Llama skeleton smoke | `20` | `20/20` | `0.20` | `0.20` | `1.75` |

이 결과는 query-only SFT가 완전한 실패가 아니라는 점을 보여 준다. Query-only adapter는
end-to-end agent로는 실패했지만, constrained query generator로 쓰면 full trajectory r16보다
더 높은 retrieved-reference rate를 보인다. 즉 query-only SFT에는 유효한 query-policy signal이
있고, 문제는 그 signal을 tool-call transport와 stop-and-answer policy까지 동시에 요구하는
interface에서 사용하려 한 데 있다.

### 7.9 Teacher Query Skeleton Replay

Deterministic query skeleton의 query-quality gap을 정량화하기 위해, GPT-4.1 teacher가 실제로
사용한 `conversation_search` query를 같은 local substring skeleton으로 replay했다. 이 조건은
teacher-query hint ablation과 다르다. Teacher-query hint는 teacher query chain을 prompt로 주고
student가 Letta loop 안에서 행동하게 하지만, teacher query skeleton replay는 query string만
가져와 동일한 retrieval wrapper와 동일한 evidence-only answer prompt를 적용한다. 따라서 query
자체가 reference-bearing message를 얼마나 잘 찾아오는지 직접 볼 수 있다.

| Query source | Subset | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher full query chain | approved | `398` | `0.442` | `0.405` | `4.36` |
| Teacher full query chain | teacher-search only | `302` | `0.583` | `0.533` | `5.74` |
| Teacher max-3 query | approved | `398` | `0.367` | `0.342` | `3.25` |
| Teacher max-3 query | teacher-search only | `302` | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | same approved | `398` | `0.276` | `0.249` | `3.76` |
| Query-only r16 skeleton | teacher-search subset | `302` | `0.272` | `0.238` | `3.76` |

Budget을 맞춘 비교에서는 `Teacher max-3 query`가 가장 적절한 upper reference다. Approved
398 subset에서 query-only skeleton은 teacher max-3 containment의 약 `73%` (`0.249 / 0.342`)를
달성한다. 하지만 teacher가 실제 search를 수행한 302개 subset에서는 약 `53%`
(`0.238 / 0.450`)에 그친다. 이 차이는 query-only adapter가 단순한 random query generator는
아니지만, teacher처럼 indirect probe에서 answer-bearing literal을 안정적으로 고르는 수준에는
아직 도달하지 못했음을 보여 준다.

또 하나 중요한 점은 mean retrieved다. Query-only skeleton은 approved subset에서 teacher max-3보다
더 많은 message를 retrieve하지만(`3.76` vs `3.25`), reference hit는 더 낮다. 즉 다음 학습은
검색 횟수나 recall 폭을 단순히 늘리는 방향이 아니라, reference-bearing query와 distractor-heavy
query를 구분하는 retrieval-supervised objective로 가야 한다.

## 8. Post-LoRA Failure Audit

LoRA가 Teacher-Trace Oracle 수준까지 올라가지 못한 이유를 분석하기 위해, 각 post-LoRA row의 tool call, tool return, GPT-4.1 judge label을 이용해 failure audit을 수행했다.

### 8.1 Audit Summary

| Run | Rows | OK | Judge acc | Lexical acc | Search rate | Evidence-hit rate | Evidence-hit among wrong | Mean searches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r8 | 500 | 489 | 0.4765 | 0.2474 | 0.7280 | 0.1534 | 0.0352 | 1.3701 |
| r16 | 500 | 497 | 0.4809 | 0.2656 | 0.7948 | 0.1489 | 0.0194 | 1.7022 |

`Evidence-hit rate`는 tool output 안에 reference가 exact normalized substring으로 들어 있는지 보는 lower bound다. Paraphrase evidence는 놓칠 수 있다.

### 8.2 Failure Types

| Failure type | r8 | r16 | Interpretation |
| --- | ---: | ---: | --- |
| `correct_semantic` | `233` | `239` | GPT-4.1 judge가 정답으로 인정 |
| `no_search` | `77` | `60` | 검색이 없거나 검색 의도만 prose로 말함 |
| `searched_wrong_or_insufficient_evidence` | `170` | `193` | 검색은 했지만 answer evidence를 가져오지 못함 |
| `evidence_found_but_not_used` | `9` | `5` | reference evidence가 있었지만 최종 답변이 틀림 |
| `tool_call_format_failure` | `11` | `3` | control-loop failure |

r16의 judge 오답 row 중 `193/258` (`74.8%`)이 `searched_wrong_or_insufficient_evidence`다. 이것이 post-LoRA의 핵심 진단이다. Distillation 이후 모델은 자주 검색을 시도하지만, query가 answer-bearing utterance를 안정적으로 끌어오지 못한다.

### 8.3 Example: Broad Query Fails

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

검색 결과에는 probe 자체와 country-music 관련 문장만 있었고, `Taylor Swift`는 없었다. 모델은 Chris Stapleton/Jason Aldean 같은 country artist를 답했다. 이 row는 주로 answer-generation 문제가 아니라, answer-bearing evidence가 retrieval되지 않은 문제다.

### 8.4 Example: Evidence Found but Ignored

Reference:

```text
The drums!
```

Queries:

```text
What instrument
play
```

검색 결과 중 하나에는 `I play the drums`가 있었지만, 모델은 guitar를 답했다. 이것은 genuine evidence-grounding failure다. 다만 현재 exact-reference audit에서는 이 유형이 비교적 드물다.

### 8.5 Surface Leakage

일부 final answer에는 실행 의도가 그대로 새어 나온다.

```text
Let me check our past conversations...
Thinking aloud...
```

이런 wrong row는 r8에서 31개, r16에서 39개 관찰되었다. 이는 모델이 tool-use trajectory의 일부를 imitation했지만, internal action planning과 user-facing final answer boundary를 완전히 분리하지 못했다는 신호다.

## 9. Discussion

### 9.1 LoRA가 해결한 것

LoRA teacher-trajectory distillation은 Type-0 control failure를 분명히 줄인다. Raw vanilla에서 strict adapter, LoRA로 갈수록 다음 변화가 보인다.

1. raw model은 MemGPT loop에 안정적으로 진입하지 못한다.
2. strict parsing은 Llama를 loop에 넣지만 좋은 memory behavior를 학습시키지는 않는다.
3. LoRA는 format failure를 줄이고 containment를 개선한다.
4. r16은 r8보다 operationally 안정적이다.

이는 작은 모델의 MemGPT 실패가 부분적으로 behavioral failure이며, training으로 일부 복구 가능하다는 주장을 지지한다.

### 9.2 LoRA가 해결하지 못한 것

LoRA는 Oracle gap을 닫지 못했다.

| Condition | Diagnostic metric |
| --- | ---: |
| Base Llama Teacher-Trace Oracle replay | `0.7337` |
| LoRA r8 end-to-end | `0.4765` |
| LoRA r16 end-to-end | `0.4809` |
| LoRA r16 + teacher-query hint | `0.6294` |
| LoRA r8 + teacher trace | `0.8618` |
| LoRA r16 + teacher trace | `0.8668` |
| Phase-routed query-only evidence-only | containment `0.11` |
| Query-only deterministic skeleton | containment `0.23` |
| Query-only deterministic skeleton, raw 500 | containment `0.216` |
| Teacher max-3 query skeleton, approved subset | containment `0.342` |
| Teacher max-3 query skeleton, teacher-search subset | containment `0.450` |
| Hard-positive query SFT skeleton, raw 500 | containment `0.180` |
| Preference zero-DPO query skeleton, raw 500 | containment `0.158` |
| Candidate count rerank query skeleton, raw 500 | containment `0.202` |
| Candidate count rerank query skeleton, hard72 | containment `0.236` |
| Candidate lexical rerank query skeleton, raw 500 | containment `0.224` |
| Candidate lexical rerank query skeleton, hard72 | containment `0.306` |
| Evidence filter top-6, raw 500 | containment `0.224`, mean retrieved `3.426` |
| Evidence filter top-6, hard72 | containment `0.306`, mean retrieved `3.014` |

이 표는 여러 종류의 gap을 분리한다. End-to-end LoRA와 teacher-query hint 사이의 gap은 query selection에서 발생한다. Teacher-query hint와 teacher trace 사이의 gap은 query 실행 이후의 evidence selection, tool-result interpretation, final-answer boundary에서 발생한다. Phase-routed query-only evidence-only 결과는 query-only adapter의 검색어가 tool-call JSON shell 아래에서는 answer-bearing evidence를 거의 찾지 못함을 보여 준다. Deterministic skeleton 결과는 같은 query-only adapter가 query string 전용 interface에서는 훨씬 나아진다는 것을 보여 준다. 반대로 teacher evidence가 주어졌을 때 LoRA가 base Llama Oracle보다 높아지는 것은 trajectory distillation이 answer-from-evidence behavior도 일부 개선했음을 시사한다.

Teacher query skeleton replay는 이 gap의 상한선을 더 명확히 한다. Search budget을 최대 3회로 맞춰도 teacher query는 approved subset에서 containment `0.342`, teacher-search subset에서 `0.450`을 기록한다. 같은 subset에서 query-only skeleton은 각각 `0.249`, `0.238`이다. 따라서 query-only SFT는 query signal을 일부 학습했지만, teacher의 indirect-probe-to-literal-query mapping에는 아직 도달하지 못했다.

Row-level gap 분석도 같은 방향을 지지한다. Teacher-search 302 subset에서 retrieved-reference 기준
`both` success는 74개, `teacher_only` success는 72개, `student_only` success는 8개, `neither`는
148개였다. 즉 student가 teacher를 대체하는 row보다 teacher가 맞히고 student가 놓치는 row가
훨씬 많다. 이 72개 teacher-only retrieval row가 다음 retrieval-supervised query objective의
가장 좋은 hard set이다.

이 hard set에서 preference dataset도 export했다. Default setting은 72개 `prompt/chosen/rejected`
record를 만들고, 더 보수적인 zero-result-only negative subset은 51개 record를 만든다. 같은
positive query를 기존 SFT trainer 형식으로도 export했으며, `72`개 record 모두 overlength drop 없이
prepare check를 통과했다.

Hard-positive query SFT pilot도 실행했다. 72개 positive query로 1 epoch 학습한 adapter는 hard set
내부에서는 retrieved-reference를 `0/72`에서 `7/72`로 올렸지만, raw 500-row skeleton에서는
retrieved-reference `0.184`, containment `0.180`으로 기존 query-only r16의 `0.244`/`0.216`보다
낮아졌다. 즉 positive-only SFT는 국소적 신호는 배우지만 전체 query distribution을 좁혀
no-result/UNKNOWN을 늘린다.

Zero-result-only preference DPO pilot도 같은 결론을 강화했다. 51개 clean preference pair로 DPO를
1 epoch 학습한 adapter는 hard 72 rows 내부에서 retrieved-reference를 `8/72`까지 올렸지만, raw
500-row skeleton에서는 retrieved-reference `0.166`, containment `0.158`로 더 낮아졌다. 따라서
teacher query signal은 존재하지만, 소규모 hard-case SFT/DPO는 general query policy를 개선하지
못한다.

Candidate query generation + local retrieval reranking도 실행했다. 단순 result-count reranker는 raw
500-row에서 retrieved-reference `0.210`, containment `0.202`로 query-only skeleton `0.244`/`0.216`을
넘지 못했다. 그러나 hard 72 rows에서는 retrieved-reference를 `0/72`에서 `20/72`, containment를
`1/72`에서 `17/72`로 올렸다. 이후 candidate result text와 probe/query의 lexical overlap, broad-query
penalty를 넣은 lexical target-5 reranker는 raw 500-row에서 retrieved-reference `0.246`, containment
`0.224`를 기록해 query-only skeleton을 처음으로 소폭 넘었다. Hard 72 rows에서도 retrieved-reference를
`25/72`, containment를 `22/72`까지 올렸다. 즉 search-time candidate selection은 SFT/DPO보다
teacher-only failure class에 훨씬 직접적으로 맞고, lexical evidence feedback을 추가하면 full
distribution도 무너뜨리지 않는다. 다음 단계는 query LoRA를 더 오래 학습하는 것이 아니라, reference
recall을 유지하면서 mean retrieved `4.742`가 만드는 distractor를 줄이는 learned/embedding reranker와
evidence-grounded answer adapter다.

마지막으로 evidence filtering final diagnostic을 실행했다. Lexical target-5 reranker의 retrieved evidence를
answer 직전에 non-oracle lexical top-6으로 줄인 뒤 같은 answer model로 replay했다. Raw 500-row에서는
retrieved-reference가 `0.246`에서 `0.234`로 조금 낮아졌지만, containment는 `0.224`로 유지되었고
mean retrieved는 `4.742`에서 `3.426`으로 줄었다. Hard72에서도 retrieved-reference는 `25/72`에서
`23/72`로 줄었지만 containment는 `22/72`로 유지되었다. Zero51에서도 retrieved-reference는 `18/51`에서
`16/51`로 줄었지만 containment는 `15/51`로 유지되었다. 이는 lexical filter가 answer-bearing evidence를
완벽하게 고르지는 못하지만, candidate reranker의 높은 evidence volume을 줄이면서 final answer accuracy를
보존할 수 있음을 보여 준다. 이 결과를 v1 실험 종료선으로 둔다.

### 9.3 Query Selection이 어려운 이유

DMR probe는 종종 indirect하다. Probe 안에 literal answer가 직접 등장하지 않을 수 있다. 예를 들어 probe는 "the artist you could get into"라고 묻지만, 정답은 "Taylor Swift"일 수 있다. `music`을 검색하면 관련 메시지는 나오지만 answer-bearing utterance는 나오지 않을 수 있다. 좋은 memory policy는 candidate literal을 추론하고, 좁은 phrase를 검색하고, broad search가 실패했을 때 추가 query를 chain해야 한다.

Teacher trajectory에는 이런 behavior가 포함되어 있지만, 현재 LoRA objective는 여러 target을 한꺼번에 섞는다.

```text
tool-call format
query choice
heartbeat chaining
tool-result interpretation
final answer
```

모델은 표면 패턴은 배웠지만 query-selection strategy를 충분히 학습하지 못했을 가능성이 높다.

후속 query-only SFT는 이 해석을 더 세밀하게 만들었다. Query target만 남겨도 모델은 실제
`conversation_search` tool call을 여러 번 생성했지만, 몇 턴 뒤 같은 JSON을 assistant
`content`로 출력하면서 Letta contract를 깨뜨렸다. 따라서 query selection difficulty는
두 층으로 나뉜다. 하나는 어떤 literal query를 고를지의 semantic policy이고, 다른 하나는
그 query를 agent runtime이 요구하는 tool-call channel로 끝까지 유지하는 transport policy다.

Deterministic query skeleton은 이 둘을 더 분리했다. Tool-call shell을 wrapper가 고정하고
모델이 query string만 생성하면 query-only r16의 retrieved-reference rate가 `0.23`까지 오른다.
이는 query-only SFT가 semantic query policy를 일부 학습했다는 뜻이다. 다만 mean retrieved가
`3.70`으로 올라 distractor도 늘었으므로, 다음 문제는 query hit rate뿐 아니라 evidence
specificity와 distractor filtering이다.

### 9.4 r8과 r16의 Judge Accuracy가 비슷한 이유

r16은 operational metric을 개선하지만 semantic judge accuracy를 크게 올리지는 못했다. 가능한 해석은, 유용한 evidence가 retrieval된 row에서는 두 adapter 모두 어느 정도 답할 수 있다는 것이다. 차이는 r16이 더 자주 검색하고 더 적게 format failure를 낸다는 점이다. 그러나 r16의 많은 검색 역시 insufficient evidence를 가져오므로 semantic score는 query quality에 의해 bottleneck된다.

## 10. Limitations

1. **Teacher model mismatch**: 원본 MemGPT 논문은 `gpt-4-1106-preview`를 사용했지만, 본 연구는 `gpt-4.1-2025-04-14`를 사용한다.
2. **Letta implementation drift**: Maintained Letta는 historical MemGPT와 tool description 및 server behavior에서 차이가 있다. Paper-substring contract로 완화했지만, 완전한 historical reproduction은 아니다.
3. **Judge dependency**: GPT-4.1 judge는 semantic scoring에 유용하지만 완전한 ground truth는 아니다. 최종 논문에는 표본 manual audit이 필요하다.
4. **Exact evidence-hit lower bound**: Failure audit은 normalized substring matching으로 evidence presence를 판단하므로 paraphrased evidence를 과소평가한다.
5. **Document-QA incomplete reproduction**: 원 논문의 20M Wikipedia embedding index가 현재 public artifact 경로에 없어 Document-QA는 proxy로만 다룬다.
6. **LoRA only**: 메모리 제약 때문에 full fine-tuning은 수행하지 않았다.

## 11. Planned Ablations

Teacher-evidence ablation과 teacher-query hint ablation은 이미 실행되었다. 이 둘은 end-to-end LoRA 실패가 query selection과 evidence-grounded answer generation 양쪽에 걸쳐 있음을 보여 준다. Query+answer SFT와 query-only SFT도 실행되었고, 둘 다 token-level proxy metric은 개선했지만 Letta loop에서는 tool-call channel failure를 만들었다. vLLM parser-rescue와 endpoint proxy-rescue는 일부 channel failure를 구제했지만, query-only는 search-only behavior 때문에 final-answer transition을 거의 수행하지 못했다. Phase-routed diagnostic은 final-answer transition을 controller로 해결해도 query-only evidence hit가 낮다는 점을 추가로 보여 주었다. Deterministic query skeleton은 tool-call shell까지 제거하면 query-only evidence hit가 다시 올라간다는 점을 보여 주었다. Teacher query skeleton replay도 완료되었고, teacher max-3 query가 query-only skeleton보다 명확히 높은 reference retrieval을 보인다는 점을 확인했다.

### 11.1 Phase-Routed Adapter

Query-only adapter는 search action을 계속 생성하는 데에는 도움이 되지만, final answer로
전환하지 못한다. Phase-routed diagnostic은 이 조건을 100-row로 실행한 첫 결과다.

```text
search phase: query-only adapter 또는 deterministic query generator
answer phase: full LoRA/base/answer-only adapter
controller: evidence sufficiency 또는 max-search rule로 phase 전환
```

이 조건은 query-only 학습의 쓸모를 버리지 않고, end-to-end agent loop의 stop-and-answer
문제를 분리한다. 현재 결과에서는 completion이 `100/100`으로 회복되었지만, retrieved-reference
rate가 `0.09`라 query generator 자체를 더 개선해야 한다.

### 11.2 Deterministic Tool Skeleton

모델이 전체 JSON tool call을 생성하지 않고 query string만 생성하도록 한다.

```text
input: DMR probe + memory-search instruction
target: query string only
wrapper: conversation_search(name, roles, limit, request_heartbeat) 고정
```

이 조건은 실행되었고, query-only r16 skeleton이 full trajectory r16 skeleton보다 높은
retrieved-reference rate를 보였다. 이어서 teacher query도 같은 skeleton으로 replay해 query
upper reference를 측정했다.

### 11.3 Teacher Query Skeleton Replay

Teacher가 사용한 query string을 같은 local substring skeleton으로 실행한다. 이 조건은
teacher-query hint ablation과 달리 Letta agent loop를 거치지 않으므로, teacher query 자체가
reference-bearing utterance를 얼마나 잘 retrieve하는지 직접 측정한다.

이 조건은 완료되었다. Search budget을 최대 3회로 제한해도 teacher query는 approved 398 subset에서
retrieved-reference `0.367`, containment `0.342`를 기록했고, teacher-search 302 subset에서는
`0.483`, `0.450`을 기록했다. Query-only skeleton은 같은 subset에서 각각 `0.276`/`0.249`,
`0.272`/`0.238`에 머물렀다. 따라서 다음 단계는 teacher query를 더 많이 모방하는 것보다,
row-level로 teacher-correct/student-wrong query를 분석하고 retrieval-supervised query objective를
설계하는 것이다.

### 11.4 Query-Chain SFT

Teacher-query hint ablation은 query selection을 개선하면 성능이 오르지만, hint만으로는 충분하지 않음을 보여 주었다. 다음 학습에서는 query chain 자체를 별도 target으로 분리한다.

다만 query-only SFT negative result 때문에, 다음 query-chain 학습은 전체 JSON action을
그대로 생성하게 하는 방식보다 parser-rescue 또는 skeleton 조건과 함께 평가해야 한다.

### 11.5 Teacher-Result Ablation

Student에게 teacher의 retrieved evidence 또는 answer-bearing snippet을 직접 제공하고 final answer만 생성하게 한다.

이 조건은 다음 질문에 답한다.

```text
정답 evidence가 있을 때에도 answer-generation error가 얼마나 남는가?
```

현재 실행한 LoRA Teacher-Trace Replay는 이 조건에 가깝지만, teacher의 전체 tool trace를 replay한다는 점에서 더 넓은 evidence ablation이다. 더 엄밀한 Teacher-Result Ablation은 answer-bearing snippet만 주거나, teacher query와 tool result를 분리하여 입력해야 한다.

### 11.6 Query SFT

별도 objective를 학습한다.

```text
input: DMR probe and memory-search instruction
target: teacher conversation_search query
```

목표는 `no_search`와 `searched_wrong_or_insufficient_evidence`를 줄이는 것이다.

### 11.7 Evidence-Grounded Answer SFT

다음 objective를 학습한다.

```text
input: DMR probe + retrieved evidence
target: concise teacher answer
```

목표는 `evidence_found_but_not_used`와 final-answer surface leakage를 줄이는 것이다.

## 12. Conclusion

본 연구는 작은 open-source LLM이 MemGPT-style DMR에서 실패하는 이유가 단일하지 않음을 보인다. Raw vanilla model은 structured tool-call control contract에서 먼저 실패한다. Strict parser repair는 Llama를 loop에 진입시키지만 accuracy는 낮게 남는다. GPT-4.1 Teacher-Trace Oracle replay는 frozen student가 teacher evidence를 받으면 훨씬 잘 답할 수 있음을 보여 주며, memory-management behavior가 주요 병목임을 시사한다.

승인된 teacher trajectory에 대한 LoRA distillation은 control surface를 상당히 복구한다. Llama-3-8B r16 LoRA는 `497/500` loop completion, `3` format failure, GPT-4.1 judge accuracy `0.4809`를 기록했다. 그러나 같은 r16 LoRA에 teacher trace evidence를 직접 주면 `345/398` (`0.8668`)까지 회복된다. Failure audit 결과와 함께 보면, 남은 end-to-end 오답의 대부분은 student가 insufficient query로 검색해 answer-bearing evidence를 가져오지 못한 경우였다.

따라서 다음 연구 방향은 단순히 LoRA rank를 키우거나 generic SFT를 추가하는 것이 아니다. Memory query selection과 evidence-grounded answer generation을 명시적으로 분리해 학습해야 한다. Teacher-query hint ablation은 query hint만으로도 성능이 `0.4809`에서 `0.6294`로 오르지만, teacher evidence replay의 `0.8668`에는 도달하지 못함을 보여 주었다. Query+answer 및 query-only SFT negative result는 여기에 한 가지 제약을 더한다. Query content를 학습시키는 것과 tool-call channel을 안정적으로 유지하는 것은 같은 문제가 아니다. Phase-routed diagnostic은 channel과 phase 전환을 controller가 맡아도 query-only adapter의 retrieved-reference rate가 `0.09`에 그친다는 더 좁은 결론을 준다. Deterministic skeleton은 같은 adapter가 query string만 생성하면 raw 500-row에서 retrieved-reference `0.244`, containment `0.216`까지 회복됨을 보여 주어, query-only SFT 안에 유효한 query-policy signal이 있음을 확인했다.

이후 candidate query generation + retrieval-feedback reranking은 이 방향을 더 강화했다. 단순 result-count reranker는 hard failure class를 크게 복구했지만 raw500 전체에서는 query-only보다 낮았다. 반면 lexical target-5 reranker는 raw500에서 retrieved-reference `0.246`, containment `0.224`를 기록해 query-only skeleton을 소폭 넘었고, hard72에서는 retrieved-reference를 `25/72`, containment를 `22/72`까지 올렸다. 마지막 evidence filtering diagnostic에서는 top-6 filter가 raw500 containment `0.224`를 유지하면서 mean retrieved를 `4.742`에서 `3.426`으로 줄였다. Hard72와 zero51에서도 containment는 각각 `22/72`, `15/51`로 유지되었다.

그러나 teacher max-3 query skeleton은 approved subset에서 retrieved-reference `0.367`, containment `0.342`를 기록하므로, query-only adapter에는 여전히 teacher-level query selection gap이 남아 있다. 또한 evidence filter는 retrieved-reference를 raw500 `0.246`에서 `0.234`로 낮췄으므로, lexical filtering만으로 answer-bearing evidence를 완벽하게 식별하지는 못한다. 작은 모델의 MemGPT 복구에는 final answer model만이 아니라 memory policy, evidence-to-answer boundary, tool-call transport/channel control을 함께 분리해서 다뤄야 한다. v1 연구는 여기서 실험을 freeze하고, 이후 작업은 paper writing과 example analysis로 전환한다.

## Appendix A. Artifact Map

### Baseline and DMR Evaluation

```text
docs/experiment_1_report.md
docs/vanilla_dmr_protocol.md
data/evaluation/experiment_1/dmr/
```

### Oracle and Teacher Trajectories

```text
docs/oracle_experiment.md
docs/oracle_experiment_report.md
data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/
data/trajectories/gpt41_paper_substring_scaled_approved_rows.jsonl
data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl
data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl
```

### LoRA Training and Post-LoRA Evaluation

```text
docs/lora_training.md
docs/post_lora_evaluation_report.md
docs/teacher_evidence_ablation_report.md
docs/teacher_query_ablation_report.md
docs/query_only_lora_report.md
docs/phase_routed_dmr_report.md
docs/query_skeleton_dmr_report.md
docs/teacher_query_skeleton_report.md
docs/query_skeleton_gap_report.md
docs/query_preference_dataset_report.md
docs/query_hard_positive_lora_report.md
docs/query_preference_dpo_report.md
docs/query_candidate_rerank_report.md
docs/evidence_filter_report.md
outputs/lora_student_r8/final_adapter/
outputs/lora_student_r16/final_adapter/
outputs/lora_query_only_r16/final_adapter/
data/evaluation/post_lora_dmr_r8_lenient_v3/
data/evaluation/post_lora_dmr_r16_lenient_v3/
data/evaluation/post_lora_dmr_query_only_r16_smoke/
data/evaluation/phase_routed_dmr_evidence_only100/
data/evaluation/query_skeleton_dmr_evidence_only100/
data/evaluation/query_skeleton_dmr_evidence_only500/
data/evaluation/query_skeleton_dmr_r16_100/
data/evaluation/teacher_query_skeleton_dmr_approved500/
data/evaluation/teacher_query_skeleton_dmr_approved500_max3/
data/analysis/query_skeleton_gap/
data/trajectories/query_hard_negative_preferences.jsonl
data/trajectories/query_hard_positive_sft.jsonl
data/trajectories/query_hard_negative_preferences_zero_only.jsonl
outputs/query_hard_positive_sft_prepare_check/run_manifest.json
outputs/lora_query_hard_positive_r16/final_adapter/
outputs/lora_query_preference_zero_dpo_r16/final_adapter/
data/evaluation/query_skeleton_dmr_hard_positive500/
data/evaluation/query_skeleton_dmr_pref_zero_dpo500/
data/evaluation/query_candidate_rerank_query_only500_temp0_target3/
data/evaluation/query_candidate_rerank_lexical500_temp0_target5/
data/evaluation/evidence_filter_lexical500_k6/
data/analysis/query_skeleton_gap_hard_positive/
data/evaluation/oracle_dmr_lora_teacher_trace/
data/evaluation/teacher_query_hint_r16_all/
```

### Failure Audit

```text
docs/failure_audit_report.md
scripts/audit_post_lora_failures.py
scripts/analyze_query_skeleton_gap.py
scripts/export_query_preference_dataset.py
scripts/train_query_preference_dpo.py
scripts/eval_query_candidate_rerank_dmr.py
scripts/eval_evidence_filter_dmr.py
data/analysis/post_lora_failure_audit/
```

## Appendix B. LaTeX 변환용 핵심 표

### Table B1. Baseline DMR

| Condition | Loop completion | Format failure | Search rate | Containment |
| --- | ---: | ---: | ---: | ---: |
| Raw vanilla Llama pilot | `0/5` | `5` | n/a | n/a |
| Raw vanilla Mistral pilot | `0/5` | `5` | n/a | n/a |
| Strict-template Llama scaled | `481/500` | `19` | `0.8462` | `0.1954` |

### Table B2. Oracle Replay

| Condition | Model | Rows | Judge accuracy | Containment |
| --- | --- | ---: | ---: | ---: |
| Full-history | Llama-3-8B | `500` | not judged | `0.5080` |
| Full-history | Mistral-7B | `500` | not judged | `0.4240` |
| Teacher-trace Oracle | Llama-3-8B | `398` | `0.7337` | `0.4246` |
| Teacher-trace Oracle | Mistral-7B | `398` | `0.8769` | `0.4598` |

### Table B3. LoRA Distillation

| Condition | Eval loss | Token acc | Loop completion | Search rate | Containment | Judge accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA r8 | `0.9009` | `0.7546` | `489/500` | `0.7280` | `0.2474` | `0.4765` |
| LoRA r16 | `0.8694` | `0.7599` | `497/500` | `0.7948` | `0.2656` | `0.4809` |

### Table B3a. Query-Supervision Negative Results

| Condition | Eval loss | Token acc | Smoke outcome | Interpretation |
| --- | ---: | ---: | --- | --- |
| Query+answer r16 | `0.8583` | `0.7859` | `34/35` tool-call format failure | answer target 혼합이 tool-call surface를 무너뜨림 |
| Query-only r16 | `0.8244` | `0.7703` | first `5/5` behavioral failure | 초반 tool call은 생성하지만 JSON-as-content channel failure 발생 |
| Query-only r16 + vLLM rescue | n/a | n/a | `1/6` completed | vLLM parser layer만으로는 channel failure 구제 불충분 |
| Query-only r16 + endpoint proxy rescue | n/a | n/a | `2/20` completed, containment `0.0` | channel을 구제해도 stop-and-answer policy 부재 |

### Table B3b. LoRA Teacher-Trace Replay

| Condition | Rows | Containment | Judge accuracy |
| --- | ---: | ---: | ---: |
| LoRA r8 + teacher trace | `398` | `0.4874` | `0.8618` |
| LoRA r16 + teacher trace | `398` | `0.4899` | `0.8668` |

### Table B3c. LoRA Teacher-Query Hint

| Condition | Rows | Completed | Search rate | Containment | Judge accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA r16 + teacher-query hint | `398` | `394` | `0.9594` | `0.3452` | `0.6294` |

### Table B3d. Phase-Routed Query-Only Diagnostic

| Condition | Rows | Completion | Retrieved-reference rate | Containment | Mean searches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only search + r16 evidence-only answer | `100` | `100/100` | `0.09` | `0.11` | `2.22` |

### Table B3e. Deterministic Query Skeleton

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, skeleton raw 500 | `500` | `500/500` | `0.244` | `0.216` | `3.744` |
| Query-only r16, skeleton | `100` | `100/100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `100/100` | `0.16` | `0.17` | `2.86` |
| Base Llama skeleton smoke | `20` | `20/20` | `0.20` | `0.20` | `1.75` |

### Table B3f. Teacher Query Skeleton Replay

| Query source | Subset | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher full query chain | approved | `398` | `0.442` | `0.405` | `4.36` |
| Teacher full query chain | teacher-search only | `302` | `0.583` | `0.533` | `5.74` |
| Teacher max-3 query | approved | `398` | `0.367` | `0.342` | `3.25` |
| Teacher max-3 query | teacher-search only | `302` | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | same approved | `398` | `0.276` | `0.249` | `3.76` |
| Query-only r16 skeleton | teacher-search subset | `302` | `0.272` | `0.238` | `3.76` |

### Table B3g. Query Skeleton Row-Level Gap

| Subset | Category | Retrieval rows | Containment rows |
| --- | --- | ---: | ---: |
| Approved 398 | both | `74` | `63` |
| Approved 398 | teacher only | `72` | `73` |
| Approved 398 | student only | `36` | `36` |
| Approved 398 | neither | `216` | `226` |
| Teacher-search 302 | both | `74` | `63` |
| Teacher-search 302 | teacher only | `72` | `73` |
| Teacher-search 302 | student only | `8` | `9` |
| Teacher-search 302 | neither | `148` | `157` |

### Table B4. Failure Audit

| Failure type | r8 | r16 |
| --- | ---: | ---: |
| correct semantic | `233` | `239` |
| no search | `77` | `60` |
| searched wrong or insufficient evidence | `170` | `193` |
| evidence found but not used | `9` | `5` |
| tool-call format failure | `11` | `3` |
