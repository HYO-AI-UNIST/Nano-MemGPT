# 작은 오픈소스 LLM은 왜 MemGPT 기억 검색에 실패하는가?

GPT-4.1 Teacher Trajectory Distillation, Query Policy 진단, Candidate Reranking, Evidence Filtering을 통한 Nano-MemGPT v1 연구

> 이 문서는 Nano-MemGPT 연구의 v1 최종 통합본을 논문 초안 형태로 정리한 한국어 원고다. 이후 영어 LaTeX 논문으로 옮기기 전, 연구의 동기, 문제 정의, 실험 설계, 주요 결과, 실패 분석, 최종 주장을 하나의 흐름으로 읽을 수 있도록 재구성했다. 문장은 설명적인 논문체로 작성했다. 즉 MemGPT, tool calling, LoRA, DMR benchmark에 익숙하지 않은 독자도 논리의 출발점부터 따라올 수 있도록 배경 설명과 해석을 충분히 포함한다.

## 0. 초록

대화형 LLM agent가 장기 기억을 사용하려면 단순히 정답을 생성하는 능력만으로는 충분하지 않다. 모델은 사용자의 질문을 읽고, 과거 대화에서 어떤 문자열을 검색해야 할지 결정하고, `conversation_search`와 같은 도구를 올바른 schema로 호출하고, 검색 결과가 충분한지 판단한 뒤, evidence에 근거해 짧은 답변을 생성해야 한다. GPT-4급 모델에서는 이러한 MemGPT-style memory control이 비교적 자연스럽게 작동하는 것처럼 보이지만, Llama-3-8B나 Mistral-7B 같은 작은 오픈소스 모델에서도 같은 behavior가 안정적으로 나타나는지는 별도의 실증이 필요하다.

본 연구는 MSC Deep Memory Retrieval(DMR)을 중심으로 작은 오픈소스 LLM의 MemGPT 실패를 단계별로 분해한다. 먼저 vanilla Letta/MemGPT loop에서 Llama-3-8B와 Mistral-7B가 tool-call contract 자체를 안정적으로 통과하지 못함을 확인했다. 이후 strict-template adapter를 통해 tool-call surface를 보정하자 Llama는 500-row DMR에서 `481/500` loop completion을 달성했지만, containment는 `0.1954`에 머물렀다. 이는 schema formatting을 고치는 것만으로는 memory retrieval quality가 회복되지 않음을 보여 준다.

다음으로 GPT-4.1 teacher trajectory를 수집하고 judge가 승인한 398개 trajectory를 teacher-trace replay 조건으로 사용했다. 이 teacher trajectory는 gold oracle이 아니라 judge-filtered supervision이다. 실제로 teacher raw 500개 중 judge-approved row는 `398/500`이었지만, exact containment를 통과한 row는 `206/500`이었다. 따라서 본 연구는 teacher trajectory를 완전한 정답 경로가 아니라 작은 모델보다 강한 reference behavior로 취급한다. 이 조건에서 Frozen Llama-3-8B는 teacher evidence가 주어졌을 때 `292/398` judge accuracy를, Frozen Mistral-7B는 `349/398` judge accuracy를 기록했다. 이 결과는 작은 모델이 evidence를 받으면 답할 수 있으며, end-to-end 실패의 핵심이 answer generation 자체가 아니라 evidence acquisition, 특히 memory query selection에 있음을 시사한다.

승인된 teacher trajectory로 Llama-3-8B LoRA를 학습한 결과, r16 adapter는 500-row DMR에서 `497/500` loop completion과 GPT-4.1 judge accuracy `0.4809`를 달성했다. 그러나 같은 adapter에 teacher trace evidence를 직접 주면 judge accuracy가 `0.8668`까지 상승했다. 또한 teacher query hint만 제공해도 judge accuracy가 `0.6294`로 상승했다. 이러한 ablation은 LoRA distillation이 tool-use loop와 answer-from-evidence behavior를 일부 복구하지만, teacher 수준의 query policy를 자동으로 복구하지 못한다는 점을 보여 준다.

후속 실험에서는 query policy 병목을 더 좁혀 분석했다. Query-only LoRA는 end-to-end agent로는 실패했지만, tool-call shell을 deterministic wrapper가 담당하고 모델은 query string만 생성하게 하자 raw500에서 retrieved-reference `0.244`, containment `0.216`을 기록했다. Teacher query skeleton은 approved subset에서 retrieved-reference `0.367`, containment `0.342`를 기록해 여전히 큰 gap을 보였다. Row-level gap에서는 teacher-only retrieval success가 72개인 반면 student-only success는 8개뿐이었다. 이 hard set에 대해 hard-positive SFT와 zero-result-only DPO를 수행했으나, 두 방법 모두 hard rows 일부만 복구하고 raw distribution 성능을 낮췄다.

마지막으로 본 연구는 search-time candidate query generation과 retrieval-feedback reranking을 도입했다. Candidate lexical reranker는 raw500에서 retrieved-reference `0.246`, containment `0.224`를 기록해 query-only skeleton을 처음으로 소폭 개선했고, teacher-only hard72에서는 reference retrieval을 `25/72`, containment를 `22/72`까지 복구했다. 이후 answer 직전 evidence filtering을 적용하자 containment `0.224`를 유지하면서 mean retrieved를 `4.742`에서 `3.426`으로 줄였다.

종합하면, 작은 오픈소스 LLM의 MemGPT 복구는 단순 fine-tuning 문제가 아니다. Teacher trajectory distillation은 agent loop와 evidence-based answering을 복구하지만, robust memory retrieval을 위해서는 tool-call channel control, query selection, retrieval-feedback reranking, evidence filtering을 명시적으로 분리해야 한다. 이 연구는 그 실패 지점을 실험적으로 분해하고, query policy가 핵심 병목임을 정량적, 정성적으로 보인다.

키워드: MemGPT, Letta, long-term memory agent, tool calling, LoRA distillation, query policy, retrieval-feedback reranking, evidence filtering, Deep Memory Retrieval

## 0.1 핵심 주장 요약

이 연구가 답하고자 한 질문은 다음처럼 간단히 표현할 수 있다.

```text
GPT-4급 모델에서는 MemGPT-style memory agent가 잘 되는 것처럼 보인다.
그렇다면 Llama-3-8B나 Mistral-7B 같은 작은 오픈소스 모델에서도 같은 방식이 가능한가?
가능하지 않다면 실패 지점은 tool-call format인가, retrieval planning인가, evidence use인가?
그리고 이 실패를 teacher trajectory distillation과 search-time controller로 얼마나 복구할 수 있는가?
```

최종 답은 다음과 같다.

```text
작은 오픈소스 LLM의 MemGPT 실패는 단순히 "모델이 답을 모른다"는 문제가 아니다.
가장 큰 병목은 memory query selection이다.

LoRA distillation은 tool-call loop와 answer-from-evidence behavior를 일부 복구하지만,
teacher 수준의 memory search policy를 자동으로 복구하지 못한다.

단일 query를 한 번 생성하게 하는 것보다,
여러 candidate query를 만들고 local retrieval 결과를 보고 고르는 search-time controller가
teacher-only hard failure class를 훨씬 잘 복구한다.

마지막으로 evidence filtering은 정답률을 더 올리지는 못했지만,
같은 containment를 유지하면서 answer model이 읽어야 하는 evidence 양을 줄였다.
```

숫자로 요약하면 다음과 같다.

| 단계 | 핵심 결과 | 의미 |
| --- | ---: | --- |
| Raw vanilla Llama | `0/5` loop 완료 | 원시 작은 모델은 MemGPT tool-call contract부터 실패 |
| Strict-template Llama | `481/500` loop 완료, containment `0.1954` | tool-call 표면만 고쳐도 loop는 돌지만 정확도는 낮음 |
| GPT-4.1 teacher | `398/500` judge 통과, containment `206/500` | 강한 teacher지만 gold oracle은 아님. judge-filtered supervision으로 사용 |
| Frozen Llama + teacher trace | `292/398` judge 통과 | evidence가 주어지면 작은 모델도 상당히 답함 |
| Frozen Mistral + teacher trace | `349/398` judge 통과 | Mistral은 evidence replay 조건에서 매우 강함 |
| Llama LoRA r16 end-to-end | judge `0.4809` | trajectory distillation은 loop를 복구하지만 teacher replay에는 못 미침 |
| Llama LoRA r16 + teacher query hint | judge `0.6294` | 좋은 query만 알려줘도 성능이 크게 오름 |
| Llama LoRA r16 + teacher trace | judge `0.8668` | teacher evidence가 있으면 LoRA model은 매우 잘 답함 |
| Query-only deterministic skeleton | retrieved-reference `0.244`, containment `0.216` | query-only 학습에는 유효한 signal이 있음 |
| Candidate lexical rerank | retrieved-reference `0.246`, containment `0.224` | 처음으로 raw500에서 query-only skeleton을 소폭 개선 |
| Candidate lexical rerank hard72 | retrieval `25/72`, containment `22/72` | teacher-only hard failure class를 크게 복구 |
| Evidence filter top-6 | containment `0.224`, mean retrieved `3.426` | 정답률 유지하면서 evidence volume 감소 |

이 결과는 "LoRA를 더 크게 돌리면 해결된다"는 단순한 결론과 다르다. 작은 모델의 MemGPT 복구에는 최소 네 가지 층을 분리해야 한다.

1. Tool-call channel control: 모델이 schema-valid tool call을 내는가.
2. Search phase policy: 어떤 query를 던질지 아는가.
3. Evidence selection/filtering: 검색 결과 중 어떤 message를 읽어야 하는가.
4. Evidence-grounded answer generation: evidence를 보고 짧고 정확한 답을 만드는가.

이 연구는 v1에서 1, 2, 3번을 충분히 진단했고, 4번은 teacher trace replay와 failure audit을 통해 상대적으로 작은 병목이라는 근거를 얻었다. 따라서 v1은 여기서 실험 범위를 고정하고, 이후 작업은 논문 작성, figure/table 정리, 대표 예시 선정으로 전환한다.

## 0.2 논문 구성

이 문서는 다음 순서로 전개된다. 1장과 2장은 MemGPT-style memory control과 DMR 태스크를 설명한다. 3장과 4장은 실험 환경과 연구 질문을 정리한다. 5장은 vanilla 및 strict-template baseline을 통해 작은 모델이 어디서 처음 실패하는지 보인다. 6장은 GPT-4.1 teacher trajectory와 teacher-trace replay를 통해 evidence가 주어졌을 때 작은 모델이 답할 수 있음을 보인다. 7장과 8장은 LoRA distillation 결과와 post-LoRA failure audit을 통해 남은 병목이 query/evidence acquisition임을 밝힌다. 9장부터 13장까지는 query-only SFT, phase routing, deterministic query skeleton, teacher query replay, hard query SFT/DPO, candidate reranking, evidence filtering으로 이어지는 일련의 진단 실험을 정리한다. 14장 이후는 전체 결과를 종합하고, 논문 기여점, 한계, 향후 연구, 최종 결론을 제시한다.

## 1. 연구 배경

### 1.1 MemGPT가 하려는 일

LLM이 실제 개인 비서나 장기 대화 agent로 쓰이려면, 모델은 현재 context window 안에 있는 정보만 처리해서는 부족하다. 사용자는 종종 며칠 전, 몇 주 전, 혹은 훨씬 이전 대화에서 언급한 개인적 사실을 다시 묻는다. 예를 들어 "전에 말했던 네가 좋아한다던 가수가 누구였지?", "네가 일한다고 했던 패스트푸드점이 어디였지?", "돈을 아끼려고 한다고 했던 습관이 뭐였지?" 같은 질문은 단순한 일반 지식 문제가 아니다. 모델은 과거 대화 기록에서 해당 사실이 포함된 utterance를 찾아야 한다.

MemGPT 계열 시스템은 이 문제를 LLM 자체의 context window만으로 해결하지 않고, LLM을 memory를 관리하는 agent처럼 사용한다. 사용자가 "예전에 우리가 말했던 그 사람 이름 뭐였지?"라고 물으면, 모델은 즉시 답을 생성하는 대신 다음과 같은 과정을 수행해야 한다.

```text
1. 사용자의 질문을 이해한다.
2. 과거 대화 기억에서 어떤 단어를 검색해야 할지 정한다.
3. conversation_search 같은 도구를 호출한다.
4. 검색 결과를 읽는다.
5. 결과가 부족하면 추가 query를 던진다.
6. 충분한 evidence가 모이면 최종 답변을 보낸다.
```

예를 들어 사용자가 이렇게 묻는다고 하자.

```text
Hey, remember that time we talked about music?
What was the artist you mentioned you could get into?
```

정답 memory는 과거 대화 어딘가에 다음처럼 있을 수 있다.

```text
A little bit. I can get into Taylor Swift.
```

여기서 좋은 memory agent는 처음에 `music` 같은 broad query를 던질 수도 있지만, 검색 결과가 부족하면 `get into`, `Taylor Swift`처럼 과거 utterance에 실제로 등장할 가능성이 높은 더 구체적인 문자열로 좁혀야 한다. 그리고 충분한 evidence를 찾은 뒤에는 search process를 설명하지 않고, 짧게 `Taylor Swift`라고 답해야 한다.

이처럼 MemGPT-style memory control은 일반적인 question answering보다 복합적이다. 모델은 자연어 이해, 도구 호출, 검색어 생성, 검색 결과 해석, 답변 생성이라는 여러 하위 문제를 순서대로 해결해야 한다. 따라서 전체 성능이 낮을 때 어느 하위 문제가 병목인지 분해하지 않으면, 단순히 "모델이 약하다"는 결론밖에 얻을 수 없다.

### 1.2 왜 작은 모델에서 어려운가

GPT-4급 모델은 이런 절차를 어느 정도 자연스럽게 수행한다. 강한 instruction-following 능력과 넓은 world knowledge, tool-use prior, 긴 reasoning chain을 바탕으로 query를 수정하고 검색 결과를 해석한다. 그러나 작은 오픈소스 모델은 같은 환경에서 훨씬 더 취약하다. 특히 agent loop에서는 한 번의 자연어 답변이 아니라 여러 번의 structured action을 안정적으로 이어가야 하므로, 작은 오류가 전체 loop 실패로 증폭된다.

작은 모델이 해결해야 하는 요구는 다음처럼 나눌 수 있다.

| 요구 | 어려운 이유 |
| --- | --- |
| 도구 호출 형식 지키기 | OpenAI `tool_calls` schema, function name, arguments 구조를 정확히 맞춰야 함 |
| 검색어 생성 | 질문에 나온 표현이 아니라 과거 memory에 실제로 등장할 literal substring을 골라야 함 |
| 반복 검색 제어 | 첫 검색이 부족하면 다른 query를 던지고, 충분하면 멈춰야 함 |
| evidence 해석 | 검색 결과에 여러 distractor가 섞여도 정답 utterance를 골라야 함 |
| 최종 답변 경계 | 계속 검색할지 답할지 결정하고, reasoning/tool intent를 답변에 누출하지 않아야 함 |

즉 작은 모델의 실패를 "지식이 부족해서"라고만 보면 안 된다. 실제 실패는 tool channel, query policy, retrieval result interpretation, final answer boundary가 엉켜서 나타난다. 예를 들어 모델이 정답을 생성할 수 있더라도 정답 evidence를 검색하지 못하면 틀린 답을 낸다. 반대로 좋은 evidence를 찾아도 final answer 단계에서 tool-use intent가 새면 사용자가 보는 답변은 실패로 기록된다. 그러므로 이 연구의 핵심은 end-to-end accuracy 하나를 높이는 것이 아니라, 실패를 구조적으로 분해하는 것이다.

### 1.3 연구 제안서의 중심 가설

원래 proposal의 중심 가설은 다음 두 가지였다.

```text
가설 1. 작은 모델도 teacher가 수행한 trajectory를 보면 MemGPT식 memory behavior를 배울 수 있다.
가설 2. 실패 원인을 tool-call format, retrieval planning, evidence use로 분해하면 어떤 모듈이 병목인지 알 수 있다.
```

이 연구는 처음에는 vanilla MemGPT 재현에서 시작했지만, 진행하면서 더 구체적인 결론에 도달했다.

```text
Teacher trajectory distillation은 필요하지만 충분하지 않다.
가장 큰 병목은 teacher의 query policy를 작은 모델이 그대로 일반화하지 못한다는 점이다.
```

따라서 본 연구는 단일한 학습 실험으로 끝나지 않는다. 먼저 vanilla와 strict-template 조건으로 tool-call compatibility를 확인하고, teacher-trace replay로 answer-from-evidence 능력을 측정하고, LoRA distillation으로 end-to-end behavior를 복구한 뒤, failure audit과 여러 query-specific diagnostic으로 남은 병목을 좁혀 간다. 이러한 순차적 설계가 이 논문의 방법론적 핵심이다.

## 2. 평가 태스크와 용어

### 2.1 Deep Memory Retrieval

핵심 평가는 MSC Deep Memory Retrieval, 줄여서 DMR이다. DMR은 모델이 긴 multi-session conversation history에서 특정 개인적 사실을 찾아 답하는 태스크다.

각 row는 대략 다음 요소를 가진다.

| 요소 | 의미 |
| --- | --- |
| previous dialogs | 과거 대화 세션들 |
| current dialog | 현재 대화 |
| probe | 사용자 memory question |
| reference answer | 정답 문자열 |
| personas | Speaker 1, Speaker 2의 persona fact |

DMR이 어려운 이유는 probe가 항상 정답 literal을 직접 포함하지 않기 때문이다. 예를 들어 probe는 "artist you could get into"라고 묻지만 정답은 `Taylor Swift`일 수 있다. 따라서 모델은 질문의 의미를 바탕으로 memory에 있을 법한 literal phrase를 찾아야 한다.

### 2.2 Paper-Substring Contract

초기 실험에서 중요한 구현 문제가 있었다. 현재 Letta/MemGPT의 최신 코드와 원 논문 당시의 검색 조건은 완전히 같지 않다. 그래서 이 연구에서는 DMR recall을 다음처럼 정의했다.

```text
conversation_search(query):
  과거 대화 message content에서 query가 case-insensitive substring으로 등장하는 message를 반환한다.
```

이를 `paper_substring` 또는 `Paper-Substring Contract`라고 부른다.

이 계약은 매우 중요하다. substring 검색에서는 `audio studio location`처럼 긴 query가 과거 utterance에 그대로 없으면 실패한다. 반대로 `studio`, `California`, `Santa Barbara`처럼 실제로 등장할 가능성이 높은 짧은 literal query는 성공 가능성이 높다.

### 2.3 주요 metric

이 연구에서 사용한 metric은 다음과 같다.

| Metric | 의미 | 주의점 |
| --- | --- | --- |
| loop completion | agent loop가 중간 오류 없이 끝났는가 | 답이 맞는지와 별개 |
| format failure | tool-call schema나 channel 문제로 실패했는가 | 작은 모델에서 매우 중요 |
| search rate | `conversation_search`를 호출한 row 비율 | 높다고 항상 좋은 것은 아님 |
| retrieved-reference rate | retrieved evidence 안에 reference string이 포함되는가 | retrieval quality proxy |
| containment | final answer에 reference normalized string이 포함되는가 | exact/substring 기반이라 엄격함 |
| ROUGE-L recall | answer와 reference의 longest common subsequence recall | 설명형 답변에는 유용하지만 과신 금물 |
| GPT-4.1 judge accuracy | semantic judge가 정답으로 인정했는가 | 비용이 들지만 containment보다 의미적으로 관대 |

특히 containment와 GPT judge는 다르다. 예를 들어 reference가 `Three miles!`이고 모델이 `a three-mile walk`라고 답하면 의미상 맞지만, 단순 normalization containment에서는 false가 될 수 있다. 그래서 이 연구는 containment와 judge를 함께 보았다.

## 3. 실험 환경

### 3.1 모델

주요 student model은 다음과 같다.

| 표기 | checkpoint | 역할 |
| --- | --- | --- |
| Llama-3-8B | `NousResearch/Meta-Llama-3-8B-Instruct` | 핵심 student, LoRA 학습 대상 |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | frozen teacher-evidence replay 비교 대상 |
| GPT-4.1 | `gpt-4.1-2025-04-14` 계열 | teacher trajectory 생성과 judge |

Llama-3-8B는 full fine-tuning이 아니라 LoRA로 학습했다. 실제 환경에서 8B 모델 full fine-tuning은 VRAM과 optimizer state 때문에 부담이 매우 크다. 반면 LoRA는 일부 low-rank adapter만 학습하므로 실험적으로 가능했다.

### 3.2 인프라

실험은 Docker 기반으로 구성했다.

| 구성 요소 | 역할 |
| --- | --- |
| `nano-memgpt-dev` | 데이터 처리, 평가 script 실행 |
| `llama-vllm` | Llama base 및 LoRA adapter serving |
| `letta-server` | MemGPT/Letta agent loop |
| `pgvector` | Letta backend storage |

vLLM serving에서는 base model과 LoRA adapter를 함께 노출했다.

| vLLM model id | 의미 |
| --- | --- |
| `nano-memgpt-llama3-r8` | teacher trajectory SFT LoRA rank 8 |
| `nano-memgpt-llama3-r16` | teacher trajectory SFT LoRA rank 16 |
| `nano-memgpt-llama3-query-only-r16` | query-call target만 학습한 LoRA |
| `nano-memgpt-llama3-query-hard-positive-r16` | hard-positive query SFT LoRA |
| `nano-memgpt-llama3-query-pref-zero-dpo-r16` | zero-result-only preference DPO LoRA |

## 4. 연구 질문

최종 연구 질문은 다섯 개로 정리할 수 있다.

### RQ1. 작은 오픈소스 모델은 vanilla MemGPT control loop를 실행할 수 있는가?

이 질문은 가장 기본적인 compatibility gate다. 모델이 tool-call 형식을 지키지 못하면 memory search 능력을 평가할 수 없다.

### RQ2. tool-call surface format을 고치면 retrieval quality도 회복되는가?

strict template adapter로 tool-call schema를 강제로 맞추면 loop completion은 오를 수 있다. 하지만 이게 곧 좋은 query와 좋은 답변을 의미하지는 않는다.

### RQ3. teacher evidence가 주어지면 frozen student는 답할 수 있는가?

작은 모델이 오답을 내는 이유가 "답을 생성할 능력이 없어서"인지, "evidence를 못 찾아서"인지 분리해야 한다. teacher-trace replay는 이 질문에 답한다.

### RQ4. teacher trajectory LoRA distillation은 end-to-end MemGPT behavior를 복구하는가?

GPT teacher가 실제로 수행한 tool call과 answer를 SFT 데이터로 만들어 LoRA를 학습하면 작은 모델이 MemGPT behavior를 배울 수 있는지 본다.

### RQ5. LoRA 이후에도 남는 병목은 무엇인가?

LoRA 후에도 정확도가 낮다면, 남은 병목이 query selection인지, evidence filtering인지, final answer generation인지 분해해야 한다.

### 4.1 실험 설계 원칙

본 연구의 실험은 단순히 여러 모델과 여러 prompt를 비교하는 방식이 아니다. 각 실험은 이전 실험에서 남은 ambiguity를 줄이도록 설계되었다. 예를 들어 vanilla MemGPT가 실패했을 때, 그 원인이 tool-call schema인지 query selection인지 알 수 없으므로 strict-template adapter를 둔다. Strict-template으로도 정확도가 낮으면, 모델이 evidence를 받아도 답하지 못하는지 확인하기 위해 teacher-trace replay를 둔다. Teacher evidence를 받으면 답할 수 있음이 확인되면, LoRA distillation으로 behavior를 학습시키고, 그 뒤에도 남는 gap은 teacher-query hint와 teacher-evidence replay로 다시 나눈다.

이러한 설계는 다음과 같은 계단식 진단 구조를 가진다.

| 단계 | 제거하려는 불확실성 | 실험 |
| --- | --- | --- |
| Tool-call compatibility | 작은 모델이 loop 자체를 못 도는가 | raw vanilla, strict-template |
| Answer capacity | evidence가 있어도 답을 못 하는가 | full-history, teacher-trace replay |
| Trajectory imitation | teacher behavior를 LoRA로 배울 수 있는가 | r8/r16 teacher trajectory SFT |
| Query bottleneck | 좋은 query가 주어지면 성능이 오르는가 | teacher-query hint, teacher query skeleton |
| Query policy signal | query-only 학습에 유효한 signal이 있는가 | query-only LoRA, phase-routed, deterministic skeleton |
| Hard failure recovery | teacher-only failure row를 복구할 수 있는가 | hard-positive SFT, zero-DPO, candidate rerank |
| Context efficiency | 검색 결과가 많을 때 evidence budget을 줄일 수 있는가 | evidence filtering |

따라서 각 결과는 단독 숫자가 아니라 다음 실험의 조건을 결정하는 진단 신호로 해석해야 한다. 이 점이 이 연구를 단순 benchmark report가 아니라 failure decomposition study로 만든다.

## 5. 실험 1: Vanilla MemGPT Baseline

### 5.1 Raw Vanilla 결과

먼저 Llama-3-8B와 Mistral-7B를 원시 serving 조건으로 Letta/MemGPT loop에 넣었다. 결과는 좋지 않았다.

| 모델 | 평가 행 | 완료 loop | 행동 실패 |
| --- | ---: | ---: | ---: |
| Llama-3-8B | 5 | 0 | 5 |
| Mistral-7B | 5 | 0 | 5 |

이 단계의 실패는 "정답을 못 맞힘" 이전의 문제다. 모델이 현재 Letta가 기대하는 OpenAI-compatible tool call 형식을 안정적으로 만들지 못했다. 따라서 raw vanilla만으로 작은 모델의 memory reasoning 능력을 평가할 수 없었다.

### 5.2 Strict Template Adapter

다음으로 strict template adapter를 붙였다. 이 adapter는 모델 weight를 바꾸지 않고, 모델이 내는 search intent를 schema-valid `conversation_search` tool call로 변환한다.

5-row pilot에서 Llama는 loop에 들어갔다.

| 모델 | 평가 행 | 완료 loop | ROUGE-L recall | containment | search rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 5 | 5 | 0.6323 | 0.4000 | 0.8000 |
| Mistral-7B | 5 | 0 | n/a | n/a | n/a |

이후 Llama strict-template을 500-row로 확장했다.

| 평가 행 | 완료 loop | 행동 실패 | ROUGE-L recall | containment | search rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 481 | 19 | 0.3929 | 0.1954 | 0.8462 |

해석은 분명하다. Strict template은 tool-call surface를 상당히 복구하지만, 정확도는 낮다. 즉 작은 모델의 문제는 schema formatting만이 아니다. 모델은 검색을 자주 하지만, 검색어가 정답 memory를 안정적으로 찾지 못한다.

### 5.3 Document-QA Proxy

원 proposal에는 Document-QA 평가도 포함되어 있었다. 하지만 원 논문 수준의 20M Wikipedia index를 로컬에 완전히 재현하지 못했기 때문에, 이 연구에서는 Document-QA를 core claim이 아니라 context-pack proxy로 다루었다.

Llama-3-8B의 proxy 결과는 다음과 같았다.

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

Mistral-7B의 proxy 결과도 비슷하게 gold evidence가 있으면 답변이 가능하지만 retrieval-only 조건에서는 약했다.

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

이 proxy는 DMR과 같은 결론을 보강한다. 작은 모델은 evidence가 주어지면 답변할 수 있지만, retrieval이 약하면 전체 성능이 무너진다.

## 6. Teacher Trajectory와 Oracle Replay

### 6.1 왜 teacher trajectory가 필요한가

Vanilla/strict-template 결과만으로는 작은 모델이 무엇을 배워야 하는지 알기 어렵다. 그래서 GPT-4.1 teacher를 사용해 DMR row마다 실제 memory-search trajectory를 수집했다.

Teacher trajectory에는 다음이 포함된다.

| 항목 | 의미 |
| --- | --- |
| query chain | teacher가 호출한 `conversation_search` query 목록 |
| tool output | 각 query의 검색 결과 |
| final answer | teacher의 최종 답변 |
| judge decision | GPT-4.1 judge가 teacher answer를 정답으로 인정했는지 |

Teacher가 틀린 trajectory까지 학습시키면 student가 잘못된 behavior를 배울 수 있다. 그래서 judge가 승인한 trajectory만 LoRA 데이터로 사용했다. 다만 이 선택은 teacher trajectory를 gold로 간주한다는 뜻이 아니다. Judge-approved trajectory는 강한 teacher가 만든 고품질 후보군이지만, automatic judge와 exact containment 사이에는 차이가 있으며, 일부 row에는 paraphrase, no-search answer, memory patch, hedged answer가 섞일 수 있다.

### 6.2 20-row corrected pilot

초기 GPT-4.1 20-row pilot에서는 judge accuracy가 `17/20`이었다. 승인된 17개 trace를 frozen student에게 replay했다.

| Model | Approved traces | Completed | Judge accuracy | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 17 | 17 | `14/17` (`0.8235`) | 0.6590 | 0.5294 |
| Mistral-7B | 17 | 17 | `15/17` (`0.8824`) | 0.7372 | 0.4706 |

이 결과는 매우 중요하다. Vanilla MemGPT loop에서는 실패하던 모델도 teacher evidence를 주면 대부분 답한다. 즉 작은 모델이 완전히 무능한 것이 아니라, memory management 단계가 병목이다.

### 6.3 500-row scaled teacher collection

이후 500-row scaled trajectory를 수집했다.

| Metric | Value |
| --- | ---: |
| raw teacher rows | `500` |
| completed | `500` |
| GPT-4.1 judge accuracy | `398/500` (`0.7960`) |
| ROUGE-L recall | `0.7393` |
| containment | `0.4120` |
| search rate | `0.7740` |

승인된 `398`행이 이후 teacher-trace replay와 LoRA 학습의 기반이 되었다.

### 6.3.1 Teacher trajectory 품질 감사

500-row teacher collection 이후, judge-approved trajectory가 정말 학습 supervision으로 충분히 깨끗한지 별도로 감사했다. 핵심 문제는 containment와 semantic judge 사이의 큰 차이다. Teacher answer의 exact/reference-string containment는 `206/500`(`0.4120`)에 그쳤지만, GPT-4.1 judge는 `398/500`(`0.7960`)을 정답으로 승인했다.

이를 교차 분석하면 다음과 같다.

| Containment | Judge | Rows | 해석 |
| --- | --- | ---: | --- |
| Pass | Correct | `204` | exact string과 semantic judge가 모두 맞다고 본 가장 깨끗한 row |
| Pass | Incorrect | `2` | reference string은 포함했지만 문맥상 오답 또는 contradiction 포함 |
| Fail | Correct | `194` | exact string은 없지만 judge가 의미상 정답으로 인정한 row |
| Fail | Incorrect | `100` | teacher answer 실패 또는 retrieval/answer 실패 |

특히 `Containment = Fail`, `Judge = Correct`인 194개를 감사했다. 이 row들을 모두 폐기할 필요는 없었다. 상당수는 `I have a cat.`과 `I have several cats`, `Gray!`와 `grey`, `I was three!`와 `three years old`처럼 exact string 기준만 통과하지 못한 정상 paraphrase였다. 하지만 194개 전체가 같은 품질은 아니었다.

| 분류 | Rows | 의미 | 권장 처리 |
| --- | ---: | --- | --- |
| Clean search paraphrase | `108` | 검색을 수행했고 memory patch 없이 답변도 명확한 paraphrase | answer/evidence supervision과 query supervision에 비교적 안전 |
| No-search correct | `47` | teacher가 `conversation_search` 없이 core/persona/immediate context에서 답함 | answer-only에는 가능하지만 query-policy 학습에는 부적합 |
| Search with memory patch | `11` | 검색 후 `memory_apply_patch`를 호출함 | full trajectory SFT에는 노이즈가 될 수 있음 |
| Search noisy or lenient | `28` | hedging, "couldn't find exact", judge lenient language, 부분 정답 포함 | high-quality teacher set에서는 보류 또는 수동 검토 권장 |

이 감사 결과는 본 연구의 teacher 사용 방식을 더 보수적으로 만든다. 본 연구는 GPT-4.1 trajectory를 완전한 oracle로 부르지 않는다. 대신 `judge-filtered teacher supervision`으로 사용하며, query-policy 분석에서는 no-search row, memory-patch row, hedged/lenient row를 별도로 구분한다. 자세한 감사 결과는 `docs/teacher_containment_mismatch_audit.md`에 정리했다.

### 6.4 Scaled Teacher-Trace Replay

승인된 398개 teacher trace를 frozen Llama와 Mistral에 replay했다. 이 조건에서는 student가 직접 검색하지 않는다. teacher가 찾은 evidence를 주고 final answer만 생성하게 한다. 여기서 `replay`는 answer-from-evidence 능력을 측정하기 위한 diagnostic condition이며, teacher trace가 gold-optimal trajectory라는 뜻은 아니다.

| Model | Completed | Judge accuracy | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: |
| Llama-3-8B | `398/398` | `292/398` (`0.7337`) | `0.6597` | `0.4246` |
| Mistral-7B | `398/398` | `349/398` (`0.8769`) | `0.7208` | `0.4598` |

이 결과는 연구의 첫 번째 큰 결론이다.

```text
작은 모델은 teacher evidence가 있으면 DMR answer를 상당히 잘 생성한다.
따라서 end-to-end 실패의 큰 부분은 final answer 능력 부족이 아니라
teacher 수준의 memory search behavior를 스스로 수행하지 못하는 데 있다.
```

Mistral이 vanilla Letta gate에서는 실패했지만 teacher evidence replay에서는 매우 강했다는 점도 중요하다. tool-call compatibility와 answer-from-evidence ability는 별개의 능력이다.

## 7. LoRA Teacher-Trajectory Distillation

### 7.1 학습 데이터 구성

승인된 teacher trajectory 398개에서 context-complete SFT step을 만들었다. 하나의 DMR row는 여러 search call과 final answer를 포함하므로, row 수보다 SFT step 수가 많다. 다만 6.3.1의 감사 결과처럼 이 398개는 gold trajectory가 아니라 judge-filtered supervision이다. 따라서 기존 r8/r16 LoRA는 강한 teacher behavior를 모방하도록 학습되었지만, 일부 no-search row, memory patch row, hedged paraphrase row가 섞인 noisy supervision의 영향을 받을 수 있다.

| 항목 | 값 |
| --- | ---: |
| 승인 teacher trajectory | `398` |
| LoRA용 context-complete SFT step | `1,664` |
| median total tokens | `2,464` |
| p95 total tokens | `4,677` |
| max total tokens | `8,980` |
| 8,192 token 초과 | `3` |

학습 target은 단순 final answer가 아니었다. search step에서는 teacher의 `conversation_search` call을 target으로 두고, answer step에서는 final answer를 target으로 두었다. 즉 student가 MemGPT control loop 전체를 imitation하도록 했다.

### 7.2 LoRA 설정

두 가지 LoRA rank를 학습했다.

| Condition | rank | alpha | final train loss | final eval loss | final token accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA r8 | 8 | 16 | 1.0890 | 0.9009 | 0.7546 |
| LoRA r16 | 16 | 32 | 1.0270 | 0.8694 | 0.7599 |

r16은 r8보다 eval loss와 token accuracy가 조금 좋았다. 하지만 나중에 보듯 semantic judge accuracy는 거의 같았다. 이는 token-level imitation이 end-to-end memory success와 완전히 같지 않다는 점을 보여 준다.

### 7.3 Post-training DMR 평가

LoRA adapter를 vLLM에 mount하고 500-row DMR을 다시 평가했다.

| Condition | Loop completion | Format failure | Search rate | ROUGE-L recall | containment | GPT-4.1 judge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA r8 | `489/500` | `11` | `0.7280` | `0.5489` | `0.2474` | `0.4765` |
| LoRA r16 | `497/500` | `3` | `0.7948` | `0.5515` | `0.2656` | `0.4809` |

Strict-template Llama와 비교하면 LoRA는 loop stability와 containment를 개선했다.

| Condition | Loop completion | Format failure | Search rate | containment | Semantic judge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw vanilla Llama pilot | `0/5` | `5` | n/a | n/a | n/a |
| Strict-template Llama scaled | `481/500` | `19` | `0.8462` | `0.1954` | not run |
| LoRA r8 | `489/500` | `11` | `0.7280` | `0.2474` | `0.4765` |
| LoRA r16 | `497/500` | `3` | `0.7948` | `0.2656` | `0.4809` |

LoRA는 분명히 효과가 있었다. tool-call format failure가 줄고, loop completion이 올라가고, 정답률도 strict-template보다 좋아졌다. 그러나 judge accuracy `0.4809`는 teacher trace replay `0.8668`과 큰 차이가 난다.

### 7.4 LoRA Teacher-Trace Replay

LoRA 모델에게 teacher evidence를 직접 주면 어떻게 되는지 평가했다.

| Condition | Evidence source | Student controls search? | Rows | containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: |
| LoRA r8 end-to-end | student retrieval | Yes | `489` completed | `0.2474` | `0.4765` |
| LoRA r8 + teacher trace | teacher evidence | No | `398` approved | `0.4874` | `0.8618` |
| LoRA r16 end-to-end | student retrieval | Yes | `497` completed | `0.2656` | `0.4809` |
| LoRA r16 + teacher trace | teacher evidence | No | `398` approved | `0.4899` | `0.8668` |

이 결과는 LoRA가 answer-from-evidence behavior도 배웠음을 보여 준다. 같은 r16 LoRA라도 student가 직접 검색하면 judge `0.4809`에 머물지만, teacher trace evidence를 받으면 `0.8668`까지 오른다.

즉 남은 gap은 주로 search/evidence acquisition이다.

### 7.5 LoRA Teacher-Query Hint

다음으로 teacher evidence 전체가 아니라 teacher query chain만 hint로 주었다. 이 조건에서는 student가 여전히 직접 search를 실행하고, search result를 읽고 답한다.

| Condition | Search source | Evidence source | Rows judged | Search rate | containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LoRA r16 end-to-end | student | student retrieval | `497` | `0.7948` | `0.2656` | `0.4809` |
| LoRA r16 + teacher-query hint | teacher query hint + student execution | student retrieval | `394` | `0.9594` | `0.3452` | `0.6294` |
| LoRA r16 + teacher trace | teacher trace | teacher evidence replay | `398` | n/a | `0.4899` | `0.8668` |

Teacher query hint는 성능을 크게 올렸다. 이는 query selection이 중요한 병목이라는 강한 증거다. 하지만 teacher trace replay에는 아직 못 미친다. query만 좋아져도 검색 실행, result interpretation, final answer boundary에서 추가 손실이 생긴다.

## 8. Post-LoRA Failure Audit

### 8.1 왜 failure audit이 필요한가

LoRA r16의 judge accuracy는 `0.4809`였다. 이 숫자만 보면 "아직 안 좋다"라고만 말할 수 있다. 하지만 연구적으로 중요한 것은 어떤 실패가 남았는지다.

Failure audit은 각 row를 다음 feature로 분해했다.

| Feature | 의미 |
| --- | --- |
| `searched` | `conversation_search`를 호출했는가 |
| `queries` | 실제 검색 query |
| `evidence_contains_reference` | tool output 안에 reference string이 포함되는가 |
| `semantic_correct` | GPT-4.1 judge가 정답으로 인정했는가 |
| `surface_issue` | final answer에 `Thinking`, `Let me search` 같은 실행 의도가 새었는가 |

### 8.2 Audit Summary

| Run | Rows | OK | Judge acc | Lexical acc | Search rate | Evidence-hit rate | Evidence-hit among wrong | Mean searches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r8 | 500 | 489 | 0.4765 | 0.2474 | 0.7280 | 0.1534 | 0.0352 | 1.3701 |
| r16 | 500 | 497 | 0.4809 | 0.2656 | 0.7948 | 0.1489 | 0.0194 | 1.7022 |

r16은 r8보다 검색을 더 많이 하지만 evidence-hit rate는 오히려 조금 낮다. 즉 검색을 더 자주 하는 것만으로는 충분하지 않다. query quality가 중요하다.

### 8.3 Failure Type 분포

| Failure type | r8 | r16 | 의미 |
| --- | ---: | ---: | --- |
| `correct_semantic` | `233` | `239` | GPT-4.1 judge가 정답으로 인정 |
| `no_search` | `77` | `60` | 검색 없이 답하거나 검색 의도만 말함 |
| `searched_wrong_or_insufficient_evidence` | `170` | `193` | 검색했지만 exact reference evidence가 없음 |
| `evidence_found_but_not_used` | `9` | `5` | reference evidence가 있었지만 judge 기준 오답 |
| `tool_call_format_failure` | `11` | `3` | tool-call loop 실패 |

r16 judge 오답 `258`개 중 `193`개, 약 `74.8%`가 `searched_wrong_or_insufficient_evidence`였다. 즉 가장 큰 병목은 answer generation보다 query/evidence retrieval이다.

### 8.4 대표 실패 예시: query가 너무 broad한 경우

Probe:

```text
Hey, remember that time we talked about music?
What was the artist you mentioned you could get into?
```

Reference:

```text
Taylor Swift!
```

r16 query:

```text
music
```

검색 결과에는 `country music` 관련 문장만 들어왔고, `Taylor Swift`가 들어 있는 utterance는 들어오지 않았다. 모델은 이후 Chris Stapleton/Jason Aldean을 답했다.

이 row는 final answer 문제가 아니다. 정답 memory를 못 가져왔기 때문에 모델이 맞힐 수 없었다.

### 8.5 대표 실패 예시: evidence가 있었지만 못 쓴 경우

Reference:

```text
The drums!
```

r16 query:

```text
What instrument
play
```

검색 결과 중 하나에 `I play the drums`가 있었지만 모델은 guitar를 답했다. 이런 경우는 `evidence_found_but_not_used`다.

하지만 이 유형은 r16에서 `5`개뿐이었다. 따라서 존재는 하지만 전체 병목의 중심은 아니다.

### 8.6 실행 의도 누출

일부 row에서는 final answer가 실제 답변이 아니라 다음처럼 끝났다.

```text
Let me check our past conversations...
```

또는:

```text
Thinking aloud: ...
```

이는 학습 데이터의 tool-use chain을 imitation하면서 final answer boundary가 완전히 분리되지 않았다는 신호다. 다만 가장 큰 병목은 여전히 query/evidence retrieval이었다.

## 9. Query Supervision 실험들

Failure audit 이후 연구는 query selection 병목을 직접 겨냥했다.

### 9.1 Query+Answer SFT 실패

먼저 teacher의 query-call step과 evidence-grounded answer step을 결합한 dataset으로 추가 LoRA를 학습했다.

| 항목 | 결과 |
| --- | ---: |
| combined SFT step | `1,578` |
| final eval loss | `0.8583` |
| final token accuracy | `0.7859` |
| DMR early attempted rows | `35` |
| DMR completed rows | `1` |
| DMR behavioral failures | `34` |
| 주된 에러 | `No tool calls found in response` |

Token-level metric은 좋아졌지만 Letta loop에서는 거의 실패했다. query target과 answer target을 섞으면 모델이 언제 tool call을 내고 언제 final answer를 내야 하는지 혼동했다.

이는 중요한 negative result다.

```text
더 많은 supervised token target이 항상 agent behavior를 개선하지 않는다.
특히 tool-call channel과 answer channel이 섞이면 loop contract가 무너질 수 있다.
```

### 9.2 Query-Only SFT와 channel failure

다음으로 `conversation_search` call만 target으로 둔 query-only LoRA를 학습했다. 의도는 answer target을 제거해 tool-call shell 안정성을 보존하는 것이었다.

학습 metric은 나쁘지 않았다.

```text
eval loss: 0.8244
token accuracy: 0.7703
```

하지만 end-to-end smoke에서는 실패했다. 모델은 search action을 생성하려는 의도는 보였지만, OpenAI tool_calls channel이 아니라 JSON-like text를 assistant content에 출력하는 경우가 많았다. 즉 tool-call channel contract를 지키지 못했다.

vLLM parser-rescue와 endpoint proxy-rescue도 시도했다.

| 조건 | 결과 | 해석 |
| --- | --- | --- |
| Query-only raw smoke | 5/5 behavioral failure | JSON-as-content channel failure |
| vLLM parser-rescue | `1/6` 완료 | parser layer만으로는 부족 |
| endpoint proxy-rescue | `2/20` 완료, containment `0.0` | channel을 구제해도 stop-and-answer policy 부재 |

Query-only adapter는 검색 행동을 계속하려는 경향이 강했고, 언제 멈추고 답해야 하는지 배우지 못했다.

### 9.3 Phase-Routed Query-Only Diagnostic

그래서 controller가 search phase와 answer phase를 강제로 분리했다.

```text
search phase:
  query-only adapter가 query를 생성
  local substring search 실행

answer phase:
  full trajectory r16 answer model이 retrieved evidence만 보고 답변
```

결과는 다음과 같았다.

| Condition | Rows | Completion | Retrieved-reference rate | Containment | Mean searches | `UNKNOWN` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase-routed query-only search + r16 answer | `100` | `100/100` | `0.09` | `0.11` | `2.22` | `67/100` |

Phase routing은 stop-and-answer failure를 제거했다. 모든 row가 완료되었다. 하지만 retrieved-reference rate가 `0.09`에 그쳤다.

즉 query-only adapter는 loop phase 문제가 제거되어도 좋은 query를 충분히 만들지 못했다.

성공 row는 대부분 probe와 answer-bearing utterance가 같은 literal phrase를 공유했다.

| Index | Query chain | Reference | Answer |
| ---: | --- | --- | --- |
| `35` | `favorite food`, `favorite food` | `Cheeseburgers!` | `My favorite food is cheeseburgers.` |
| `50` | `german shepherd`, `german shepherd` | `Barnaby!` | `Your German Shepherd's name is Barnaby.` |
| `58` | `pizza`, `pepperoni`, `pepperoni` | `Pepperoni!` | `Pepperoni.` |

실패 row는 더 흥미롭다.

```text
Probe:
  What was the artist you mentioned you could get into?

Reference:
  Taylor Swift!

Query-only search:
  artist
  artist

Result:
  No results found.

Answer:
  UNKNOWN
```

이것은 hallucination이 아니라 retrieval miss다. 모델은 질문의 semantic category인 `artist`를 골랐지만, 과거 utterance에는 `artist`라는 단어가 없었다.

### 9.4 Deterministic Query Skeleton

Phase-routed 조건에서도 모델은 여전히 tool-call JSON을 만들어야 했다. 그래서 더 강한 diagnostic을 만들었다.

```text
모델은 query string만 생성한다.
wrapper가 conversation_search shell을 고정한다.
answer phase는 기존 r16 answer model이 담당한다.
```

결과는 크게 좋아졌다.

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `100/100` | `0.09` | `0.11` | `1.10` |
| Query-only r16, skeleton | `100` | `100/100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `100/100` | `0.16` | `0.17` | `2.86` |
| Base Llama skeleton smoke | `20` | `20/20` | `0.20` | `0.20` | `1.75` |

Raw 500-row에서도 query-only skeleton은 retrieved-reference `0.244`, containment `0.216`을 기록했다.

이 결과는 query-only SFT를 재평가하게 만든다. End-to-end agent로는 실패했지만, query string만 생성하게 하면 유효한 query-policy signal이 드러난다.

## 10. Teacher Query Skeleton과 Query Gap

### 10.1 Teacher Query Skeleton Replay

Teacher가 사용한 query chain을 같은 deterministic skeleton으로 replay했다.

| Query source | Subset | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher full query chain | approved | `398` | `0.442` | `0.405` | `4.36` |
| Teacher full query chain | teacher-search only | `302` | `0.583` | `0.533` | `5.74` |
| Teacher max-3 query | approved | `398` | `0.367` | `0.342` | `3.25` |
| Teacher max-3 query | teacher-search only | `302` | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | same approved | `398` | `0.276` | `0.249` | `3.76` |
| Query-only r16 skeleton | teacher-search subset | `302` | `0.272` | `0.238` | `3.76` |

Teacher max-3 query는 query-only skeleton보다 훨씬 강했다. 특히 teacher-search subset에서 teacher는 retrieved-reference `0.483`, containment `0.450`이지만 query-only는 `0.272`, `0.238`에 머문다.

즉 query-only SFT에는 signal이 있지만 teacher 수준의 query selection에는 아직 도달하지 못했다.

### 10.2 Row-Level Gap

Teacher-search 302 subset에서 retrieved-reference 기준 row-level category는 다음과 같았다.

| Category | Retrieval rows | 의미 |
| --- | ---: | --- |
| both | `74` | teacher와 student 모두 reference evidence를 찾음 |
| teacher only | `72` | teacher만 찾고 student는 놓침 |
| student only | `8` | student만 찾음 |
| neither | `148` | 둘 다 못 찾음 |

핵심은 `teacher only`가 `72`개이고 `student only`가 `8`개뿐이라는 점이다. Student가 teacher를 대체하는 row보다 teacher가 맞히고 student가 놓치는 row가 훨씬 많다. 이 72개가 hard set이 되었다.

### 10.3 Qualitative Gap Pattern

Teacher-only 사례에는 세 가지 패턴이 반복되었다.

첫째, student는 probe phrase를 그대로 query로 복사한다.

| Row | Reference | Teacher queries | Student queries |
| ---: | --- | --- | --- |
| `33` | `California!` | `studio`, `California`, `Santa Barbara` | `audio studio`, `audio studio location`, `your audio studio` |
| `77` | `National parks!` | `national parks` | `what we do for fun`, `that place`, `like to visit` |

Substring recall에서는 긴 paraphrase query가 불리하다. 과거 utterance에 그대로 등장하지 않기 때문이다.

둘째, student는 plausible하지만 틀린 answer prior를 query로 쓴다.

| Row | Reference | Teacher queries | Student queries | Student answer |
| ---: | --- | --- | --- | --- |
| `5` | `I have a cat.` | `pets`, `cat`, `cats` | `pet`, `dogs`, `dogs` | `I have a dog.` |
| `19` | `Lawn mowing!` | `hobbies`, `outdoors`, `lawn` | `outdoors`, `grilling`, `food` | `I enjoy grilling my food outdoors.` |
| `65` | `The Navy!` | `military`, `Navy` | `military service`, `branch`, `army` | `UNKNOWN` |

셋째, teacher는 broad query 뒤 discriminative literal로 좁히지만 student는 그 마지막 literal을 놓친다.

| Row | Reference | Teacher queries | Student queries |
| ---: | --- | --- | --- |
| `10` | `Construction, like my dad.` | `job`, `work`, `construction` | `living`, `jobs`, `work` |
| `36` | `A Twitter contest!` | `win`, `contest`, `prize` | `what did you win`, `award`, `prize` |
| `81` | `Soda!` | `stopped drinking`, `drink`, `soda` | `stopped drinking`, `coffee`, `drinking` |

이 분석은 다음 실험 설계를 결정했다. 단순히 teacher query를 더 학습시키기보다, hard row에서 query candidate를 만들고 retrieval feedback으로 고르는 구조가 필요했다.

## 11. Hard Query Dataset, SFT, DPO

### 11.1 Preference Dataset

Teacher-only hard row 72개에서 query preference dataset을 만들었다.

예시는 다음과 같다.

```text
Probe:
  Hey, remember that time we talked about our pets?
  What kind of pet do you have?

Previous teacher search:
  query: pets
  output: I really love dogs. Do you have any pets?

Chosen:
  cat

Rejected:
  dogs

Reference:
  I have a cat.
```

이 예시는 병목을 잘 보여 준다. Student는 memory를 더 찾기 전에 plausible answer prior인 `dogs`를 query로 사용한다. Teacher는 broad query가 부족하다는 것을 보고 answer-bearing literal `cat`으로 좁힌다.

생성된 dataset은 다음과 같다.

| Dataset | Records | 의미 |
| --- | ---: | --- |
| hard-negative preference | `72` | teacher-only row 기반 chosen/rejected |
| zero-result-only clean subset | `51` | rejected query가 zero result인 더 깨끗한 subset |
| hard-positive SFT | `72` | chosen query만 SFT target으로 사용 |

### 11.2 Hard-Positive Query SFT

먼저 72개 positive query로 SFT를 했다.

Raw 500-row 결과:

| Query generator | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only r16 skeleton | `0.244` | `0.216` | `3.744` |
| Hard-positive r16 skeleton | `0.184` | `0.180` | `2.882` |

Hard set 내부:

| Metric on 72 hard rows | Query-only r16 | Hard-positive r16 |
| --- | ---: | ---: |
| Retrieved-reference | `0/72` | `7/72` |
| Containment | `1/72` | `7/72` |

Hard set에서는 조금 좋아졌지만 raw500 전체 성능은 떨어졌다. 작은 hard set만 SFT하면 query distribution이 좁아지고, 일반 row에서 no-result/UNKNOWN이 늘어난다.

### 11.3 Zero-Result-Only DPO

다음으로 zero-result-only 51개 clean preference pair로 DPO를 했다.

Raw 500-row 결과:

| Query generator | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only r16 skeleton | `0.244` | `0.216` | `3.744` |
| Preference zero-DPO r16 skeleton | `0.166` | `0.158` | `2.564` |

Hard set 내부:

| Metric on 72 hard rows | Query-only r16 | Preference zero-DPO r16 |
| --- | ---: | ---: |
| Retrieved-reference | `0/72` | `8/72` |
| Containment | `1/72` | `7/72` |

DPO도 같은 결론이다. Hard set 일부는 복구하지만 raw distribution을 망친다.

이 단계에서 얻은 결론은 강하다.

```text
Teacher query signal은 존재한다.
하지만 작은 hard dataset으로 weight를 더 업데이트하는 방식은 general query policy를 개선하지 못한다.
다음 방향은 학습이 아니라 search-time candidate generation과 retrieval feedback이다.
```

## 12. Candidate Query Reranking

### 12.1 방법

Candidate reranking은 search step마다 query 후보를 여러 개 만들고, 각 후보를 실제 local substring retrieval에 넣은 뒤 non-oracle score로 하나를 고른다.

기존 deterministic skeleton:

```text
probe -> one query -> local search -> answer
```

Candidate reranking:

```text
probe -> 5 candidate queries -> local search each candidate -> select one query -> answer
```

중요한 점은 reference answer를 reranking score에 쓰지 않는다는 것이다. Score는 retrieval 결과 수, query specificity, lexical overlap, broad-result penalty, repeated-evidence penalty 같은 non-oracle feature만 사용한다.

| Component | Meaning |
| --- | --- |
| result-count score | zero-result query를 벌점 주고 target result count에 가까운 query 선호 |
| specificity score | 너무 짧거나 긴 query를 벌점 |
| repeat penalty | 이미 선택한 query 반복 벌점 |
| lexical evidence overlap | retrieved message와 probe 사이의 overlap |
| query overlap | query가 probe 핵심 단어와 너무 무관하지 않도록 보정 |
| broad-result penalty | 너무 broad한 query가 많은 결과를 가져오면 벌점 |
| repeated-evidence penalty | 같은 evidence 반복 검색 벌점 |

### 12.2 100-row Ablation

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.23` | `0.23` | `3.70` |
| Candidate count rerank, temp `0.4`, target `3` | `0.19` | `0.20` | `3.13` |
| Candidate count rerank, temp `0`, target `3` | `0.24` | `0.26` | `3.03` |
| Candidate count rerank, temp `0`, target `5` | `0.24` | `0.26` | `3.96` |
| Candidate lexical rerank, temp `0`, target `3` | `0.23` | `0.26` | `3.25` |
| Candidate lexical rerank, temp `0`, target `5` | `0.25` | `0.27` | `4.52` |

100-row에서는 lexical target-5가 가장 높았다.

### 12.3 Raw500 결과

| Query strategy | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.244` | `0.216` | `3.744` |
| Hard-positive SFT skeleton | `0.184` | `0.180` | `2.882` |
| Preference zero-DPO skeleton | `0.166` | `0.158` | `2.564` |
| Candidate count rerank, target `3` | `0.210` | `0.202` | `3.218` |
| Candidate lexical rerank, target `5` | `0.246` | `0.224` | `4.742` |

Lexical target-5 reranker는 raw500에서 query-only skeleton을 아주 작게 넘었다. 차이는 작지만 의미는 크다. 이전 SFT/DPO/count-rerank 조건은 모두 raw500에서 query-only보다 낮았기 때문이다.

### 12.4 Teacher Subset 비교

Approved 398 subset:

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.367` | `0.342` | `3.249` |
| Query-only skeleton | `0.276` | `0.249` | `3.761` |
| Candidate count rerank, target `3` | `0.239` | `0.231` | `3.219` |
| Candidate lexical rerank, target `5` | `0.276` | `0.256` | `4.774` |

Teacher-search 302 subset:

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.483` | `0.450` | `4.281` |
| Query-only skeleton | `0.272` | `0.238` | `3.762` |
| Candidate count rerank, target `3` | `0.232` | `0.222` | `3.195` |
| Candidate lexical rerank, target `5` | `0.278` | `0.245` | `4.732` |

Candidate lexical rerank는 query-only보다 조금 낫지만 teacher gap을 닫지는 못했다. 즉 "teacher-level query selection 달성"은 아니다. 하지만 hard failure 복구에는 매우 강했다.

### 12.5 Hard Failure Class

Teacher-only hard 72 rows:

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `1.000` | `0.847` | `4.125` |
| Query-only skeleton | `0.000` | `0.014` | `2.028` |
| Hard-positive SFT | `0.097` | `0.097` | `1.611` |
| Preference zero-DPO | `0.111` | `0.097` | `1.500` |
| Candidate count rerank, target `3` | `0.278` | `0.236` | `2.958` |
| Candidate lexical rerank, target `5` | `0.347` | `0.306` | `4.097` |

Lexical target-5는 query-only가 하나도 retrieve하지 못한 72개 hard row에서 `25/72` reference retrieval을 복구했다. Containment도 `1/72`에서 `22/72`로 올렸다.

Zero-result-only 51 rows:

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `1.000` | `0.804` | `4.078` |
| Query-only skeleton | `0.000` | `0.020` | `1.118` |
| Hard-positive SFT | `0.098` | `0.098` | `0.647` |
| Preference zero-DPO | `0.098` | `0.078` | `0.706` |
| Candidate count rerank, target `3` | `0.314` | `0.255` | `3.000` |
| Candidate lexical rerank, target `5` | `0.353` | `0.294` | `3.882` |

이 결과는 v1 연구의 두 번째 큰 결론이다.

```text
작은 모델의 query policy는 single-shot imitation보다 search-time candidate generation과
local retrieval feedback으로 더 잘 복구된다.
```

## 13. Evidence Filtering Final Diagnostic

### 13.1 왜 마지막 실험이 필요했나

Candidate lexical rerank는 reference recall과 hard recovery를 개선했지만 mean retrieved를 `4.742`까지 늘렸다. 이는 answer model에게 더 많은 distractor evidence를 주는 문제를 만든다.

그래서 마지막 실험은 query generation을 바꾸지 않고, 이미 가져온 evidence만 answer 직전에 filtering했다.

질문은 다음이다.

```text
정답률은 유지하면서 evidence volume을 줄일 수 있는가?
```

### 13.2 Evidence Filter 방법

Evidence filter는 reference answer를 보지 않는다. 각 evidence message를 다음 feature로 score한다.

| Feature | Meaning |
| --- | --- |
| probe lexical recall | evidence가 probe 핵심 단어를 얼마나 포함하는가 |
| query overlap | evidence가 selected query들과 얼마나 겹치는가 |
| exact query hit | selected query가 evidence content에 substring으로 들어 있는가 |
| speaker bonus | Speaker 1 answer-bearing utterance 가능성을 약하게 선호 |
| length penalty | 너무 긴 message가 여러 topic을 섞는 경우 약하게 벌점 |

20-row smoke에서는 top-k가 너무 낮으면 reference evidence를 많이 잃었다.

| Condition | Source retrieved-reference | Filtered retrieved-reference | Source mean retrieved | Filtered mean retrieved | Containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical top-3, 20 rows | `0.45` | `0.30` | `6.10` | `2.45` | `0.30` |
| lexical top-5, 20 rows | `0.45` | `0.35` | `6.10` | `3.75` | `0.30` |
| first top-5, 20 rows | `0.45` | `0.35` | `6.10` | `3.75` | `0.30` |

따라서 final raw500은 top-k `6`으로 실행했다.

### 13.3 Raw500 결과

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.244` | `0.216` | `3.744` |
| Candidate lexical rerank, target `5` | `0.246` | `0.224` | `4.742` |
| Candidate lexical rerank + evidence filter top-6 | `0.234` | `0.224` | `3.426` |

Evidence filter는 retrieved-reference rate를 `0.246`에서 `0.234`로 조금 낮췄다. 하지만 final answer containment는 `0.224`로 유지했고, mean retrieved는 `4.742`에서 `3.426`으로 줄였다.

즉 성능을 더 올린 실험은 아니다. 대신 같은 containment를 더 작은 evidence budget으로 유지했다.

### 13.4 Hard Subset 결과

Hard72:

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.000` | `0.014` | `2.028` |
| Candidate lexical rerank, target `5` | `0.347` (`25/72`) | `0.306` (`22/72`) | `4.097` |
| Candidate lexical rerank + evidence filter top-6 | `0.319` (`23/72`) | `0.306` (`22/72`) | `3.014` |

Zero51:

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.000` | `0.020` | `1.118` |
| Candidate lexical rerank, target `5` | `0.353` (`18/51`) | `0.294` (`15/51`) | `3.882` |
| Candidate lexical rerank + evidence filter top-6 | `0.314` (`16/51`) | `0.294` (`15/51`) | `2.843` |

Hard set에서도 containment는 유지되고 evidence volume은 줄었다.

### 13.5 최종 해석

Evidence filter의 결론은 미묘하지만 중요하다.

```text
Lexical filter는 answer-bearing evidence를 완벽히 식별하지 못한다.
그래서 retrieved-reference rate는 조금 떨어진다.

하지만 answer containment는 유지된다.
즉 일부 distractor를 제거해도 answer model이 맞히던 row는 대부분 유지된다.
```

이 실험은 v1의 종료선으로 적합하다. 성능을 계속 튜닝하기보다, 이제 논문 메시지를 정리할 충분한 근거가 생겼다.

## 14. 전체 결과 종합

이 장에서는 앞선 실험들을 하나의 논리로 다시 묶는다. 개별 실험만 보면 어떤 결과는 긍정적이고 어떤 결과는 부정적으로 보일 수 있다. 예를 들어 LoRA r16은 strict-template baseline보다 좋아졌지만 teacher-trace replay에는 크게 못 미친다. Query-only SFT는 end-to-end agent로는 실패했지만 deterministic skeleton에서는 의미 있는 retrieval signal을 보인다. Hard-positive SFT와 DPO는 hard rows 일부를 복구하지만 raw distribution을 망친다. Candidate reranking은 hard rows를 크게 복구하지만 retrieved evidence volume을 늘린다. Evidence filtering은 성능을 올리지는 못하지만 context budget을 줄인다.

이러한 결과들은 서로 모순되지 않는다. 오히려 모두 같은 구조적 결론을 가리킨다. 작은 모델의 MemGPT 성능은 하나의 단일 능력으로 결정되지 않고, tool-call channel, query policy, retrieval feedback, evidence filtering, answer grounding이 연쇄적으로 결합된 결과다. 따라서 어느 한 단계만 학습하거나 어느 한 metric만 최적화하면 다른 단계에서 새로운 실패가 나타난다.

### 14.1 핵심 비교표

| Condition | Main metric | 해석 |
| --- | ---: | --- |
| Raw vanilla Llama | `0/5` loop completion | tool-call contract 실패 |
| Strict-template Llama | containment `0.1954` | format repair만으로는 부족 |
| Full-history Llama | containment `0.5080` | memory가 context에 있으면 훨씬 나음 |
| Frozen Llama + teacher trace | judge `0.7337` | teacher evidence replay로 크게 회복 |
| Frozen Mistral + teacher trace | judge `0.8769` | Mistral도 evidence가 있으면 강함 |
| LoRA r16 end-to-end | judge `0.4809` | distillation은 loop를 복구하지만 query gap 남음 |
| LoRA r16 + teacher-query hint | judge `0.6294` | query selection이 큰 병목 |
| LoRA r16 + teacher trace | judge `0.8668` | evidence가 있으면 LoRA answer는 강함 |
| Query-only phase-routed | containment `0.11` | channel/phase를 분리해도 query content 약함 |
| Query-only skeleton raw500 | retrieved `0.244`, containment `0.216` | query-only signal은 존재 |
| Teacher max-3 query approved | retrieved `0.367`, containment `0.342` | teacher query upper reference |
| Hard-positive SFT | raw containment `0.180` | hard set은 일부 복구, raw는 하락 |
| Zero-DPO | raw containment `0.158` | preference update도 raw는 하락 |
| Candidate lexical rerank | raw containment `0.224` | 첫 full-distribution non-oracle 개선 |
| Evidence filter top-6 | raw containment `0.224`, mean retrieved `3.426` | 정답률 유지하며 evidence budget 감소 |

### 14.2 연구 질문별 답

RQ1. 작은 오픈소스 모델은 vanilla MemGPT control loop를 실행할 수 있는가?

답은 "그대로는 어렵다"이다. Raw vanilla Llama와 Mistral은 5-row pilot에서 모두 loop completion `0`이었다.

RQ2. tool-call surface format을 고치면 retrieval quality도 회복되는가?

부분적으로만 그렇다. Strict-template adapter는 Llama loop completion을 크게 올렸지만 containment는 `0.1954`에 머물렀다. Tool-call format과 query quality는 별개의 문제다.

RQ3. teacher evidence가 주어지면 frozen student는 답할 수 있는가?

그렇다. Llama는 teacher trace replay에서 judge `0.7337`, Mistral은 `0.8769`를 기록했다. Evidence가 있으면 작은 모델도 답변 능력은 충분히 있다.

RQ4. teacher trajectory LoRA distillation은 end-to-end MemGPT behavior를 복구하는가?

부분적으로 복구한다. r16 LoRA는 loop completion `497/500`, judge `0.4809`까지 올랐다. 하지만 teacher trace replay `0.8668`에는 못 미쳤다.

RQ5. LoRA 이후 남는 병목은 무엇인가?

가장 큰 병목은 query selection이다. r16 오답의 대부분은 `searched_wrong_or_insufficient_evidence`였고, teacher-query hint는 judge를 `0.4809`에서 `0.6294`로 올렸다. Query-only skeleton과 teacher skeleton gap도 같은 결론을 지지한다.

이 다섯 연구 질문에 대한 답을 종합하면 다음과 같은 그림이 나온다. 작은 모델은 raw agent loop에서는 도구 호출 계약을 안정적으로 지키지 못한다. 도구 호출 표면을 고쳐도 query quality는 낮다. 그러나 teacher evidence가 주어지면 작은 모델은 답할 수 있다. LoRA distillation은 agent loop를 복구하지만 teacher query policy를 그대로 일반화하지 못한다. 따라서 남은 병목은 "답을 아는가"가 아니라 "어떤 memory를 찾아야 하는가"에 있다.

## 15. 논문에서 주장할 수 있는 기여점

이 연구의 contribution은 단순히 "LoRA를 학습했다"가 아니다. 더 정확히 말하면, 본 연구는 작은 오픈소스 LLM을 MemGPT-style memory agent로 만들 때 발생하는 실패를 실험적으로 분해하고, 그중 query policy가 핵심 병목임을 보인다. 또한 단순 supervised weight update보다 search-time candidate generation과 retrieval feedback이 hard failure class를 더 직접적으로 복구한다는 근거를 제시한다.

### 기여점 1. 작은 모델의 MemGPT 실패를 단계별로 분해했다

Raw vanilla, strict-template, full-history, teacher-trace replay, LoRA end-to-end, teacher-query hint, teacher-evidence replay를 순서대로 배치함으로써 실패 원인을 tool formatting, query selection, evidence use로 나누었다. 이 decomposition은 중요하다. end-to-end DMR accuracy만 보면 모델이 왜 실패했는지 알 수 없지만, 각 조건을 비교하면 실패가 어느 단계에서 발생하는지 좁혀갈 수 있다. 특히 teacher evidence replay와 teacher query hint의 차이는 query selection과 evidence interpretation을 분리하는 핵심 ablation으로 기능한다.

### 기여점 2. Teacher trajectory distillation의 효과와 한계를 보였다

LoRA distillation은 control loop를 복구했다. r16 adapter는 `497/500` loop completion을 달성했고 format failure를 크게 줄였다. 그러나 end-to-end judge accuracy는 `0.4809`에 머물렀고, 같은 adapter가 teacher evidence를 받으면 `0.8668`까지 상승했다. 이는 agent distillation에서 token-level imitation만으로는 retrieval policy가 충분히 일반화되지 않는다는 근거다.

### 기여점 3. Query-only SFT negative result를 통해 channel/phase 문제가 있음을 보였다

Query-only adapter는 학습 metric이 좋았지만 end-to-end agent로는 실패했다. 모델은 search action을 생성하려는 의도는 보였지만, OpenAI `tool_calls` channel 대신 JSON-like text를 assistant content에 출력하거나, 계속 검색만 하고 final answer로 전환하지 못했다. 이 negative result는 tool-call channel, search phase, answer phase를 분리해야 한다는 설계적 교훈을 준다.

### 기여점 4. Teacher query skeleton과 row-level gap으로 query policy 병목을 정량화했다

Teacher-search 302 subset에서 teacher-only retrieval row `72`, student-only row `8`이라는 비대칭 gap을 확인했다. 이는 teacher가 student보다 단순히 조금 더 나은 것이 아니라, student가 놓치는 특정 class를 체계적으로 잡고 있음을 의미한다. Qualitative analysis에서도 probe phrase copying, plausible but wrong answer prior, missing discriminative literal이라는 반복 패턴이 나타났다.

### 기여점 5. Candidate reranking이 hard failure class를 가장 잘 복구함을 보였다

Hard-positive SFT와 DPO는 hard set 일부만 복구하고 raw distribution을 망쳤다. 반면 candidate lexical reranking은 raw500도 소폭 개선하고 hard72에서 `25/72` retrieval, `22/72` containment를 달성했다. 이는 query policy를 weight update만으로 고치기보다, search-time에 여러 후보를 만들고 실제 retrieval feedback을 이용해 고르는 구조가 더 안정적일 수 있음을 보여 준다.

### 기여점 6. Evidence filtering으로 context budget을 줄일 수 있음을 보였다

Evidence filter top-6은 raw500 containment `0.224`를 유지하면서 mean retrieved를 `4.742`에서 `3.426`으로 줄였다. 이 결과는 성능 향상 자체보다 context efficiency 측면에서 의미가 있다. Candidate reranking이 hard rows를 복구하기 위해 더 많은 evidence를 가져오면, answer model은 distractor를 더 많이 읽어야 한다. Evidence filtering은 이 부담을 줄이면서 final answer containment를 유지할 수 있음을 보였다.

## 16. 한계

### 16.1 Local substring recall은 원 논문 전체 retrieval stack과 다르다

이 연구는 paper-era recall semantics를 맞추기 위해 case-insensitive substring search를 사용했다. 이는 원 논문의 모든 archival retrieval 조건을 완전히 재현한 것은 아니다. 따라서 결과는 "local paper-substring DMR 재현"으로 해석해야 한다.

### 16.2 GPT-4.1 teacher는 원 논문 당시 GPT-4와 다르다

원 논문은 GPT-4 계열을 사용했지만, 이 연구에서는 비용과 접근성을 고려해 GPT-4.1 snapshot을 teacher와 judge로 사용했다. 강한 teacher라는 역할은 유지되지만, 원 논문과 완전히 같은 teacher는 아니다.

### 16.3 Teacher trajectory는 gold oracle이 아니다

GPT-4.1 teacher trajectory는 강한 supervision source이지만 완전한 gold trajectory는 아니다. Teacher raw 500개 중 GPT-4.1 judge가 승인한 row는 `398`개였지만, exact containment를 통과한 row는 `206`개였다. `containment=false`, `judge=true`인 194개를 감사한 결과, `108`개는 깨끗한 search paraphrase였지만 `47`개는 no-search correct row였고, `11`개는 memory patch를 포함했으며, `28`개는 hedging 또는 lenient semantic match를 포함했다.

따라서 기존 LoRA 실험은 "gold oracle distillation"이 아니라 "judge-filtered teacher trajectory distillation"으로 해석해야 한다. 후속 query-policy 학습에서는 no-search row, memory-patch row, hedged/lenient row를 제외한 high-precision subset을 구성하는 것이 바람직하다.

### 16.4 Judge accuracy는 자동 평가다

GPT-4.1 judge는 containment보다 semantic하게 낫지만, 완벽한 human evaluation은 아니다. 대표 row 수동 검산은 필요하다.

### 16.5 Mistral은 LoRA 학습까지 확장하지 않았다

Mistral은 teacher-trace replay에서 강했지만, 이후 LoRA distillation과 query reranking 실험은 Llama 중심으로 진행했다. v1 논문에서는 Llama를 핵심 student로 두고, Mistral은 frozen teacher-evidence use comparison으로 위치시키는 것이 적절하다.

### 16.6 Evidence filter는 learned reranker가 아니다

마지막 evidence filter는 lexical heuristic이다. Reference answer를 보지 않는 non-oracle filter이지만, 학습된 reranker는 아니다. 따라서 semantic paraphrase나 indirect clue를 안정적으로 처리하지 못할 수 있다. 더 강한 embedding 기반 reranker나 학습형 reranker는 후속 연구로 남긴다.

## 17. 향후 연구

v1 실험 범위는 여기서 고정하는 것이 적절하다. 현재 결과만으로도 작은 오픈소스 LLM의 MemGPT 실패가 어디서 발생하는지, LoRA trajectory distillation이 무엇을 복구하고 무엇을 복구하지 못하는지, 그리고 search-time controller가 왜 필요한지 충분히 말할 수 있다. 다만 이 연구를 더 강한 시스템 논문이나 후속 모델 개발로 확장한다면, 다음 방향들이 자연스럽다.

| 방향 | 설명 |
| --- | --- |
| 학습형 query reranker | 여러 candidate query와 retrieved evidence를 작은 cross-encoder 또는 embedding model로 점수화해, lexical heuristic보다 안정적인 query 선택기를 학습한다. |
| 동적 검색 개수 정책 | 쉬운 row에서는 적은 수의 memory만 검색하고, 어려운 row에서는 더 많은 검색 결과를 허용하는 동적 검색 정책을 학습한다. |
| 단계적 query-chain 학습 | 처음에는 넓게 검색하고 이후 더 구체적인 query로 좁혀 가는 broad-to-specific refinement를 step-level trajectory로 학습한다. |
| Evidence 충분성 classifier | 현재 검색 결과만으로 답변이 가능한지 판단하는 classifier를 두어, 불충분할 때만 추가 검색을 수행하게 한다. |
| Answer-only grounding 학습 | retrieval은 성공했지만 answer boundary에서 틀린 row만 따로 모아 answer adapter를 학습한다. |
| Mistral LoRA 재현 실험 | Mistral의 강한 teacher-trace replay 성능이 end-to-end distillation에서도 재현되는지 확인한다. |
| Human evaluation 보강 | GPT judge의 false positive/negative를 사람이 검산해, reported accuracy와 containment metric의 신뢰도를 보강한다. |

이 방향들은 중요하지만, v1의 필수 실험은 아니다. 오히려 현재 단계에서 더 많은 실험을 추가하면 핵심 메시지가 흐려질 수 있다. v1이 방어해야 하는 중심 명제는 다음처럼 정리할 수 있다.

```text
작은 오픈소스 LLM은 MemGPT behavior의 일부를 모방할 수 있다.
그러나 안정적인 memory retrieval을 위해서는 tool-call control,
query selection, retrieval-feedback reranking,
evidence-grounded answering을 명시적으로 분리해야 한다.
```

## 18. 최종 결론

이 연구는 작은 오픈소스 LLM이 MemGPT-style DMR에서 실패하는 이유가 단일하지 않음을 보였다. 표면적으로는 모델이 긴 대화 기억에서 답을 찾지 못하는 문제처럼 보이지만, 실제로는 여러 하위 실패가 겹쳐 있다. 어떤 조건에서는 tool-call schema를 지키지 못하고, 어떤 조건에서는 search phase에서 멈추지 못하며, 어떤 조건에서는 검색은 하지만 정답 utterance를 가져오지 못한다. 또 어떤 경우에는 evidence를 가져왔지만 final answer boundary에서 잘못된 답을 내거나 search intent를 누출한다.

첫 번째 결론은 tool-call compatibility와 memory retrieval quality가 다르다는 점이다. Raw vanilla model은 structured tool-call control contract에서 먼저 실패한다. Strict parser/template repair는 loop 진입을 가능하게 하지만, retrieval quality와 answer accuracy를 충분히 회복하지 못한다. 이는 작은 모델의 MemGPT 평가에서 tool-call surface를 통과했다는 사실만으로 memory competence를 주장할 수 없음을 의미한다.

두 번째 결론은 answer-from-evidence ability와 evidence acquisition ability도 다르다는 점이다. GPT-4.1 teacher-trace replay에서 frozen Llama와 Mistral은 teacher evidence를 받으면 상당히 잘 답했다. 특히 Mistral은 vanilla Letta gate를 통과하지 못했음에도 teacher evidence replay에서는 judge accuracy `0.8769`를 기록했다. 따라서 작은 모델이 end-to-end DMR에서 실패한다고 해서 곧바로 "작은 모델은 답을 생성할 능력이 없다"고 결론내릴 수 없다. 오히려 핵심은 어떤 evidence를 찾아야 하는지, 그리고 그 evidence를 어떻게 agent loop 안에서 획득할지에 있다.

세 번째 결론은 teacher trajectory LoRA distillation의 효과와 한계다. Llama-3-8B r16 LoRA는 `497/500` loop completion과 judge accuracy `0.4809`를 기록하며 vanilla/strict-template 조건보다 안정적인 agent behavior를 보였다. 그러나 같은 LoRA에 teacher trace evidence를 주면 judge accuracy가 `0.8668`까지 오른다. 즉 LoRA는 tool-use loop와 answer-from-evidence behavior를 상당 부분 학습했지만, end-to-end에서는 query/evidence acquisition이 약해서 성능이 낮다.

네 번째 결론은 query policy가 가장 중요한 남은 병목이라는 점이다. Teacher-query hint는 r16 end-to-end judge accuracy를 `0.4809`에서 `0.6294`로 올렸다. Query-only SFT는 end-to-end agent로는 실패했지만 deterministic skeleton에서는 raw500 retrieved-reference `0.244`, containment `0.216`을 보였다. Teacher query skeleton은 approved subset에서 retrieved-reference `0.367`, containment `0.342`를 보이며 더 강했다. Row-level gap에서는 teacher-only retrieval row가 72개인 반면 student-only retrieval row는 8개뿐이었다. 이 모든 결과가 같은 방향을 가리킨다. 작은 모델은 query syntax를 일부 학습하지만, indirect probe를 answer-bearing literal query로 바꾸는 능력은 teacher 수준에 미치지 못한다.

다섯 번째 결론은 query policy를 단순 weight update만으로 고치기 어렵다는 점이다. Hard-positive SFT와 zero-DPO는 hard row 일부를 복구했지만 raw distribution을 망쳤다. 반면 candidate lexical reranking은 raw500에서 retrieved-reference `0.246`, containment `0.224`를 기록해 query-only skeleton을 소폭 넘었고, hard72에서는 retrieved-reference `25/72`, containment `22/72`까지 회복했다. 이는 hard-example imitation보다 search-time candidate generation과 retrieval-feedback selection이 더 안정적인 방향임을 보여 준다.

마지막으로 evidence filtering은 candidate reranking 이후 생기는 context budget 문제를 다룬다. Candidate lexical reranker는 hard rows를 복구하기 위해 더 많은 evidence를 가져오지만, answer model 입장에서는 distractor도 늘어난다. Top-6 evidence filter는 containment를 유지하면서 mean retrieved를 `4.742`에서 `3.426`으로 줄였다. 이 결과는 성능을 더 올리지는 못했지만, search-time recall과 answer-time context efficiency 사이의 균형을 다룰 수 있음을 보여 준다.

따라서 v1의 최종 메시지는 다음이다.

```text
작은 오픈소스 LLM의 MemGPT 복구는 단순 fine-tuning 문제가 아니다.
Teacher trajectory distillation은 loop와 evidence-based answering을 복구하지만,
query policy는 별도의 search-time candidate generation과 retrieval-feedback selection이 필요하다.
그리고 retrieval volume이 증가하면 answer 전 evidence filtering으로 context budget을 관리해야 한다.
```

이 지점에서 v1 실험은 충분히 닫혔다. 다음 작업은 새로운 실험이 아니라 논문 작성, 그림과 표 정리, 대표 성공/실패 예시 선정이다.

본 연구의 실용적 교훈은 명확하다. 작은 모델을 memory agent로 만들 때, teacher trajectory를 그대로 SFT하는 것만으로는 충분하지 않다. Agent loop의 각 단계가 서로 다른 실패 모드를 가지기 때문이다. Query generation은 query generation대로, tool-call channel은 channel대로, evidence filtering은 filtering대로, final answer는 final answer대로 평가하고 제어해야 한다. Nano-MemGPT v1은 이 분해가 실제 실험에서 필요하며, 특히 query policy가 작은 모델 MemGPT 복구의 중심 병목임을 보였다.

## Appendix A. 문서와 산출물 지도

### A.1 주요 문서

| 문서 | 역할 |
| --- | --- |
| `docs/proposal_summary.md` | 원 proposal의 연구 질문과 가설 |
| `docs/research_plan.md` | 단계별 연구 계획과 현재 상태 |
| `docs/vanilla_dmr_protocol.md` | DMR 재현 규약과 metric |
| `docs/experiment_1_report.md` | vanilla/strict-template baseline |
| `docs/oracle_experiment_report.md` | GPT-4.1 teacher와 teacher-trace replay |
| `docs/teacher_containment_mismatch_audit.md` | GPT-4.1 teacher의 containment mismatch와 judge-filtered supervision 품질 감사 |
| `docs/lora_training.md` | LoRA distillation 전체 실행 기록 |
| `docs/post_lora_evaluation_report.md` | LoRA end-to-end 평가 |
| `docs/failure_audit_report.md` | LoRA 이후 실패 유형 분석 |
| `docs/teacher_evidence_ablation_report.md` | teacher evidence ablation |
| `docs/teacher_query_ablation_report.md` | teacher query hint ablation |
| `docs/query_only_lora_report.md` | query-only SFT 진단 |
| `docs/phase_routed_dmr_report.md` | phase-routed query-only diagnostic |
| `docs/query_skeleton_dmr_report.md` | deterministic query skeleton |
| `docs/teacher_query_skeleton_report.md` | teacher query skeleton replay |
| `docs/query_skeleton_gap_report.md` | teacher-student query gap |
| `docs/query_preference_dataset_report.md` | hard query preference dataset |
| `docs/query_hard_positive_lora_report.md` | hard-positive query SFT |
| `docs/query_preference_dpo_report.md` | zero-result-only DPO |
| `docs/query_candidate_rerank_report.md` | candidate query reranking |
| `docs/evidence_filter_report.md` | final evidence filtering diagnostic |

### A.2 주요 scripts

| Script | 역할 |
| --- | --- |
| `scripts/eval_vanilla_dmr.py` | Letta/MemGPT DMR 평가 |
| `scripts/eval_dmr_oracle_replay.py` | full-history 및 teacher-trace replay |
| `scripts/train_lora_sft.py` | SFT LoRA training |
| `scripts/prepare_memgpt_datasets.py` | teacher trajectory dataset 준비 |
| `scripts/judge_dmr_answers.py` | GPT judge |
| `scripts/audit_lora_failures.py` | failure audit |
| `scripts/eval_phase_routed_dmr.py` | phase-routed query/answer 분리 평가 |
| `scripts/eval_query_skeleton_dmr.py` | deterministic query skeleton |
| `scripts/eval_teacher_query_skeleton_dmr.py` | teacher query replay |
| `scripts/export_query_preference_dataset.py` | hard query preference export |
| `scripts/eval_query_candidate_rerank_dmr.py` | candidate query reranking |
| `scripts/eval_evidence_filter_dmr.py` | final evidence filtering |

### A.3 주요 데이터와 output

| Path | 의미 |
| --- | --- |
| `data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/` | GPT-4.1 teacher raw trajectory |
| `data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl` | 승인 teacher trajectory SFT data |
| `outputs/lora_student_r8/final_adapter/` | r8 LoRA adapter |
| `outputs/lora_student_r16/final_adapter/` | r16 LoRA adapter |
| `outputs/lora_query_only_r16/final_adapter/` | query-only LoRA |
| `data/evaluation/post_lora_dmr_r16_lenient_v3/` | r16 end-to-end DMR |
| `data/evaluation/oracle_dmr_lora_teacher_trace/` | LoRA teacher trace replay |
| `data/evaluation/teacher_query_hint_r16_all/` | teacher-query hint ablation |
| `data/evaluation/query_skeleton_dmr_evidence_only500/` | query-only skeleton raw500 |
| `data/analysis/query_skeleton_gap/` | teacher-student query gap |
| `data/trajectories/query_hard_negative_preferences.jsonl` | hard query preference set |
| `data/evaluation/query_skeleton_dmr_hard_positive500/` | hard-positive SFT evaluation |
| `data/evaluation/query_skeleton_dmr_pref_zero_dpo500/` | zero-DPO evaluation |
| `data/evaluation/query_candidate_rerank_lexical500_temp0_target5/` | candidate lexical rerank final |
| `data/evaluation/evidence_filter_lexical500_k6/` | final evidence filter |

## Appendix B. 논문용 핵심 표 모음

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

### Table B4. Query Decomposition

| Condition | Rows | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Phase-routed query-only | `100` | `0.09` | `0.11` | `1.10` |
| Query-only skeleton raw500 | `500` | `0.244` | `0.216` | `3.744` |
| Teacher max-3 approved | `398` | `0.367` | `0.342` | `3.249` |
| Teacher max-3 teacher-search | `302` | `0.483` | `0.450` | `4.281` |

### Table B5. Query Learning Negative Results

| Condition | Raw500 retrieved-reference | Raw500 containment | Hard72 retrieval | Hard72 containment |
| --- | ---: | ---: | ---: | ---: |
| Query-only skeleton | `0.244` | `0.216` | `0/72` | `1/72` |
| Hard-positive SFT | `0.184` | `0.180` | `7/72` | `7/72` |
| Zero-DPO | `0.166` | `0.158` | `8/72` | `7/72` |

### Table B6. Candidate Reranking and Evidence Filtering

| Condition | Raw500 retrieved-reference | Raw500 containment | Mean retrieved | Hard72 containment |
| --- | ---: | ---: | ---: | ---: |
| Query-only skeleton | `0.244` | `0.216` | `3.744` | `1/72` |
| Candidate count rerank | `0.210` | `0.202` | `3.218` | `17/72` |
| Candidate lexical rerank | `0.246` | `0.224` | `4.742` | `22/72` |
| Evidence filter top-6 | `0.234` | `0.224` | `3.426` | `22/72` |
