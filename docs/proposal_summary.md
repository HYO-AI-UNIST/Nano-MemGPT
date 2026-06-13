# 연구 제안서 요약

## 연구 배경

MemGPT는 LLM의 제한된 context window를 운영체제의 가상 메모리처럼 관리한다. 모델은
항상 prompt 안에 있는 main context만 읽는 것이 아니라, 필요할 때 외부 저장소를 검색하고
결과를 다시 context로 가져온다. 이때 중요한 특징은 별도의 고정 retriever가 항상 검색하는
것이 아니라, LLM 자체가 memory-management function call을 선택한다는 점이다.

예를 들어 이전 세션에서 사용자가 좋아하는 음악가를 물었다면 모델은 다음과 같은 흐름을
스스로 만들어야 한다.

```text
User probe:
  "우리가 음악 이야기했을 때 내가 좋아할 수 있다고 말한 가수가 누구였지?"

Expected behavior:
  conversation_search(query="music", request_heartbeat=true)
  conversation_search(query="Taylor Swift", request_heartbeat=true)
  send_message(message="Taylor Swift라고 말했어.")
```

이 구조는 강한 function-calling 모델에서는 효과적이지만, 작은 open-source model에서는
도구 호출 형식, 검색어 선택, heartbeat 기반 chaining이 쉽게 무너질 수 있다.

## 핵심 연구 질문

작은 open-source LLM을 MemGPT의 underlying model로 사용할 때 성능이 하락하는 이유는
무엇인가?

1. **지식 부족 가설**: 작은 모델은 retrieved evidence를 받아도 답을 만들 지식이나 추론
   능력이 부족하다.
2. **행동 패턴 부족 가설**: 작은 모델은 답을 만들 수 있지만, 적절한 시점에 올바른
   memory tool을 호출하고 chaining하는 행동을 배우지 못했다.

본 연구는 두 가능성을 Oracle experiment로 분리한다. GPT teacher가 수행한 tool call과
tool output을 student에게 직접 주입했을 때 성능이 크게 회복된다면, 주요 병목은 지식보다
memory-management behavior에 있다고 해석할 수 있다.

## 원 논문과의 연결

[MemGPT 원 논문](https://arxiv.org/abs/2310.08560)은 GPT-4 Turbo 기반 DMR에서
accuracy `93.4%`, ROUGE-L recall `0.827`을 보고한다. GPT-3.5 Turbo로 underlying
model을 교체하면 accuracy는 `66.9%`, ROUGE-L recall은 `0.629`로 하락한다.

제안서는 이 격차를 출발점으로 삼는다. 단순히 더 큰 모델을 쓰는 대신, GPT teacher의
MemGPT execution trajectory를 수집하고 Llama-3-8B 또는 Mistral-7B에 distillation하여
행동 패턴을 복구할 수 있는지 검증한다.

## 평가 태스크

### 1. MSC Deep Memory Retrieval

DMR은 Multi-Session Chat의 과거 세션에 등장한 사실을 회상하는 태스크다.

```text
Input:
  과거 대화 5개 세션 + 세션 6의 probe 질문

Output:
  과거 세션에서 검색한 사실을 이용한 자유형식 답변

Example:
  Probe: "은퇴 후 여가 시간에 무엇을 배우고 싶다고 했지?"
  Reference: "Culinary school에 가고 싶다고 했어."
```

DMR은 단순 factual QA가 아니다. 모델은 `conversation_search`의 query를 선택하고,
필요하면 여러 검색을 연쇄적으로 호출한 뒤 답해야 한다.

### 2. Document QA

원 논문의 Document-QA 조건은 NaturalQuestions-Open 50문항과 20M-passage Wikipedia
embedding index를 사용한다. archival memory 검색은 vector retrieval을 사용하며, retrieved
document 수 `K`가 커질 때 성능이 얼마나 유지되는지 본다.

현재 공개 artifact에는 원 논문의 20M index가 포함되어 있지 않다. 따라서 로컬에서는
[`document_qa_proxy.md`](document_qa_proxy.md)에 정의한 context-pack proxy를 먼저
실행하되, 이를 paper-faithful archival retrieval 결과로 보고하지 않는다.

## 실험 구조

| 단계 | 조건 | 질문 |
| --- | --- | --- |
| Stage 1-A | Small model + Raw Vanilla MemGPT | model이 기본 tool-call contract를 통과하는가? |
| Stage 1-B | Small model + Strict Template Adapter | 형식 장벽 제거 후 retrieval behavior는 얼마나 남는가? |
| Stage 1-C | Full-History Upper Bound | retrieval을 제거하면 final answer 성능이 회복되는가? |
| Stage 1-D | GPT Teacher-Trace Oracle | teacher evidence를 주입하면 student가 답을 복구하는가? |
| Stage 2-A | Full SFT | teacher trajectory distillation이 행동 실패를 줄이는가? |
| Stage 2-B | LoRA SFT | 제한된 GPU 환경에서도 비슷한 개선을 얻을 수 있는가? |

## 실패 유형

제안서의 분석 기여는 단순 accuracy 비교에 그치지 않고 failure distribution을 보고하는 데
있다.

| 유형 | 정의 | 예시 |
| --- | --- | --- |
| Type 0: Control-Contract Failure | 유효한 structured tool call을 만들지 못함 | prose로 검색 의도를 설명하지만 `tool_calls`를 내지 않음 |
| Type 1: Retrieval Miss | 검색이 필요한데 검색하지 않거나 관련 query를 끝내 시도하지 않음 | `culinary school` 문장을 찾아야 하지만 `school`을 검색하지 않음 |
| Type 2: Retrieval Hallucination | malformed query, 불필요한 검색, 잘못된 tool name을 생성함 | 존재하지 않는 `core_memory_search`를 호출함 |
| Type 3: Chain Failure | 검색 후 heartbeat chaining을 중단하거나 답변 전에 종료됨 | search 결과를 받았지만 follow-up call 또는 최종 답변을 생성하지 않음 |
| Final-Answer Failure | evidence가 있어도 정답을 구성하지 못함 | `Home Depot manager`에서 `Home Depot`만 답함 |

## 구현 선택의 근거

- **Letta**: 공개 MemGPT 코드의 유지보수 successor이므로 baseline framework로 사용한다.
- **PostgreSQL + pgvector**: persisted recall message와 vector-backed archival memory
  실험을 지원한다.
- **Paper-Substring Contract**: paper-era DMR recall은 기본적으로 case-insensitive string
  matching이다. Document-QA archival retrieval의 embedding search와 구분한다.
- **PEFT + TRL**: Full SFT와 LoRA distillation을 비교하기 위한 training stack이다.
- **vLLM**: BF16 student serving과 tool parser 실험에 사용한다.

## 현재까지의 핵심 관찰

초기 GPT-4.1 teacher pilot은 최신 Letta의 hybrid-search 설명과 로컬 substring 실행 사이의
불일치 때문에 judge accuracy `12/20` (`0.6000`)에 머물렀다. tool schema를 paper-era
substring semantics에 맞춘 뒤 corrected pilot은 `17/20` (`0.8500`)으로 개선되었다.

이 결과는 function-call interface의 작은 표현 차이도 retrieval behavior와 trajectory 품질에
큰 영향을 줄 수 있음을 보여 준다.
