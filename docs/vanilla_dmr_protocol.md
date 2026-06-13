# Vanilla MemGPT DMR 재현 규약

## 1. 목적

이 문서는 Deep Memory Retrieval(DMR) 실험을 동일한 조건으로 반복하기 위한 실행
규약이다. DMR은 작은 언어 모델이 장기 대화 기록에서 필요한 사실을 직접 검색하고,
검색 결과를 근거로 최종 답변을 구성할 수 있는지 평가한다.

제안서의 첫 질문은 단순하다.

> 학습하지 않은 7B-8B급 모델을 Vanilla MemGPT 실행 루프에 연결했을 때, 모델은
> 장기 기억 도구를 올바르게 호출하고 필요한 사실을 회수할 수 있는가?

이 질문을 답하려면 세 층을 분리해야 한다.

| 층 | 확인할 내용 | 대표 실패 |
| --- | --- | --- |
| 제어 계약 | 모델이 Letta 형식의 tool call을 만들 수 있는가 | tool-call format failure |
| 검색 행동 | 적절한 검색어로 `conversation_search`를 호출하는가 | retrieval miss |
| 답변 구성 | 회수된 근거를 최종 답변에 반영하는가 | chain 또는 final-answer failure |

## 2. 데이터 단위

평가는 `MemGPT/MSC-Self-Instruct`의 각 행을 독립적인 사례로 처리한다. 한 행에는
과거 대화와 마지막 질문이 함께 들어 있다.

| 필드 | 의미 |
| --- | --- |
| `previous_dialogs` | 검색 가능한 장기 기억으로 적재할 MSC 세션 1-5 |
| `dlg` | 세션 6의 질문과 기준 답변 |
| `self_instruct` 계열 필드 | 생성 및 평가 과정에서 사용된 부가 메타데이터 |

예를 들어 과거 세션에 다음 문장이 있고 마지막 probe가 좋아하는 음악가를 묻는다면,
모델은 즉시 문맥에 없는 이름을 recall memory에서 찾아야 한다.

```text
[과거 세션] Speaker 1: I have always liked Taylor Swift.
[세션 6 probe] Who is one of my favorite musicians?
[기준 답변] Taylor Swift.
```

## 3. 행 단위 실행 생명주기

`scripts/eval_vanilla_dmr.py`는 각 행마다 새 `memgpt_agent`를 만들고 다음 절차를
수행한다.

1. DMR persona와 MSC 초기 persona 사실을 포함한 OG `memgpt_agent`를 생성한다.
2. 과거 세션 1-5를 로컬 capture endpoint를 통해 Letta recall storage에 적재한다.
3. Letta의 `reset-messages` endpoint를 호출한다.
4. system prompt를 다시 compile하여 recall-memory count를 갱신한다.
5. 세션 6 probe만 모델에 전달한다.
6. 최종 답변, Letta message, tool trace, metric, 진단 label을 JSONL로 저장한다.

capture endpoint는 user/assistant 교환을 저장한다. MSC 원문은 Speaker 1부터 시작하거나
Speaker 2 발화로 끝나는 경우가 있어 API의 대화 교환 구조와 정확히 일치하지 않는다.
이 경우 harness는 대괄호로 감싼 session-start 또는 session-end marker를 삽입한다.
marker는 데이터셋 발화를 버리지 않고 저장 구조만 맞추기 위한 적응 계층이다.

### reset과 recompile이 필요한 이유

과거 세션을 prompt에 그대로 남기면 모델은 검색 도구 없이도 답할 수 있다. 반대로
저장 후 메시지만 지우고 prompt를 갱신하지 않으면 모델에 표시되는 recall count가
stale 상태가 된다. 현재 규약은 저장된 메시지를 검색 가능하게 유지하면서 즉시
문맥에서는 제거한다.

검증 사례에서는 36개 교환을 capture한 후 72개의 recall message가 system prompt에
정상적으로 반영되었다.

## 4. Recall 검색 계약

### paper-faithful `paper_substring`

MemGPT 논문 시기의 기본 DMR recall search는 대소문자를 구분하지 않는 문자열 검색을
사용했다. embedding search는 archival memory에 사용된다. 유지보수 중인 Letta의
`conversation_search` 설명은 text와 semantic similarity를 함께 사용하는 것처럼
읽힐 수 있지만, 현재 로컬 PostgreSQL 실행 경로는 substring matching을 수행한다.

이 불일치는 모델 행동에 영향을 준다. 모델이 의미적으로 넓은 검색어를 고르면 SQL
substring 검색에서는 아무 결과도 얻지 못할 수 있다.

```text
나쁜 예: conversation_search(query="favorite musician preference")
좋은 예: conversation_search(query="Taylor")
```

`docker-compose.yaml`은 다음 파일을 `letta-server`에 read-only로 mount한다.

```text
external/repos/letta/letta/functions/function_sets/base.py
```

또한 harness는 실행 전과 요청 직전에 `paper_substring` metadata를 적용한다. 이
변경은 모델에게 보이는 도구 설명을 교정하며 SQL 검색 구현 자체는 바꾸지 않는다.

수동 적용이 필요하면 다음 명령을 사용한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/configure_dmr_recall_contract.py --contract paper_substring
```

### 최대 step 수

harness의 기본값은 Letta 기본값과 같은 `max_steps=50`이다. 유료 teacher trajectory
수집에서는 비용을 제한하기 위해 검증된 명시적 상한 `--max-steps 16`을 사용한다.
이 값은 모델 능력의 일반적인 상한이 아니라 현재 수집 예산을 위한 실행 조건이다.

## 5. 출력과 metric

JSONL은 행마다 checkpoint되며 aggregate summary도 함께 갱신된다.

| 항목 | 의미 |
| --- | --- |
| `final_answer` | 실행 루프가 반환한 최종 답변 |
| raw Letta messages | 중간 reasoning과 tool call 기록 |
| provider traces | teacher API 요청과 응답 추적 정보 |
| `rouge_l_recall` | 기준 답변 token sequence가 결과에 얼마나 회수되었는지 보는 proxy |
| `reference_containment` | 기준 답변 문자열이 결과에 포함되는지 보는 결정적 proxy |
| `conversation_search` trace | 검색 호출 여부, query, 검색 결과 |

ROUGE-L과 containment는 빠르고 재현 가능한 자동 지표지만 의미적 정답 판정과 동일하지
않다. 예를 들어 `Home Depot`가 기준 답변인 사례에서 모델이 직장 이름은 맞히고
`manager`라는 역할을 빠뜨리면 containment는 통과할 수 있다. 따라서 핵심 결과에는
LLM judge 판정을 함께 보고한다.

## 6. 진단 label

harness는 보수적인 1차 label을 생성한다.

| label | 조건 | 해석 시 주의점 |
| --- | --- | --- |
| `retrieval_miss_candidate` | `conversation_search` 호출이 없음 | 검색이 불필요했던 예외가 있는지 확인 필요 |
| `retrieval_hallucination_candidate` | query가 비어 있거나 JSON이 잘못됨 | parser 문제와 모델 문제를 분리해야 함 |
| `chain_failure_candidate` | 검색 후 최종 답변이 없음 | step cap 또는 도구 반복 여부 확인 필요 |
| `tool_call_format_failure` | Letta가 유효한 호출을 받지 못함 | 기억 검색 이전의 제어 계약 실패 |
| `incorrect_answer` | deterministic containment proxy 실패 | LLM judge로 의미적 정답 여부 재검토 필요 |

이 label은 제안서의 GPT-4 teacher trajectory 비교나 LLM-as-a-judge를 대체하지 않는다.
오류 사례를 빠르게 분류하기 위한 진단용 보조 지표다.

## 7. 재시도와 재개

- 출력 JSONL과 summary는 매 행 저장된다.
- `--resume`을 사용하면 완료된 행을 건너뛴다.
- infrastructure error는 새 agent를 만든 뒤 재시도한다.
- infrastructure error를 모델 품질 실패로 조용히 합산하지 않는다.
- 독립적인 행은 `--workers`로 병렬 처리할 수 있다.
- 유료 API 수집은 rate limit과 비용 가시성을 위해 현재 `--workers 1`을 사용한다.

소규모 확인 예시는 다음과 같다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/eval_vanilla_dmr.py \
  --limit 1 \
  --max-steps 16 \
  --output-dir data/evaluation/dmr_smoke
```

## 8. 모델 provenance

정확한 serving 설정은 `configs/vanilla_models.yaml`에 기록한다.

| 모델 | checkpoint | precision | quantization |
| --- | --- | --- | --- |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | BF16 | 없음 |
| Llama-3-8B | `NousResearch/Meta-Llama-3-8B-Instruct` | BF16 | 없음 |

Llama는 현재 Hugging Face 계정에서 공식 gated Meta repository 접근 권한이 없기 때문에
public mirror의 revision `53346005fb0ef11d3b6a83b12c895cca40156b6c`을 고정해서
사용한다.

## 9. 해석 경계

낮은 DMR 점수는 곧바로 모델의 “기억 용량 부족”을 뜻하지 않는다. tool-call format,
검색어 선택, 회수 결과 해석, 최종 답변 구성 중 어디에서 실패했는지 분리해야 한다.
이 분리가 이후 Oracle replay와 LoRA distillation의 학습 목표를 결정한다.
