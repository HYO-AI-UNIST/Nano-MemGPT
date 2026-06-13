# Oracle 실험 중간 보고서

## 1. 요약

Oracle 실험은 작은 모델의 DMR 실패를 검색 행동과 최종 답변 구성으로 분해하기 위해
진행한다. 강한 GPT-4.1 teacher가 같은 MemGPT loop에서 만든 검색 trajectory를
기록하고, judge가 승인한 evidence만 frozen Llama-3-8B와 Mistral-7B에 replay한다.

현재까지 얻은 가장 중요한 관찰은 다음과 같다.

1. Llama strict-template baseline은 500행에서 containment `0.1954`였다.
2. 전체 history를 직접 주입하면 Llama containment는 `0.5080`으로 회복된다.
3. Mistral은 Vanilla Letta tool-call gate를 통과하지 못하지만 teacher trace를 주입하면
   pilot judge accuracy `15/17`(`0.8824`)에 도달한다.
4. GPT-4.1도 완벽하지 않으므로 judge 승인 trajectory만 Oracle과 LoRA 데이터로 사용해야
   한다.

이 결과는 작은 모델의 핵심 병목 중 상당 부분이 단순 지식 부족이 아니라
tool-call 제어, 검색어 선택, chaining에 있음을 지지한다.

## 2. 원 논문 대비 teacher 모델 차이

[원본 MemGPT 논문](https://arxiv.org/abs/2310.08560)은 GPT-4 Turbo
`gpt-4-1106-preview`를 사용했다. OpenAI
[deprecation log](https://platform.openai.com/docs/deprecations)에 따르면 해당 endpoint는
2026-03-26 종료되었다. 따라서 이 프로젝트는 Letta가
`openai/gpt-4.1-2025-04-14`로 노출하는 고정 GPT-4.1 snapshot을 사용한다.

이 선택은 “강한 teacher가 정상 trajectory를 제공할 때 student가 얼마나 회복하는가”라는
연구 질문에는 적합하다. 다만 원 논문 GPT-4 Turbo 수치와 완전히 같은 teacher 조건을
재현했다고 주장하지 않는다.

## 3. Full-history upper bound

유료 teacher 수집 전에 no-key 보조 Oracle을 완료했다. 각 frozen student에 MSC 세션
1-5 전체를 직접 주입하고 tool call 없이 답변을 요청했다.

| 모델 | 평가 행 | 완료 | infra error | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 500 | 500 | 0 | 0.7850 | 0.5080 |
| Mistral-7B | 500 | 500 | 0 | 0.6711 | 0.4240 |

산출물:

- `data/evaluation/oracle_dmr_full_history/nousresearch-meta-llama-3-8b-instruct-full_history-offset-0-limit-500.jsonl`
- `data/evaluation/oracle_dmr_full_history/mistralai-mistral-7b-instruct-v0-3-full_history-offset-0-limit-500.jsonl`

### Experiment 1과 비교

| 조건 | 모델 | containment | 해석 |
| --- | --- | ---: | --- |
| strict-template DMR | Llama-3-8B | 0.1954 | 모델이 직접 검색과 chaining 수행 |
| full-history upper bound | Llama-3-8B | 0.5080 | 검색을 제거하고 전체 과거 문맥 제공 |

Llama의 큰 회복 폭은 retrieval selection과 tool chaining이 주요 병목임을 시사한다.
full-history에서도 오류가 남으므로 final-answer construction 역시 완벽하지 않다.

Mistral은 strict-template tool-call gate를 통과하지 못했지만 full-history 500행은 모두
완료했다. Mistral의 Vanilla 실패는 retrieval quality를 측정하기도 전의 제어 계약
장벽으로 분리된다.

## 4. GPT trace Oracle 파이프라인

다음 파이프라인은 구현과 smoke test를 마쳤다.

| 단계 | script | 출력 |
| --- | --- | --- |
| teacher raw 수집 | `scripts/eval_vanilla_dmr.py` | provider trace가 포함된 raw JSONL |
| 의미 정답 판정 | `scripts/judge_dmr_answers.py` | `.judged.jsonl` |
| 승인 행 필터링 | `scripts/filter_teacher_trajectories.py` | `approved_rows.jsonl` |
| Oracle trace 추출 | `scripts/extract_teacher_oracle_traces.py` | replay용 `approved_oracle.jsonl` |
| 학습 데이터 export | `scripts/export_teacher_sft_trajectories.py` | LoRA용 `approved_sft.jsonl` |
| student replay | `scripts/eval_dmr_oracle_replay.py` | student answer-only 평가 JSONL |

기존 Llama strict-template JSONL을 parser fixture로 사용했을 때 최신 500행 중 완료된
trajectory 481개를 읽고, 그중 407개에서 `conversation_search`를 추출했다. 5행 fixture
replay는 infra error 없이 끝났다. 이 결과는 plumbing 검증이지 GPT teacher 결과가
아니다.

context-complete 학습 export도 로컬 fixture로 검증했다. DMR 한 행에서 provider
request/response trace 6개를 보존하고 `TrajectoryStep` 5개를 export했다. 따라서 한 번의
유료 teacher pass를 Oracle replay와 이후 SFT/LoRA supervision에 함께 사용할 수 있다.

## 5. 초기 GPT-4.1 pilot

첫 paced GPT-4.1 pilot은 infra error 없이 20/20행을 완료했다. GPT-4.1 judge는
12행(`0.6000`)을 승인했다. 승인 행은 Oracle replay 12행과 context-complete SFT step
32개로 export되었다.

| 항목 | 값 |
| --- | ---: |
| raw 행 | 20 |
| judge 승인 | 12 |
| provider-trace 비용 | `$0.6696` |
| raw 행당 평균 | `$0.0335` |
| raw 500행 단순 추정 | `$16.74` |

이 수치는 paper-comparable teacher ceiling으로 보고하지 않는다. trace 감사에서 두
가지 로컬 규약 문제를 발견했다.

1. maintained Letta는 `conversation_search`를 hybrid semantic search처럼 설명했지만,
   로컬 PostgreSQL 경로는 대소문자를 무시하는 substring matching을 실행했다.
2. harness는 agent loop를 `max_steps=8`로 제한하고 있었다. 이는 Letta 기본
   `DEFAULT_MAX_STEPS=50`보다 낮아 검색이 많은 행이 최종 답변 전에 멈출 수 있다.

## 6. Recall 계약 감사

논문 시기 MemGPT `0.1.6` 기본 DMR recall memory는 대소문자를 무시하는 string
matching을 사용하고, embedding retrieval은 archival memory 경로에 속한다. Docker
설정은 현재 narrow docstring override를 mount하며 harness 기본 `max_steps`는 `50`으로
교정했다.

Taylor Swift 행은 통제된 확인 사례다.

| 조건 | 결과 | provider call | trace 비용 |
| --- | --- | ---: | ---: |
| maintained hybrid 설명, `max_steps=16` | 잘못된 추측 | 9 | `$0.0909` |
| mounted paper-substring 설명, `max_steps=16` | 정답 `Taylor Swift` | 8 | `$0.0819` |

도구 설명을 실제 실행 경로에 맞추는 것만으로 한 사례의 검색 행동이 교정되었다. 그러나
이는 teacher가 항상 정답이라는 뜻은 아니다.

## 7. 교정된 GPT-4.1 pilot

교정 pilot은 mounted paper-substring 계약과 명시적 `max_steps=16` 비용 상한을 사용했다.
infra error 없이 20/20행을 완료했다.

| metric | 초기 pilot | 교정 pilot |
| --- | ---: | ---: |
| GPT-4.1 judge accuracy | `12/20` (`0.6000`) | `17/20` (`0.8500`) |
| ROUGE-L recall | `0.6018` | `0.7208` |
| containment | `0.3500` | `0.4500` |
| provider-trace 비용 | `$0.6696` | `$0.8372` |
| 승인 Oracle 행 | `12` | `17` |
| 승인 SFT step | `32` | `69` |

교정 pilot은 raw 행당 평균 `$0.0419`, raw 행당 승인 SFT step `3.45`개다. 같은 분포를
가정하면 raw 500행은 약 `$20.93`, 승인 SFT step은 약 1,725개다.

### judge가 거절한 예시

| 행 | 관찰 | 거절 이유 |
| ---: | --- | --- |
| 1 | 이후 세션에서 반복된 `couponing`을 답함 | 기준 답변은 fresh/raw diet이며 대화가 다소 모호함 |
| 11 | 여러 literal query를 시도했지만 `Jim Shockey`를 찾지 못함 | 필요한 사실 회수 실패 |
| 14 | `Home Depot`를 답함 | 직장명은 맞지만 기준 답변의 manager 역할 누락 |

따라서 teacher raw trajectory를 전부 학습에 사용하지 않고 judge 승인 행만 export한다.

산출물:

- `data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_pilot/openai-gpt-4-1-2025-04-14-offset-0-limit-20.jsonl`
- `data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_pilot/openai-gpt-4-1-2025-04-14-offset-0-limit-20.judged.jsonl`
- `data/trajectories/gpt41_paper_substring_pilot_approved_rows.jsonl`
- `data/trajectories/gpt41_paper_substring_pilot_approved_oracle.jsonl`
- `data/trajectories/gpt41_paper_substring_pilot_approved_sft.jsonl`

## 8. Teacher-trace Oracle pilot

judge가 승인한 17개 teacher trace를 두 frozen student에 replay했다. student retrieval과
tool chaining은 제거하지만 teacher가 실제로 찾은 tool output은 유지한다.

| 모델 | 승인 trace | 완료 | judge accuracy | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 17 | 17 | `14/17` (`0.8235`) | 0.6590 | 0.5294 |
| Mistral-7B | 17 | 17 | `15/17` (`0.8824`) | 0.7372 | 0.4706 |

### 잔여 오류 예시

| 모델 | 행 | 관찰 |
| --- | ---: | --- |
| Llama | 6 | 기준 `Clueless` 대신 `Friday`를 답함 |
| Llama, Mistral | 10 | construction 관련 답변이 불완전함 |
| Llama, Mistral | 18 | attic 관련 기준 사실을 최종 답변에 반영하지 못함 |

Mistral이 Vanilla Letta gate를 통과하지 못했음에도 teacher evidence 주입 후 대부분의
질문에 답한다는 점은 behavioral-bottleneck 가설을 강하게 지지한다. Llama 역시 크게
회복하지만 evidence가 있어도 answer construction 오류가 일부 남는다.

## 9. Scaled collection과 Teacher-Trace Oracle 결과

500행 수집은 2026-06-01 21:44 KST에 시작했다.

| 설정 | 값 |
| --- | --- |
| teacher | `gpt-4.1-2025-04-14` |
| recall 계약 | mounted `paper_substring` |
| worker | 1 |
| 행간 cooldown | 65초 |
| step cap | `max_steps=16` |
| checkpoint | 매 행 JSONL과 summary 저장 |

scaled teacher 출력:

- `data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.jsonl`
- `logs/gpt41_paper_substring_scaled.log`

raw teacher 500행은 모두 infra error 없이 완료되었다.

| metric | 값 |
| --- | ---: |
| raw teacher 행 | `500` |
| 완료 | `500` |
| infra error | `0` |
| GPT-4.1 judge accuracy | `398/500` (`0.7960`) |
| ROUGE-L recall | `0.7393` |
| containment | `0.4120` |
| search rate | `0.7740` |

judge 승인 행만 학습 및 replay 데이터로 export했다.

| export | 규모 |
| --- | ---: |
| 승인 raw teacher 행 | `398` |
| 승인 Oracle trace | `398` |
| `conversation_search`를 포함한 승인 trace | `302` |
| 승인 행당 평균 `conversation_search` 호출 | `2.9648` |
| context-complete SFT step | `1,664` |

승인된 398개 teacher evidence를 두 frozen student에 replay하고 같은 GPT-4.1 judge로
판정했다.

| 모델 | 완료 | infra error | judge accuracy | ROUGE-L recall | containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | `398/398` | `0` | `292/398` (`0.7337`) | `0.6597` | `0.4246` |
| Mistral-7B | `398/398` | `0` | `349/398` (`0.8769`) | `0.7208` | `0.4598` |

scaled 결과에서도 pilot의 방향성이 유지되었다. 특히 Mistral은 Vanilla Letta
tool-call gate를 통과하지 못했지만 teacher evidence가 주어지면 `87.69%`를 맞힌다.
이는 Mistral의 주된 병목이 지식 자체보다 제어 계약과 검색 행동에 있음을 강하게
시사한다. Llama도 evidence 주입으로 정답을 상당수 복구하지만 answer construction
오류가 더 많이 남는다.

scaled 산출물:

- `data/trajectories/gpt41_paper_substring_scaled_approved_rows.jsonl`
- `data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl`
- `data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl`
- `data/evaluation/oracle_dmr_teacher_trace_gpt41_paper_substring_scaled/nousresearch-meta-llama-3-8b-instruct-teacher_trace-offset-0-limit-500.judged.jsonl`
- `data/evaluation/oracle_dmr_teacher_trace_gpt41_paper_substring_scaled/mistralai-mistral-7b-instruct-v0-3-teacher_trace-offset-0-limit-500.judged.jsonl`

## 10. 해석의 한계

- scaled Teacher-Trace Oracle은 judge 승인 teacher trace 398개를 대상으로 한다. 원본
  DMR 500행 전체가 아니라 teacher가 맞힌 부분집합에 대한 answer-only 조건이다.
- GPT-4.1 trajectory는 stochastic하며 같은 행에서도 검색 경로가 달라질 수 있다.
- `max_steps=16`은 비용 제한 조건이다. 별도 ceiling sensitivity run이 필요하다.
- LLM judge는 의미 판정에 유용하지만 절대적인 ground truth는 아니다. 거절 사례를
  사람이 표본 감사해야 한다.
- 현재 핵심 Oracle은 DMR에 관한 결과다. Document-QA는 paper-faithful Wikipedia
  index를 복구하기 전까지 proxy로만 보고한다.
