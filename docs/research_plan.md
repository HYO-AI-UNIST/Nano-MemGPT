# Nano-MemGPT 연구 계획

## 연구 목표

이 프로젝트의 목표는 작은 open-source LLM에서 발생하는 MemGPT function-calling 실패를
정량적으로 진단하고, GPT teacher trajectory를 이용한 distillation으로 성능을 복구하는
것이다.

중심 질문은 다음과 같다.

> 작은 모델의 DMR 성능 저하는 지식 부족 때문인가, 아니면 memory tool을 적절히 호출하고
> chaining하는 행동 패턴 부족 때문인가?

이를 검증하기 위해 baseline, Oracle, training, post-training evaluation을 순차적으로
진행한다.

## 단계별 연구 설계

### Stage 0. 환경 및 재현 기반 구축

목적은 모든 조건을 동일한 Docker 기반에서 반복 실행할 수 있게 만드는 것이다.

| 구성요소 | 역할 | 상태 |
| --- | --- | --- |
| `nano-memgpt-dev` | Python script 실행, dataset 처리, artifact export | 완료 |
| `letta-server` | MemGPT agent loop와 memory tool 실행 | 완료 |
| `pgvector` | persisted recall message 및 archival vector storage | 완료 |
| `llama-vllm` | Llama-3-8B 또는 Mistral-7B BF16 serving | 완료 |
| `external/repos/COMMITS.md` | Letta, MemGPT, PEFT, TRL, vLLM 등 upstream source의 고정 commit 기록 | 완료 |

### Stage 1. Vanilla Behavior 진단

#### 1-A. Raw Vanilla Gate

가중치와 parser adapter를 바꾸지 않은 상태에서 small model이 structured tool call을 만들 수
있는지 본다. 실패 시 retrieval quality를 논하기 전에 control-contract failure로 분리한다.

#### 1-B. Strict Template Adapter

가중치는 고정한 채, 명시적인 단일 schema-valid tool call만 OpenAI `tool_calls` 형식으로
변환한다. 이 조건은 model intent를 추론하거나 malformed call을 복구하지 않는다.

#### 1-C. Full-History Upper Bound

과거 전체 conversation을 student context에 직접 넣는다. retrieval과 tool chaining을
제거했을 때 최종 답변 능력이 얼마나 남는지 보는 진단 상한선이다.

#### 1-D. GPT Teacher-Trace Oracle

GPT teacher가 실제로 호출한 tool과 그 output을 student에게 evidence로 주입한다. student는
tool을 직접 호출하지 않고 final answer만 생성한다. Full-History 조건보다 proposal의
behavioral hypothesis에 더 직접적인 Oracle이다.

### Stage 2. Teacher Trajectory 수집

Teacher는 fixed snapshot `gpt-4.1-2025-04-14`를 사용한다. 원 논문의
`gpt-4-1106-preview`는 2026-03-26 종료되어 정확히 재호출할 수 없다. 이 차이는 결과
보고서에 명시한다.

한 trajectory step은 다음 정보를 보존한다.

```text
TrajectoryStep:
  context: teacher에게 실제로 전달된 전체 provider request
  function_call: teacher가 선택한 tool name과 arguments
  function_output: Letta가 반환한 tool result
  request_heartbeat: 후속 chaining 여부
  final_answer: row 종료 시 teacher 답변
```

승인되지 않은 teacher row는 학습 데이터에 넣지 않는다. teacher 역시 retrieval miss나
ambiguous reference 문제를 일으킬 수 있기 때문이다.

### Stage 3. Distillation

| 조건 | 설명 | 목적 |
| --- | --- | --- |
| Full SFT | 전체 parameter를 업데이트 | 가능한 최대 회복 폭 측정 |
| LoRA `r=8` | attention layer에 low-rank adapter 적용 | 저비용 기준선 |
| LoRA `r=16` | 더 큰 adapter capacity 사용 | parameter-efficiency trade-off 측정 |

현재 GPU 환경에서는 Llama-3-8B full fine-tuning이 현실적으로 부담스럽다. 따라서 LoRA를
우선 실행하고, Full SFT는 memory budget 확인 후 별도 조건으로 다룬다.

### Stage 4. Post-Training Evaluation

학습 전후에 동일한 DMR protocol을 사용한다.

| 평가 축 | metric |
| --- | --- |
| 답변 정확도 | LLM-as-judge accuracy |
| 답변 포함 여부 | deterministic reference containment |
| 답변 recall | ROUGE-L recall |
| 검색 행동 | `conversation_search` 호출률 |
| 실패 분포 | Type 0/1/2/3 및 final-answer failure |
| 효율성 | trainable parameter 수, GPU 시간, API 비용 |

## 현재 진행 상황

| 단계 | 상태 | 주요 결과 |
| --- | --- | --- |
| Docker 및 upstream clone | 완료 | 재현 환경 구축 |
| Raw Vanilla compatibility pilot | 완료 | Llama, Mistral 모두 `0/5` structured loop |
| Strict-template Llama scaled DMR | 완료 | `481/500` loop 완료, containment `0.1954` |
| Full-History Upper Bound | 완료 | Llama `0.5080`, Mistral `0.4240` containment |
| GPT-4.1 corrected 20-row pilot | 완료 | judge accuracy `17/20` (`0.8500`) |
| Pilot Teacher-Trace Oracle | 완료 | Llama `14/17`, Mistral `15/17` judge accuracy |
| GPT-4.1 scaled trajectory 수집 | 완료 | `500/500`, judge accuracy `0.7960` |
| 승인 trajectory export | 완료 | 승인 행 `398`, context-complete SFT step `1,664` |
| Scaled Oracle replay | 완료 | Llama `292/398`, Mistral `349/398` judge 통과 |
| LoRA `r=8` 학습 | 완료 | final eval loss `0.9009`, mean token accuracy `0.7546` |
| LoRA `r=16` 학습 | 완료 | final eval loss `0.8694`, mean token accuracy `0.7599` |
| Post-training DMR 평가 | 완료 | r8 judge `0.4765`, r16 judge `0.4809`; r16이 loop/search 안정성 우세 |
| Post-LoRA failure audit | 완료 | 남은 오답의 다수는 `searched_wrong_or_insufficient_evidence`; query selection 병목 가설 강화 |
| LoRA Teacher-evidence ablation | 완료 | teacher trace evidence 제공 시 r8 `343/398`, r16 `345/398` judge 통과 |
| LoRA Teacher-query hint ablation | 완료 | r16 search rate `0.9594`, judge `248/394` (`0.6294`) |
| Query+answer SFT r16 | 중단 | proxy metric은 개선됐지만 early DMR `35`행 중 `34` tool-call format failure |
| Query-only SFT r16 | 중단 | eval loss `0.8244`; 초반 valid tool call은 생성하지만 5-row smoke 모두 JSON-as-content channel failure |
| vLLM parser-rescue smoke | 중단 | `nano_rescue_llama` 추가 후 6-row smoke `1` 완료, `5` format failure; vLLM-level rescue만으로는 불충분 |
| Endpoint proxy-rescue smoke | 완료 | proxy가 tool call rescue `70`회 수행했지만 20-row smoke `2/20` 완료, containment `0.0`; stop-and-answer policy 부재 확인 |
| Phase-routed query-only diagnostic | 완료 | search/answer phase 분리 시 `100/100` 완료. 그러나 evidence-only containment `0.11`, retrieved-reference rate `0.09`로 query content 병목 확인 |
| Deterministic query skeleton | 완료 | query-only r16이 100-row에서 containment `0.23`, retrieved-reference `0.23`; full r16 skeleton `0.17/0.16`, base 20-row `0.20/0.20` |
| Teacher query skeleton replay | 완료 | approved 398에서 teacher max-3 query containment/retrieved-reference `0.342/0.367`; teacher-search subset `0.450/0.483`; query-only same subset `0.238/0.272` |
| Query skeleton row-level gap audit | 완료 | teacher-search 302 subset에서 retrieval 기준 teacher-only `72`, student-only `8`, both `74`, neither `148`; hard-negative query set 확보 |
| Query preference dataset export | 완료 | teacher-only row에서 preference `72`개, zero-result-only clean preference `51`개, positive SFT `72`개 생성; SFT prepare check 통과 |
| Hard-positive query SFT pilot | 완료 | raw500 retrieved/containment `0.184/0.180`으로 query-only `0.244/0.216`보다 하락; hard set 72개 내부 retrieval은 `0`에서 `7`로 개선 |
| Zero-result-only DPO pilot | 완료 | raw500 retrieved/containment `0.166/0.158`로 하락; hard set 72개 내부 retrieval은 `8/72`까지 개선 |
| Candidate lexical reranking | 완료 | raw500 `0.246/0.224`, hard72 `25/72` retrieval 및 `22/72` containment; 첫 full-distribution non-oracle 개선 |
| Evidence filtering final diagnostic | 완료 | top-6 filter가 raw500 containment `0.224` 유지, mean retrieved `4.742 -> 3.426`; v1 실험 freeze 기준 확보 |

## 예상 데이터 규모와 비용

corrected pilot은 raw row 20개에서 승인 SFT step 69개를 생성했다.

| 항목 | 관측값 |
| --- | ---: |
| raw row당 평균 API 비용 | `$0.0419` |
| raw row당 승인 SFT step | `3.45` |
| raw row 500개 예상 비용 | 약 `$20.93` |
| raw row 500개 예상 승인 SFT step | 약 `1,725` |
| raw row 500개 실제 승인 SFT step | `1,664` |
| 승인 SFT step 2,000개에 필요한 raw row | 약 `580` |

이 값은 pilot 분포를 단순 외삽한 추정치다. 긴 search chain, retry, judge 호출 비용에 따라
달라질 수 있다.

## 디렉터리 구조

| 경로 | 역할 |
| --- | --- |
| `configs/` | model, experiment, upstream repository 설정 |
| `data/raw/` | 원본 dataset |
| `data/evaluation/` | row-level 평가 JSONL과 summary |
| `data/trajectories/` | approved teacher rows, Oracle trace, SFT step |
| `external/repos/COMMITS.md` | upstream source commit snapshot; 실제 clone은 `scripts/bootstrap_external_repos.py`로 복원 |
| `logs/` | 장기 실행 로그 |
| `outputs/` | 향후 checkpoint와 adapter |
| `scripts/` | setup, evaluation, judge, export CLI |
| `src/nanomemgpt/` | local schema, metric, formatting logic |

## 다음 실행 순서

1. Query-only adapter를 end-to-end agent로 쓰는 경로는 중단한다. Proxy rescue로 channel을
   상당 부분 구제해도 완료 `2/20`, containment `0.0`이므로 stop-and-answer policy가 없다.
2. Phase-routed diagnostic은 완료했다. Controller가 search/answer phase를 나누면 loop는
   `100/100` 완료되지만 retrieved-reference rate가 `0.09`에 그친다. 따라서 query-only
   adapter는 현재 성능 개선 모듈이 아니라 query-quality diagnostic으로 해석한다.
3. Deterministic query skeleton은 완료했다. Query-only r16은 tool-call phase-routed 조건보다
   retrieved-reference rate를 `0.09`에서 `0.23`으로 올렸고, full r16 skeleton보다도 높았다.
   따라서 query-only SFT에는 유효한 query-policy signal이 들어 있지만 interface를 고정해야 한다.
4. Teacher query skeleton replay는 완료했다. Search budget을 max `3`으로 맞춰도 teacher query가
   query-only보다 높다. 특히 teacher-search subset에서 teacher max-3 retrieved-reference
   `0.483`, query-only `0.272`로 약 2배 차이가 난다.
5. Query skeleton row-level gap audit은 완료했다. Teacher-search subset에서 teacher-only
   retrieval row가 `72`개, student-only row가 `8`개였으므로, 다음 학습은 이 `teacher_only`
   hard set을 중심으로 retrieval-supervised query objective를 설계한다.
6. Query preference dataset export는 완료했다. Default preference set은 `72`개, 더 깨끗한
   zero-result-only set은 `51`개다. 기존 SFT trainer용 positive-query dataset도 `72`개 생성했고
   `--prepare-only`에서 overlength drop 없이 통과했다.
7. Hard-positive query SFT pilot은 완료했다. Hard set 내부에서는 `7/72` retrieval gain이 있었지만
   raw500 전체 성능은 하락했다. 따라서 positive-only SFT를 더 오래 돌리는 방향은 우선순위가 낮다.
8. Zero-result-only preference DPO pilot은 완료했다. Hard72 내부에서는 query-only가 `0/72`
   맞히던 reference retrieval을 `8/72`까지 회복했지만, raw500 전체 retrieved-reference는
   `0.166`으로 query-only `0.244`보다 낮았다. 따라서 naive positive SFT와 small-data DPO 모두
   hard examples를 부분적으로 복구하지만 general query policy를 개선하지 못한다.
9. Candidate query generation + local retrieval reranking은 완료했다. 단순 result-count reranker는
   hard72를 강하게 복구했지만 raw500 전체에서는 query-only skeleton보다 낮았다. 이후 lexical
   evidence overlap과 broad-query penalty를 넣은 target-5 reranker는 raw500에서 `0.246/0.224`를
   기록해 query-only skeleton `0.244/0.216`을 소폭 넘었고, hard72에서는 query-only가 `0/72`
   retrieve하던 reference를 `25/72`까지 복구했으며 containment도 `1/72`에서 `22/72`로 올렸다.
   따라서 이 방향은 teacher-only failure class를 가장 직접적으로 겨냥하면서 raw distribution도
   무너뜨리지 않는 첫 non-oracle 구조다.
10. Evidence filtering final diagnostic도 완료했다. Lexical target-5의 mean retrieved `4.742`를
   top-6 filter로 `3.426`까지 줄였고, raw500 containment는 `0.224`로 유지했다. Hard72에서도
   containment `22/72`, zero51에서도 containment `15/51`을 유지했다. 다만 filtered retrieved-reference는
   raw500 `0.246 -> 0.234`로 줄었으므로, 단순 lexical filter가 answer-bearing evidence를 완벽하게
   식별하지는 못한다.
11. v1 실험은 여기서 freeze한다. 다음 작업은 새 실험이 아니라 paper writing, figure/table 정리,
   representative success/failure example 선정이다.
