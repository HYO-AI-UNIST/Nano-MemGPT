# Evidence Filtering Final Report

## 1. 목적

이 문서는 Nano-MemGPT v1 연구의 마지막 실험을 정리한다. 직전 lexical candidate reranker는
raw500 전체에서 query-only skeleton을 처음으로 소폭 넘겼고, hard failure class를 크게 복구했다.
하지만 동시에 mean retrieved가 `4.742`까지 증가했다.

마지막 질문은 다음과 같다.

```text
Lexical candidate reranker가 가져온 evidence를 answer 직전에 non-oracle filtering하면,
정답률은 유지하면서 distractor evidence volume을 줄일 수 있는가?
```

이 실험은 query generation을 다시 학습하거나 바꾸지 않는다. 이미 완료된 lexical target-5 candidate
rerank 결과를 입력으로 사용하고, retrieved evidence만 top-k로 줄인 뒤 같은 answer model에게 다시
답하게 한다. 따라서 query selection과 evidence filtering을 분리해서 보는 마지막 diagnostic이다.

## 2. 방법

Script:

```text
scripts/eval_evidence_filter_dmr.py
```

Input:

```text
data/evaluation/query_candidate_rerank_lexical500_temp0_target5/
```

각 row는 이미 다음 정보를 가지고 있다.

```text
DMR probe
selected search query chain
retrieved evidence list
answer generated with all retrieved evidence
reference answer
```

Evidence filter는 reference answer를 보지 않는다. 각 evidence message를 아래 non-oracle feature로
score한다.

| Feature | Meaning |
| --- | --- |
| probe lexical recall | evidence가 probe의 핵심 단어를 얼마나 포함하는가 |
| query overlap | evidence가 selected query들과 얼마나 겹치는가 |
| exact query hit | selected query가 evidence content에 실제 substring으로 들어 있는가 |
| speaker bonus | Speaker 1 answer-bearing utterance 가능성을 약하게 선호 |
| length penalty | 너무 긴 message가 여러 topic을 섞는 경우 약하게 벌점 |

20-row smoke에서 top-k가 너무 낮으면 reference evidence를 많이 잃었다.

| Condition | Source retrieved-reference | Filtered retrieved-reference | Source mean retrieved | Filtered mean retrieved | Containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| lexical top-3, 20 rows | `0.45` | `0.30` | `6.10` | `2.45` | `0.30` |
| lexical top-5, 20 rows | `0.45` | `0.35` | `6.10` | `3.75` | `0.30` |
| first top-5, 20 rows | `0.45` | `0.35` | `6.10` | `3.75` | `0.30` |

따라서 final raw500은 top-k `6`으로 실행했다. 이 설정은 offline 보존율 기준으로 mean retrieved를
줄이면서 reference evidence 손실을 비교적 작게 유지하는 절충점이었다.

## 3. 실행 조건

```bash
docker compose exec -T nano-memgpt-dev python scripts/eval_evidence_filter_dmr.py \
  --input-jsonl data/evaluation/query_candidate_rerank_lexical500_temp0_target5/nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-candidate-rerank-lexical-c5-target-5-offset-0-limit-500.jsonl \
  --output-dir data/evaluation/evidence_filter_lexical500_k6 \
  --offset 0 \
  --limit 500 \
  --top-k 6 \
  --filter-mode lexical \
  --answer-model nano-memgpt-llama3-r16
```

## 4. Raw500 Result

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.244` | `0.216` | `3.744` |
| Candidate lexical rerank, target `5` | `0.246` | `0.224` | `4.742` |
| Candidate lexical rerank + evidence filter top-6 | `0.234` | `0.224` | `3.426` |

Evidence filter는 retrieved-reference rate를 `0.246`에서 `0.234`로 약간 낮췄다. 하지만 final answer
containment는 `0.224`로 유지했고, mean retrieved는 `4.742`에서 `3.426`으로 줄였다.

즉 이 실험은 accuracy improvement가 아니라 context-efficiency improvement다. 정답률을 더 올리지는
못했지만, 같은 containment를 더 작은 evidence budget으로 유지했다.

## 5. Hard Subset Result

### 5.1 Teacher-only hard 72 rows

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.000` | `0.014` | `2.028` |
| Candidate lexical rerank, target `5` | `0.347` (`25/72`) | `0.306` (`22/72`) | `4.097` |
| Candidate lexical rerank + evidence filter top-6 | `0.319` (`23/72`) | `0.306` (`22/72`) | `3.014` |

Hard72에서도 같은 패턴이다. Filter는 reference-containing evidence를 `25/72`에서 `23/72`로 조금 잃지만,
answer containment는 `22/72`로 유지한다. Mean retrieved는 `4.097`에서 `3.014`로 줄었다.

### 5.2 Zero-result-only 51 rows

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.000` | `0.020` | `1.118` |
| Candidate lexical rerank, target `5` | `0.353` (`18/51`) | `0.294` (`15/51`) | `3.882` |
| Candidate lexical rerank + evidence filter top-6 | `0.314` (`16/51`) | `0.294` (`15/51`) | `2.843` |

Zero51에서도 filter는 retrieval hit를 `18/51`에서 `16/51`로 낮추지만 containment `15/51`은 유지한다.
Mean retrieved는 `3.882`에서 `2.843`으로 줄었다.

## 6. 해석

이 마지막 실험은 "evidence filtering으로 성능을 더 올릴 수 있다"는 결과는 아니다. 오히려 더 정확한
결론은 다음과 같다.

```text
현재 small-model MemGPT 병목은 단순 distractor overload만이 아니다.
Reference evidence를 조금 줄여도 answer containment가 유지되는 row가 있지만,
filter가 answer-bearing evidence를 완벽하게 식별하지는 못한다.
```

그래도 중요한 성과가 있다. Lexical target-5 reranker가 만든 높은 evidence volume을 top-6 filter로
줄여도 final containment는 유지된다. 즉 query-time reranking으로 hard rows를 복구하고, answer-time
filtering으로 context budget을 줄이는 구조는 가능하다.

이제 v1 연구의 종료선은 충분하다.

1. Vanilla/strict-template은 tool-call surface 병목을 보였다.
2. Teacher-trace oracle은 작은 모델이 evidence를 받으면 답할 수 있음을 보였다.
3. LoRA distillation은 control loop를 회복했지만 query selection을 충분히 회복하지 못했다.
4. Query-only SFT에는 signal이 있었지만 agent loop에서는 phase/channel 문제가 섞였다.
5. Deterministic query skeleton은 query signal을 드러냈다.
6. Hard-positive SFT와 DPO는 hard rows 일부만 복구하고 raw distribution을 망쳤다.
7. Candidate lexical reranking은 hard rows를 크게 복구하고 raw distribution을 소폭 개선했다.
8. Evidence filtering은 정답률을 유지하면서 evidence budget을 줄였다.

따라서 더 이상의 v1 실험은 성능 튜닝으로 흐를 가능성이 크다. 여기서 실험을 freeze하고 paper writing으로
넘어가는 것이 좋다.

## 7. Artifacts

```text
scripts/eval_evidence_filter_dmr.py
data/evaluation/evidence_filter_lexical20_k3_v2/
data/evaluation/evidence_filter_lexical20_k5/
data/evaluation/evidence_filter_first20_k5/
data/evaluation/evidence_filter_lexical100_k6/
data/evaluation/evidence_filter_lexical500_k6/
```
