# Query Candidate Reranking Report

## 1. 목적

이 문서는 query-only LoRA가 만든 검색어를 그대로 쓰지 않고, 여러 candidate query를 생성한 뒤 local
retrieval 결과를 보고 하나를 선택하는 diagnostic 실험을 정리한다.

직전 실험들의 결론은 분명했다.

1. Hard-positive SFT는 hard rows 일부를 복구하지만 raw500 전체 성능은 하락했다.
2. Zero-result-only DPO도 hard rows 일부를 복구하지만 raw500 전체 성능은 하락했다.
3. 따라서 현재 병목은 "teacher query를 더 외우게 하는 것"보다 "query 후보를 만들고 실제 검색 결과를
   보고 고르는 것"에 가깝다.

이번 실험의 질문은 다음과 같다.

```text
Query-only r16 model이 여러 literal substring query 후보를 만들고,
local conversation_search 결과를 기준으로 non-oracle reranking하면
teacher-only hard failure class와 raw distribution을 동시에 개선할 수 있는가?
```

결론부터 말하면, 단순 result-count reranking은 hard failure class에서는 강하게 회복되지만 raw500
전체에서는 query-only skeleton보다 낮았다. 그러나 lexical/evidence-aware reranking으로 score를 확장하면
raw500 전체에서도 query-only skeleton을 아주 작게 넘고, hard failure class에서는 가장 큰 회복을 보인다.

따라서 현재까지의 핵심 결론은 다음과 같다.

```text
작은 모델은 teacher query를 단순 SFT/DPO로 더 외우게 하는 것보다,
search-time에 여러 query 후보를 만들고 local retrieval evidence로 고르는 구조에서
더 일관된 회복 신호를 보인다.
```

다만 lexical target-5 reranker는 retrieval volume도 크게 늘린다. 즉 reference recall은 좋아졌지만,
answer model 입장에서는 distractor evidence도 많아졌다. 이 후속 문제는
[`evidence_filter_report.md`](evidence_filter_report.md)에서 마지막 diagnostic으로 따로 평가했다.

## 2. 방법

Script:

```text
scripts/eval_query_candidate_rerank_dmr.py
```

기존 deterministic skeleton은 매 search step마다 query를 하나만 생성한다.

```text
probe -> one query -> local search -> answer
```

candidate rerank는 search step마다 query 후보를 여러 개 생성하고, 각 후보를 실제 local substring
retrieval에 넣은 뒤 non-oracle score로 하나를 선택한다.

```text
probe -> 5 candidate queries -> local search each candidate -> select one query -> answer
```

중요한 점은 reference answer를 reranking score에 쓰지 않는다는 것이다. Score는 아래 정보만 사용한다.

| Component | Meaning |
| --- | --- |
| result-count score | `0` result query를 벌점 주고, target result count에 가까운 query를 선호 |
| specificity score | 너무 짧거나 너무 긴 query를 벌점 주고, 적당한 phrase를 선호 |
| repeat penalty | 이미 선택한 query를 반복하면 벌점 |
| lexical evidence overlap | retrieved message와 probe 사이의 lexical overlap을 선호 |
| query overlap | query 자체가 probe의 핵심 단어와 너무 무관하지 않도록 보정 |
| broad-result penalty | 너무 broad한 query가 많은 결과를 가져오는 경우 벌점 |
| repeated-evidence penalty | 여러 query가 같은 evidence만 반복해서 가져오는 경우 벌점 |

따라서 이 실험은 oracle reranking이 아니라 local retrieval feedback을 이용한 controller diagnostic이다.

## 3. 실행 조건

Count-only 100-row ablation:

```bash
docker compose exec -T nano-memgpt-dev python scripts/eval_query_candidate_rerank_dmr.py \
  --query-model nano-memgpt-llama3-query-only-r16 \
  --answer-model nano-memgpt-llama3-r16 \
  --output-dir data/evaluation/query_candidate_rerank_query_only100_temp0 \
  --offset 0 \
  --limit 100 \
  --max-searches 3 \
  --num-candidates 5 \
  --target-results 3 \
  --query-max-tokens 160 \
  --answer-max-tokens 96 \
  --temperature 0
```

Lexical target-5 raw500 main run:

```bash
docker compose exec -T nano-memgpt-dev python scripts/eval_query_candidate_rerank_dmr.py \
  --query-model nano-memgpt-llama3-query-only-r16 \
  --answer-model nano-memgpt-llama3-r16 \
  --output-dir data/evaluation/query_candidate_rerank_lexical500_temp0_target5 \
  --offset 0 \
  --limit 500 \
  --max-searches 3 \
  --num-candidates 5 \
  --target-results 5 \
  --query-max-tokens 160 \
  --answer-max-tokens 96 \
  --temperature 0 \
  --scoring-mode lexical
```

## 4. 100-row Ablation

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.23` | `0.23` | `3.70` |
| Candidate count rerank, temp `0.4`, target `3` | `0.19` | `0.20` | `3.13` |
| Candidate count rerank, temp `0`, target `3` | `0.24` | `0.26` | `3.03` |
| Candidate count rerank, temp `0`, target `5` | `0.24` | `0.26` | `3.96` |
| Candidate lexical rerank, temp `0`, target `3` | `0.23` | `0.26` | `3.25` |
| Candidate lexical rerank, temp `0`, target `5` | `0.25` | `0.27` | `4.52` |

100-row에서는 lexical target-5가 가장 높았다. 다만 mean retrieved가 `4.52`로 늘어나기 때문에,
전체 500-row에서도 같은 방향이 유지되는지 확인해야 했다.

## 5. Raw500 Result

| Query strategy | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | `0.244` | `0.216` | `3.744` |
| Hard-positive SFT skeleton | `0.184` | `0.180` | `2.882` |
| Preference zero-DPO skeleton | `0.166` | `0.158` | `2.564` |
| Candidate count rerank, target `3` | `0.210` | `0.202` | `3.218` |
| Candidate lexical rerank, target `5` | `0.246` | `0.224` | `4.742` |

Raw500 전체에서 lexical target-5 reranker는 retrieved-reference `0.246`, containment `0.224`를 기록했다.
이는 query-only skeleton의 `0.244`/`0.216`을 아주 작게 넘는 결과다.

차이는 크지 않지만 의미는 있다. 이전 SFT/DPO/count-rerank 조건은 모두 raw500 전체에서 query-only보다
낮았고, lexical target-5는 hard set을 크게 복구하면서도 raw distribution을 무너뜨리지 않은 첫 조건이다.

## 6. Teacher Subset 비교

### 6.1 Approved 398 subset

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.367` | `0.342` | `3.249` |
| Query-only skeleton | `0.276` | `0.249` | `3.761` |
| Candidate count rerank, target `3` | `0.239` | `0.231` | `3.219` |
| Candidate lexical rerank, target `5` | `0.276` | `0.256` | `4.774` |

Approved subset에서 lexical target-5는 retrieved-reference 기준으로 query-only와 같고, containment는
`0.249`에서 `0.256`으로 조금 높다. 그러나 teacher max-3 query의 `0.367`/`0.342`에는 아직 멀다.

### 6.2 Teacher-search 302 subset

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.483` | `0.450` | `4.281` |
| Query-only skeleton | `0.272` | `0.238` | `3.762` |
| Candidate count rerank, target `3` | `0.232` | `0.222` | `3.195` |
| Candidate lexical rerank, target `5` | `0.278` | `0.245` | `4.732` |

Teacher-search subset에서도 lexical target-5는 query-only보다 조금 높지만 teacher gap을 닫지는 못한다.
즉 이 실험은 "teacher-level query selection을 달성했다"가 아니라 "hard failure를 덜 망가뜨리는
search-time selection 구조를 찾았다"로 해석해야 한다.

## 7. Hard Failure Class 복구

가장 중요한 결과는 hard set 내부 비교다.

### 7.1 Teacher-only hard 72 rows

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `1.000` | `0.847` | `4.125` |
| Query-only skeleton | `0.000` | `0.014` | `2.028` |
| Hard-positive SFT | `0.097` | `0.097` | `1.611` |
| Preference zero-DPO | `0.111` | `0.097` | `1.500` |
| Candidate count rerank, target `3` | `0.278` | `0.236` | `2.958` |
| Candidate lexical rerank, target `5` | `0.347` | `0.306` | `4.097` |

Lexical target-5 reranker는 query-only가 하나도 retrieve하지 못한 72개 hard rows에서 `25/72`
reference retrieval을 복구했다. Containment도 `1/72`에서 `22/72`로 올렸다.

이 결과가 중요하다. Hard-positive SFT와 DPO는 hard set에서 각각 `7-8`개 retrieval을 복구했지만,
lexical target-5는 `25`개를 복구한다. 즉 query 후보를 생성하고 local evidence로 고르는 구조가
hard failure class에 훨씬 직접적으로 맞는다.

### 7.2 Zero-result-only 51 rows

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `1.000` | `0.804` | `4.078` |
| Query-only skeleton | `0.000` | `0.020` | `1.118` |
| Hard-positive SFT | `0.098` | `0.098` | `0.647` |
| Preference zero-DPO | `0.098` | `0.078` | `0.706` |
| Candidate count rerank, target `3` | `0.314` | `0.255` | `3.000` |
| Candidate lexical rerank, target `5` | `0.353` | `0.294` | `3.882` |

Zero-only hard set에서도 lexical target-5가 가장 강하다. Query-only 대비 retrieval gain은 `18`개,
containment gain은 `15`개다.

## 8. 해석

이 결과는 세 층으로 읽어야 한다.

첫째, naive weight update는 retrieval policy를 안정적으로 개선하지 못했다. Hard-positive SFT와
zero-DPO는 targeted hard set에서는 약간 회복되지만 raw500 전체 성능을 뚜렷하게 떨어뜨린다.

둘째, count-only candidate reranking은 hard failure class를 크게 복구하지만 raw500 전체에서는
query-only skeleton보다 낮다. 즉 "여러 후보를 만들고 결과 수로 고른다"는 방향은 맞지만,
result count만으로는 broad query와 distractor를 충분히 제어하지 못한다.

셋째, lexical target-5 reranking은 현재까지 가장 균형 잡힌 non-oracle 조건이다. Raw500에서
query-only skeleton을 소폭 넘고, hard72와 zero51에서는 가장 큰 회복을 보인다. 다만 mean retrieved가
`4.742`까지 늘어났기 때문에 answer 단계에서는 더 많은 distractor를 처리해야 한다.

따라서 지금의 결론은 다음과 같다.

```text
Query-only LoRA 안에는 유효한 query-policy signal이 있다.
하지만 single-shot query generation이나 small hard-set SFT/DPO로는 충분하지 않다.
작은 모델의 MemGPT 복구에는 search-time candidate generation,
retrieval-feedback reranking, evidence-grounded answering을 분리해야 한다.
```

## 9. 후속 실험 결과

이 보고서 이후 evidence filtering final diagnostic을 실행했다. Lexical target-5 결과를 입력으로 사용해
answer 직전에 retrieved evidence를 non-oracle lexical top-6으로 줄였다.

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Candidate lexical rerank, target `5` | `0.246` | `0.224` | `4.742` |
| Candidate lexical rerank + evidence filter top-6 | `0.234` | `0.224` | `3.426` |

Hard72에서는 containment `22/72`, zero51에서는 containment `15/51`을 유지하면서 mean retrieved를
각각 `4.097 -> 3.014`, `3.882 -> 2.843`으로 줄였다. 따라서 evidence filtering은 성능을 추가로
올리지는 못했지만, candidate reranker의 높은 evidence volume을 줄이는 데는 성공했다.

## 10. 다음 연구 방향

v1 실험은 여기서 freeze한다. 다음 항목들은 새 실험을 즉시 시작하자는 뜻이 아니라, paper의 limitations
또는 future work로 남길 수 있는 방향이다.

| Priority | Experiment | Why |
| ---: | --- | --- |
| 1 | learned 또는 embedding evidence reranker | retrieved message와 probe/query 사이의 semantic match를 lexical overlap보다 안정적으로 평가 |
| 2 | candidate diversity constraint | 후보 query들이 같은 broad term을 반복하지 못하게 제한 |
| 3 | target-results adaptive policy | easy row는 3개 내외, hard row는 5개 내외로 retrieval volume 조절 |
| 4 | evidence-grounded answer adapter | retrieval이 맞았는데 answer가 틀린 row를 분리 개선 |

논문 framing은 다음처럼 정리할 수 있다.

```text
Small open-source models can learn query syntax and partial query style,
but direct SFT/DPO does not generalize retrieval policy.
Search-time candidate generation and retrieval-feedback reranking better targets
the teacher-only failure class. A lexical reranker is the first non-oracle variant
that slightly improves the full raw distribution, but its higher evidence volume
creates the next answer-side distractor problem.
```

## 11. Artifacts

```text
scripts/eval_query_candidate_rerank_dmr.py
data/evaluation/query_candidate_rerank_query_only20/
data/evaluation/query_candidate_rerank_query_only100/
data/evaluation/query_candidate_rerank_query_only100_temp0/
data/evaluation/query_candidate_rerank_query_only100_temp0_target5/
data/evaluation/query_candidate_rerank_query_only500_temp0_target3/
data/evaluation/query_candidate_rerank_lexical20_temp0_target3/
data/evaluation/query_candidate_rerank_lexical100_temp0_target3/
data/evaluation/query_candidate_rerank_lexical100_temp0_target5/
data/evaluation/query_candidate_rerank_lexical500_temp0_target5/
docs/evidence_filter_report.md
data/evaluation/evidence_filter_lexical500_k6/
```
