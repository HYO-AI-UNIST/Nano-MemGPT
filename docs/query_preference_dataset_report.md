# Query Preference Dataset Report

## 1. 목적

이 문서는 teacher max-3 query skeleton과 query-only r16 skeleton의 row-level gap을 이용해
retrieval-supervised query dataset을 만든 결과를 정리한다.

앞선 분석에서 teacher-search 302 subset은 다음 분포를 보였다.

| Category | Retrieval rows |
| --- | ---: |
| both | `74` |
| teacher only | `72` |
| student only | `8` |
| neither | `148` |

가장 깨끗한 학습 신호는 `teacher only` row다. 이 row에서는 teacher query가 reference-containing
message를 retrieve했지만, query-only r16 query는 retrieve하지 못했다. 따라서 teacher query를
positive, student query를 negative로 둘 수 있다.

## 2. Export 방식

스크립트:

```text
scripts/export_query_preference_dataset.py
```

기본 입력:

```text
teacher:
  data/evaluation/teacher_query_skeleton_dmr_approved500_max3/
  teacher-query-to-nano-memgpt-llama3-r16-skeleton-offset-0-limit-500.jsonl

student:
  data/evaluation/query_skeleton_dmr_evidence_only500/
  nano-memgpt-llama3-query-only-r16-to-nano-memgpt-llama3-r16-skeleton-searches-3-offset-0-limit-500.jsonl
```

기본 필터:

| Option | Value | Meaning |
| --- | --- | --- |
| `subset` | `teacher_search` | teacher가 실제 search를 수행한 row만 사용 |
| `category` | `teacher_only` | teacher는 reference retrieval 성공, student는 실패 |
| `positive_strategy` | `first_hit` | teacher trace 중 reference를 처음 hit한 query를 chosen으로 사용 |
| `negative_strategy` | `zero_result_preferred` | 가능하면 result가 0개인 student query를 rejected로 사용 |

출력은 두 종류다.

| Output | Purpose |
| --- | --- |
| `query_hard_negative_preferences.jsonl` | DPO/ORPO/ranking-style preference 학습용 `prompt/chosen/rejected` dataset |
| `query_hard_positive_sft.jsonl` | 기존 `train_lora_sft.py`가 바로 읽을 수 있는 positive-query SFT dataset |

## 3. 생성 결과

### 3.1 Default hard-negative dataset

```text
python3 scripts/export_query_preference_dataset.py
```

| Item | Value |
| --- | ---: |
| joined rows | `398` |
| filtered teacher-only rows | `72` |
| preference records | `72` |
| SFT steps | `72` |

출력:

```text
data/trajectories/query_hard_negative_preferences.jsonl
data/trajectories/query_hard_negative_preferences.summary.json
data/trajectories/query_hard_positive_sft.jsonl
```

### 3.2 Zero-result-only clean subset

```text
python3 scripts/export_query_preference_dataset.py \
  --negative-strategy zero_result_only \
  --preference-output data/trajectories/query_hard_negative_preferences_zero_only.jsonl \
  --sft-output data/trajectories/query_hard_positive_sft_zero_only.jsonl \
  --summary-output data/trajectories/query_hard_negative_preferences_zero_only.summary.json
```

| Item | Value |
| --- | ---: |
| preference records | `51` |
| SFT steps | `51` |
| skipped because no zero-result negative | `21` |

이 subset은 규모는 작지만 negative가 더 깨끗하다. Default set은 `72`개로 크지만, rejected query가
일부 유용한 paraphrase evidence를 retrieve할 수 있다. 따라서 첫 학습은 default SFT로 시도하되,
preference/ranking 학습에는 zero-result-only subset을 먼저 쓰는 것이 더 안전하다.

### 3.3 All-hit variant

Teacher trace에서 reference를 hit한 모든 query를 positive로 쓰는 variant도 만들었다.

```text
data/trajectories/query_hard_negative_preferences_all_hits.jsonl
data/trajectories/query_hard_positive_sft_all_hits.jsonl
```

| Item | Value |
| --- | ---: |
| preference records | `86` |
| SFT steps | `86` |

이 variant는 data size를 늘리지만 같은 row에서 비슷한 query가 중복될 수 있다. 초기 실험에서는
`first_hit` default가 더 해석하기 쉽다.

## 4. 예시

```text
Probe:
  Hey, remember that time we talked about our pets? What kind of pet do you have?

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

이 예시는 현재 병목을 잘 보여 준다. Student는 memory를 더 찾기 전에 plausible answer prior인
`dogs`를 query로 사용한다. Teacher는 broad query가 부족하다는 것을 보고 answer-bearing literal
`cat`으로 좁힌다.

## 5. Trainer 호환성 검증

기존 SFT trainer가 positive-query SFT dataset을 읽을 수 있는지 `--prepare-only`로 검증했다.

```text
docker compose exec -T nano-memgpt-dev python scripts/train_lora_sft.py \
  --trajectories data/trajectories/query_hard_positive_sft.jsonl \
  --output-dir outputs/query_hard_positive_sft_prepare_check \
  --max-length 4096 \
  --rank 16 \
  --alpha 32 \
  --prepare-only
```

결과:

| Item | Value |
| --- | ---: |
| records | `72` |
| train records | `68` |
| eval records | `4` |
| overlength dropped | `0` |
| action counts | `target_text: 72` |

즉 positive-query SFT dataset은 현재 LoRA trainer와 바로 호환된다.

## 6. 후속 실험 결과와 다음 단계

초기 우선순위는 두 갈래였다.

| Experiment | Dataset | Why |
| --- | --- | --- |
| Hard-positive query SFT | `query_hard_positive_sft.jsonl` | 기존 pipeline으로 가장 빨리 실행 가능 |
| Query preference/ranking pilot | `query_hard_negative_preferences_zero_only.jsonl` | negative noise가 적어 ranking objective 검증에 적합 |

Hard-positive query SFT는 실행했다. 결과는 `docs/query_hard_positive_lora_report.md`에 정리했다.
요약하면 raw500 retrieved-reference/containment가 `0.184/0.180`으로 기존 query-only r16
`0.244/0.216`보다 낮아졌다. 다만 학습 hard set 내부에서는 retrieved-reference가 `0/72`에서
`7/72`로 올랐다.

Zero-result-only preference DPO도 실행했다. 결과는 `docs/query_preference_dpo_report.md`에
정리했다. 요약하면 raw500 retrieved-reference/containment가 `0.166/0.158`로 더 낮아졌지만, hard
72 rows 내부에서는 retrieved-reference가 `0/72`에서 `8/72`로 올랐다.

따라서 teacher query signal은 존재하지만, 작은 hard set 기반 positive SFT/DPO는 전체 query policy
개선 방향으로는 부적합하다. 다음 우선순위는 더 긴 LoRA 학습이 아니라 candidate query generation과
local retrieval reranking이다.

## 7. Artifacts

```text
scripts/export_query_preference_dataset.py
scripts/train_query_preference_dpo.py
data/trajectories/query_hard_negative_preferences.jsonl
data/trajectories/query_hard_negative_preferences.summary.json
data/trajectories/query_hard_positive_sft.jsonl
data/trajectories/query_hard_negative_preferences_zero_only.jsonl
data/trajectories/query_hard_negative_preferences_zero_only.summary.json
data/trajectories/query_hard_positive_sft_zero_only.jsonl
data/trajectories/query_hard_negative_preferences_all_hits.jsonl
data/trajectories/query_hard_negative_preferences_all_hits.summary.json
data/trajectories/query_hard_positive_sft_all_hits.jsonl
outputs/query_hard_positive_sft_prepare_check/run_manifest.json
outputs/lora_query_preference_zero_dpo_r16/final_adapter/
data/evaluation/query_skeleton_dmr_pref_zero_dpo500/
```
