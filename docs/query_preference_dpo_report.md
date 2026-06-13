# Query Preference DPO Pilot Report

## 1. 목적

이 문서는 `query_hard_negative_preferences_zero_only.jsonl`로 학습한 DPO pilot 결과를 정리한다.

직전 hard-positive SFT 실험은 중요한 단서를 줬다. 학습 hard set 내부에서는 reference retrieval을 일부
복구했지만, raw500 전체에서는 query-only r16보다 성능이 하락했다. 따라서 이번 실험의 질문은 다음과
같다.

```text
Positive teacher query만 SFT로 따라 쓰게 하는 대신,
teacher hit query를 chosen, student zero-result query를 rejected로 두고
preference/ranking objective를 적용하면 query policy가 더 좋아지는가?
```

결론부터 말하면, 전체 성능은 다시 하락했다. 다만 hard set 내부에서는 query-only가 하나도 맞히지
못하던 reference retrieval을 일부 복구했다. 즉 preference signal 자체는 유효하지만, 51개 zero-only
preference pair만으로는 general query policy를 개선하기에 부족했다.

## 2. Dataset

사용한 preference dataset:

```text
data/trajectories/query_hard_negative_preferences_zero_only.jsonl
```

이 dataset은 teacher-search subset 중 다음 조건을 만족하는 row만 사용한다.

| 조건 | 의미 |
| --- | --- |
| teacher positive | teacher query trace가 reference-containing memory를 retrieve함 |
| student negative | query-only r16 trace는 reference retrieval에 실패함 |
| zero-result rejected | rejected student query의 search result 수가 정확히 `0`임 |

Prepare-only 결과:

| Item | Value |
| --- | ---: |
| preference records | `51` |
| train/eval split | `47/4` |
| overlength dropped | `0` |
| rejected result count | all `0` |
| max prompt tokens | `402` |
| max total tokens | `404` |

이 subset을 먼저 선택한 이유는 rejected signal의 noise를 줄이기 위해서다. 단순히 reference를 못 맞힌
query라고 해서 항상 나쁜 query는 아닐 수 있다. 하지만 zero-result query는 적어도 local substring
retrieval contract에서는 명확히 실패한 query이므로 preference target으로 더 깨끗하다.

## 3. 학습 조건

Script:

```text
scripts/train_query_preference_dpo.py
```

Prepare-only command:

```bash
docker compose exec -T nano-memgpt-dev bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python scripts/train_query_preference_dpo.py \
    --preferences data/trajectories/query_hard_negative_preferences_zero_only.jsonl \
    --output-dir outputs/lora_query_preference_zero_dpo_r16_prepare_check \
    --max-length 4096 \
    --rank 16 \
    --alpha 32 \
    --learning-rate 5e-6 \
    --epochs 1.0 \
    --gradient-accumulation-steps 8 \
    --prepare-only'
```

Training command:

```bash
docker compose exec -T nano-memgpt-dev bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python scripts/train_query_preference_dpo.py \
    --preferences data/trajectories/query_hard_negative_preferences_zero_only.jsonl \
    --output-dir outputs/lora_query_preference_zero_dpo_r16 \
    --max-length 4096 \
    --rank 16 \
    --alpha 32 \
    --learning-rate 5e-6 \
    --beta 0.1 \
    --epochs 1.0 \
    --gradient-accumulation-steps 8 \
    --logging-steps 1 \
    --save-steps 20'
```

Training summary:

| Item | Value |
| --- | ---: |
| train steps | `6` |
| train loss | `0.691` |
| eval loss | `0.689` |
| eval reward accuracy | `0.75` |
| eval reward margin | `0.0084` |

이 숫자는 작은 eval split 4개에서 나온 것이므로 성능 주장으로 쓰면 안 된다. 여기서는 DPO objective가
실행 가능하고, chosen/rejected 방향의 약한 학습 신호가 생겼다는 sanity check로만 사용한다.

## 4. vLLM Serving

새 adapter:

```text
outputs/lora_query_preference_zero_dpo_r16/final_adapter/
```

vLLM exposed model id:

```text
nano-memgpt-llama3-query-pref-zero-dpo-r16
```

`docker-compose.lora.yaml`에 다음 LoRA module을 추가했다.

```text
nano-memgpt-llama3-query-pref-zero-dpo-r16=/workspace/nano_memgpt_outputs/lora_query_preference_zero_dpo_r16/final_adapter
```

vLLM restart는 gated Meta repo를 피하기 위해 mirror checkpoint를 명시해 실행했다.

```bash
STUDENT_MODEL_ID=NousResearch/Meta-Llama-3-8B-Instruct \
STUDENT_TOOL_CALL_PARSER=nano_strict_llama \
docker compose -f docker-compose.yaml -f docker-compose.lora.yaml \
  --profile llama up -d --force-recreate llama-vllm
```

## 5. Skeleton Evaluation

Evaluation command:

```bash
docker compose exec -T nano-memgpt-dev python scripts/eval_query_skeleton_dmr.py \
  --query-model nano-memgpt-llama3-query-pref-zero-dpo-r16 \
  --answer-model nano-memgpt-llama3-r16 \
  --output-dir data/evaluation/query_skeleton_dmr_pref_zero_dpo500 \
  --offset 0 \
  --limit 500 \
  --max-searches 3 \
  --query-max-tokens 64 \
  --answer-max-tokens 96
```

### 5.1 100-row smoke

| Query generator | Rows | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16 skeleton | `100` | `0.23` | `0.23` | `3.70` |
| Hard-positive r16 skeleton | `100` | `0.20` | `0.21` | `2.66` |
| Preference zero-DPO r16 skeleton | `100` | `0.19` | `0.18` | `2.21` |

100-row smoke부터 negative signal이 보였다. DPO 모델은 retrieval volume을 더 줄였지만, reference hit를
회복하지 못했다.

### 5.2 Raw 500-row result

| Query generator | Rows | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16 skeleton | `500` | `0.244` | `0.216` | `3.744` |
| Hard-positive r16 skeleton | `500` | `0.184` | `0.180` | `2.882` |
| Preference zero-DPO r16 skeleton | `500` | `0.166` | `0.158` | `2.564` |

전체 결과는 hard-positive SFT보다도 낮다. DPO는 zero-result rejected를 직접 피하도록 학습했지만, 실제
generation에서는 "더 좋은 literal substring query"를 만드는 대신 더 좁고 보수적인 query policy로
이동했다. 그 결과 no-result/UNKNOWN row가 늘었다.

## 6. Teacher Max-3와의 비교

### 6.1 Approved 398 subset

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.367` | `0.342` | `3.249` |
| Query-only r16 skeleton | `0.276` | `0.249` | `3.761` |
| Hard-positive r16 skeleton | `0.219` | `0.221` | `2.930` |
| Preference zero-DPO r16 skeleton | `0.198` | `0.196` | `2.485` |

### 6.2 Teacher-search 302 subset

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.483` | `0.450` | `4.281` |
| Query-only r16 skeleton | `0.272` | `0.238` | `3.762` |
| Hard-positive r16 skeleton | `0.215` | `0.205` | `2.901` |
| Preference zero-DPO r16 skeleton | `0.205` | `0.189` | `2.533` |

Teacher-search subset은 원래 query가 필요한 row만 모은 비교다. 여기서도 DPO는 teacher와의 gap을
줄이지 못했다. 오히려 query-only 대비 reference retrieval이 낮아졌다.

## 7. Hard Set 내부 변화

전체 성능은 negative지만, 학습 hard set 내부에서는 작은 회복이 있다.

### 7.1 Teacher-only hard 72 rows

| Metric on 72 hard rows | Query-only r16 | Hard-positive r16 | Preference zero-DPO r16 |
| --- | ---: | ---: | ---: |
| Retrieved-reference | `0/72` (`0.000`) | `7/72` (`0.097`) | `8/72` (`0.111`) |
| Containment | `1/72` (`0.014`) | `7/72` (`0.097`) | `7/72` (`0.097`) |
| Mean retrieved | `2.028` | `1.611` | `1.500` |
| Retrieval gained vs query-only | n/a | `7` | `8` |
| Retrieval lost vs query-only | n/a | `0` | `0` |
| Containment gained vs query-only | n/a | `7` | `7` |
| Containment lost vs query-only | n/a | `1` | `1` |

### 7.2 Zero-result-only 51 rows

| Metric on zero-only 51 rows | Query-only r16 | Hard-positive r16 | Preference zero-DPO r16 |
| --- | ---: | ---: | ---: |
| Retrieved-reference | `0/51` (`0.000`) | `5/51` (`0.098`) | `5/51` (`0.098`) |
| Containment | `1/51` (`0.020`) | `5/51` (`0.098`) | `4/51` (`0.078`) |
| Mean retrieved | `1.118` | `0.647` | `0.706` |
| Retrieval gained vs query-only | n/a | `5` | `5` |
| Retrieval lost vs query-only | n/a | `0` | `0` |
| Containment gained vs query-only | n/a | `5` | `4` |
| Containment lost vs query-only | n/a | `1` | `1` |

이 결과는 중요하다. Hard-positive SFT와 zero-DPO 모두 hard set 내부에서는 query-only보다 낫다.
하지만 이 이득은 전체 distribution으로 일반화되지 않는다.

## 8. 해석

이번 pilot의 핵심 해석은 다음과 같다.

1. Teacher query에는 실제로 student query를 보정할 수 있는 signal이 있다.
2. Positive-only SFT도, zero-negative DPO도 hard rows 일부는 복구한다.
3. 그러나 데이터가 작고 hard-case biased라서 일반 query policy는 오히려 좁아진다.
4. 현재 병목은 단순히 "좋은 query 문자열을 더 많이 외우게 하는 것"이 아니다.
5. 더 타당한 다음 방향은 candidate generation 후 reranking, 또는 retrieval feedback을 이용한
   online query selection이다.

특히 DPO는 rejected zero-result query를 직접 피하도록 학습했는데도 raw500 성능이 낮아졌다. 이는
모델이 "검색 결과가 나올 만한 literal substring을 만드는 법"을 일반화하지 못하고, 특정 hard examples의
surface form만 부분적으로 흡수했음을 시사한다.

## 9. 다음 우선순위

이제 `query_hard_positive_sft.jsonl` 또는 `query_hard_negative_preferences_zero_only.jsonl`을 더 오래
학습하는 방향은 우선순위가 낮다. 다음 실험은 아래 순서가 더 적합하다.

| Priority | Experiment | Rationale |
| ---: | --- | --- |
| 1 | Candidate query generation + local retrieval reranking | 모델이 여러 후보 query를 만들고, 실제 retrieval result가 reference-like evidence를 포함할 가능성이 높은 후보를 선택하게 한다. |
| 2 | Teacher query decomposition audit | teacher query가 왜 맞는지 lexical category, entity type, phrase length, previous-search dependency로 분해한다. |
| 3 | Evidence-aware answer adapter | retrieval channel을 고정하거나 teacher/candidate-selected evidence를 준 상태에서 answer module을 따로 개선한다. |
| 4 | Larger preference data collection | 51/72개가 아니라 수백~수천 개 preference pair를 만들 수 있을 때 DPO/IPO를 다시 시도한다. |

현재 가장 좋은 연구 framing은 "small LLM의 MemGPT 실패는 tool-call syntax 문제가 아니라 query/evidence
selection 문제이며, naive SFT/DPO는 hard examples를 부분적으로 외우지만 retrieval policy를 일반화하지
못한다"이다.

## 10. Artifacts

```text
scripts/train_query_preference_dpo.py
outputs/lora_query_preference_zero_dpo_r16_prepare_check/run_manifest.json
outputs/lora_query_preference_zero_dpo_r16/final_adapter/
data/evaluation/query_skeleton_dmr_pref_zero_dpo100/
data/evaluation/query_skeleton_dmr_pref_zero_dpo500/
docker-compose.lora.yaml
```
