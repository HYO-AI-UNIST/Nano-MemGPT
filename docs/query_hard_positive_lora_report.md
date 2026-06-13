# Query Hard-Positive LoRA Pilot Report

## 1. 목적

이 문서는 `query_hard_positive_sft.jsonl`로 학습한 작은 LoRA pilot 결과를 정리한다.

이 실험의 질문은 단순하다.

```text
Teacher-only hard query row 72개를 positive-query SFT로 추가 학습하면,
deterministic query skeleton에서 reference retrieval이 좋아지는가?
```

결론부터 말하면, 전체 성능은 하락했다. 하지만 학습에 사용한 hard set 내부에서는 작은 개선이
있었다. 따라서 이 결과는 "hard-positive query signal은 유효하지만, 72개 SFT만으로는 general
query policy를 개선하지 못하고 오히려 over-specialization을 만든다"로 해석하는 것이 맞다.

## 2. 학습 조건

Dataset:

```text
data/trajectories/query_hard_positive_sft.jsonl
```

Training command:

```bash
docker compose exec -T nano-memgpt-dev bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_sft.py \
    --trajectories data/trajectories/query_hard_positive_sft.jsonl \
    --output-dir outputs/lora_query_hard_positive_r16 \
    --max-length 4096 \
    --rank 16 \
    --alpha 32 \
    --learning-rate 1.0e-5 \
    --epochs 1.0 \
    --gradient-accumulation-steps 8 \
    --logging-steps 1 \
    --save-steps 20'
```

Training summary:

| Item | Value |
| --- | ---: |
| records | `72` |
| train/eval split | `68/4` |
| overlength dropped | `0` |
| train steps | `9` |
| train loss | `3.622` |
| eval loss | `3.899` |
| eval mean token accuracy | `0.2917` |

학습 중 `bitsandbytes` CUDA 13.1 binary warning이 출력되었지만, 이 run은 BF16 PEFT/SFT 경로로
정상 완료되었다.

## 3. vLLM Serving

새 adapter:

```text
outputs/lora_query_hard_positive_r16/final_adapter/
```

vLLM exposed model id:

```text
nano-memgpt-llama3-query-hard-positive-r16
```

`docker-compose.lora.yaml`에 다음 LoRA module을 추가했다.

```text
nano-memgpt-llama3-query-hard-positive-r16=/workspace/nano_memgpt_outputs/lora_query_hard_positive_r16/final_adapter
```

주의할 점: `.env`의 `STUDENT_MODEL_ID`가 `meta-llama/Meta-Llama-3-8B-Instruct`로 되어 있으면
vLLM restart 시 gated repo 403으로 실패한다. 이번 run에서는 아래처럼 mirror checkpoint를 명시해
재시작했다.

```bash
STUDENT_MODEL_ID=NousResearch/Meta-Llama-3-8B-Instruct \
STUDENT_TOOL_CALL_PARSER=nano_strict_llama \
docker compose -f docker-compose.yaml -f docker-compose.lora.yaml \
  --profile llama up -d --force-recreate llama-vllm
```

## 4. Skeleton Evaluation

Evaluation command:

```bash
docker compose exec -T nano-memgpt-dev python scripts/eval_query_skeleton_dmr.py \
  --query-model nano-memgpt-llama3-query-hard-positive-r16 \
  --answer-model nano-memgpt-llama3-r16 \
  --output-dir data/evaluation/query_skeleton_dmr_hard_positive500 \
  --offset 0 \
  --limit 500 \
  --max-searches 3 \
  --query-max-tokens 64 \
  --answer-max-tokens 96
```

### 4.1 100-row smoke

| Query generator | Rows | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16 skeleton | `100` | `0.23` | `0.23` | `3.70` |
| Hard-positive r16 skeleton | `100` | `0.20` | `0.21` | `2.66` |

### 4.2 Raw 500-row result

| Query generator | Rows | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16 skeleton | `500` | `0.244` | `0.216` | `3.744` |
| Hard-positive r16 skeleton | `500` | `0.184` | `0.180` | `2.882` |

전체 결과는 negative다. Hard-positive SFT는 mean retrieved를 줄였지만 reference hit도 같이 줄였다.
즉 retrieval volume을 줄이는 방향으로 query distribution이 이동했지만, specificity가 충분히 좋아지지
않았다.

## 5. Teacher Max-3와의 비교

### 5.1 Approved 398 subset

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.367` | `0.342` | `3.25` |
| Query-only r16 skeleton | `0.276` | `0.249` | `3.76` |
| Hard-positive r16 skeleton | `0.219` | `0.221` | `2.93` |

### 5.2 Teacher-search 302 subset

| Query source | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | `0.272` | `0.238` | `3.76` |
| Hard-positive r16 skeleton | `0.215` | `0.205` | `2.90` |

이 비교도 같은 결론을 준다. Hard-positive SFT는 query-only r16보다 더 적게 retrieve하지만,
teacher-like reference retrieval에는 가까워지지 못했다.

## 6. Hard Set 내부 변화

학습에 사용한 teacher-only hard row 72개만 따로 보면 작은 개선이 있다.

| Metric on 72 hard rows | Query-only r16 | Hard-positive r16 |
| --- | ---: | ---: |
| Retrieved-reference | `0/72` (`0.000`) | `7/72` (`0.097`) |
| Containment | `1/72` (`0.014`) | `7/72` (`0.097`) |
| Mean retrieved | `2.03` | `1.61` |
| Retrieval gained | n/a | `7` |
| Retrieval lost | n/a | `0` |
| Containment gained | n/a | `7` |
| Containment lost | n/a | `1` |

이 부분은 중요하다. Hard-positive SFT가 완전히 무의미한 것은 아니다. 학습한 hard row에서는
실제로 일부 reference retrieval을 복구한다. 그러나 그 이득이 전체 distribution으로 generalize되지
않고, 오히려 non-hard rows에서 no-result/UNKNOWN을 늘린다.

## 7. 해석

이번 pilot은 다음 결론을 준다.

1. Teacher-only hard rows에는 유효한 학습 신호가 있다.
2. 하지만 72개 positive-only SFT는 너무 작고 편향되어, 전체 query policy를 좁히는 부작용이 크다.
3. Query objective는 positive-only SFT보다 preference/ranking 또는 candidate reranking 쪽이 더
   타당하다.
4. 특히 zero-result-only negative subset 51개를 이용한 preference pilot이 다음 우선순위다.

즉 다음 실험은 `query_hard_positive_sft.jsonl`을 더 오래 학습하는 것이 아니다. 그 방향은 overfit을
키울 가능성이 높다. 대신 `query_hard_negative_preferences_zero_only.jsonl`을 사용해 "teacher query가
student query보다 왜 나은지"를 직접 비교하는 objective가 더 맞다.

## 8. Artifacts

```text
outputs/lora_query_hard_positive_r16/final_adapter/
data/evaluation/query_skeleton_dmr_hard_positive100/
data/evaluation/query_skeleton_dmr_hard_positive500/
data/analysis/query_skeleton_gap_hard_positive/
docker-compose.lora.yaml
```
