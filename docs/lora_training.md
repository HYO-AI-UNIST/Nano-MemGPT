# LoRA Distillation 실행 가이드

## 1. 목적

이 단계는 judge가 승인한 GPT-4.1 MemGPT trajectory를 이용해 frozen student의
function-calling 행동을 LoRA로 distill한다. 첫 대상은
`NousResearch/Meta-Llama-3-8B-Instruct`다.

학습 데이터는 다음 artifact다.

```text
data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl
```

scaled teacher 수집에서 raw 500행 중 398행이 judge를 통과했고, 이 행들에서
context-complete `TrajectoryStep` 1,664개를 export했다.

## 2. 학습 target

각 step은 student가 다음에 생성해야 할 명시적 JSON tool call을 target으로 가진다.

```json
{
  "name": "conversation_search",
  "arguments": {
    "query": "Burger King",
    "roles": ["assistant", "user"],
    "limit": 10
  },
  "request_heartbeat": true
}
```

이 표면 형식은 `nano_strict_llama` parser가 받아들일 수 있다. parser는 tool을 대신
선택하거나 빠진 argument를 복구하지 않는다.

## 3. Chat-template 정규화

초기 `scripts/export_sft_jsonl.py`는 artifact 확인용으로 context를 JSON 배열 문자열로
평탄화한다. 실제 inference prompt는 tokenizer chat template을 통과하므로 장시간
학습에는 이 legacy export를 직접 사용하지 않는다.

`src/nanomemgpt/training/formatting.py`의 chat-template 경로는 structured tool-call
history를 명시적 JSON 텍스트로 정규화하고 student tokenizer로 prompt를 렌더링한다.
`scripts/train_lora_sft.py`는 이 경로를 사용한다.

## 4. 길이 감사

학습 전에는 반드시 token-length audit를 실행한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/audit_teacher_sft_dataset.py \
  --trajectories data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl \
  --model NousResearch/Meta-Llama-3-8B-Instruct \
  --max-length 8192 \
  --output-json data/processed/sft/gpt41_paper_substring_scaled_llama_audit.json
```

현재 Llama audit 결과:

| 항목 | 값 |
| --- | ---: |
| step | `1,664` |
| median total tokens | `2,464` |
| p95 total tokens | `4,677` |
| max total tokens | `8,980` |
| 8,192 token 초과 | `3` |

기본 launcher는 TRL의 `chunked_nll` loss를 사용한다. 이 경로는 전체 vocabulary
logits를 한 번에 FP32로 확장하지 않아 긴 sequence의 peak GPU memory를 줄인다.
8,192 token을 넘는 3개 step은 잘라서 target을 훼손하지 않고 학습에서 제외한다.

기본 `nll`로 수행한 6,041-token과 7,792-token 경계 smoke는 GPU OOM을 일으켰다.
`chunked_nll`로 다시 실행한 7,792-token smoke는 정상 완료되었다. 따라서 본 학습은
`max_length=8192`, `loss_type=chunked_nll`을 사용한다.

## 5. Prepare-only 검증

장시간 GPU 학습 전에 split과 제외 행을 확인한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/train_lora_sft.py \
  --rank 16 \
  --output-dir outputs/lora_student_r16 \
  --prepare-only
```

## 6. 짧은 overfit smoke

GPU 0을 학습에 사용하고 GPU 1의 vLLM serving과 분리한다.

```bash
docker compose exec -T \
  -e CUDA_VISIBLE_DEVICES=0 \
  nano-memgpt-dev \
  python scripts/train_lora_sft.py \
  --rank 16 \
  --max-samples 32 \
  --max-steps 2 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --save-steps 1 \
  --output-dir outputs/lora_smoke_r16
```

smoke에서 확인할 것은 loss 계산, GPU memory 적합성, checkpoint와 adapter 저장이다.

추가로 허용 범위 내 긴 context memory smoke를 실행할 수 있다.

```bash
docker compose exec -T \
  -e CUDA_VISIBLE_DEVICES=0 \
  nano-memgpt-dev \
  python scripts/train_lora_sft.py \
  --rank 16 \
  --sample-id dmr-379-step-11-call-0 \
  --eval-ratio 0 \
  --max-steps 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --save-steps 1 \
  --output-dir outputs/lora_chunked_8192_context_smoke_r16
```

## 7. 본 학습

smoke 통과 후 `r=8`, `r=16`을 별도 output directory에 학습한다. `r=8`은
`alpha=16`, `r=16`은 `alpha=32`를 사용해 `alpha / r = 2`를 유지한다.

```bash
docker compose exec -T \
  -e CUDA_VISIBLE_DEVICES=0 \
  nano-memgpt-dev \
  python scripts/train_lora_sft.py \
  --rank 16 \
  --alpha 32 \
  --output-dir outputs/lora_student_r16
```

`r=8`, `alpha=16` 학습은 완료되었다.

| 항목 | 값 |
| --- | ---: |
| optimizer step | `294/294` |
| epoch | `3` |
| train runtime | 약 `2시간 36분 51초` |
| train loss | `1.0890` |
| final eval loss | `0.9009` |
| final eval mean token accuracy | `0.7546` |
| train records | `1,563` |
| eval records | `98` |
| overlength 제외 | `3` |

저장된 adapter는 다음 위치에 있다.

```text
outputs/lora_student_r8/final_adapter/
```

주요 파일은 `adapter_model.safetensors`, `adapter_config.json`, tokenizer 관련 파일이다.
최종 adapter 크기는 약 `31M`이다.

`r=16`, `alpha=32` 학습도 완료되었다. 실행 명령은 다음과 같다.

```bash
docker compose exec -d -T \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e WANDB_DISABLED=true \
  nano-memgpt-dev \
  bash -lc 'python scripts/train_lora_sft.py --rank 16 --alpha 32 --max-length 8192 --output-dir outputs/lora_student_r16 > logs/lora_student_r16.log 2>&1'
```

학습 후에는 adapter를 vLLM에 mount하고 Experiment 1과 동일한 DMR 규약으로
post-training 평가를 반복한다.

| 조건 | train loss | final eval loss | final token accuracy | adapter |
| --- | ---: | ---: | ---: | --- |
| LoRA `r=8`, `alpha=16` | `1.0890` | `0.9009` | `0.7546` | `outputs/lora_student_r8/final_adapter/` |
| LoRA `r=16`, `alpha=32` | `1.0270` | `0.8694` | `0.7599` | `outputs/lora_student_r16/final_adapter/` |

현재 SFT proxy metric에서는 `r=16`이 `r=8`보다 낮은 eval loss와 약간 높은 token accuracy를
보인다. 다만 이것은 teacher tool-call target에 대한 token-level 지표이며, 실제 연구 결론은
post-training DMR loop 평가에서 judge accuracy와 tool-call failure distribution을 다시 측정한
뒤 내려야 한다.

## 8. Post-training serving

LoRA adapter는 `docker-compose.lora.yaml` override로 vLLM에 mount한다. 이 override는
base model `NousResearch/Meta-Llama-3-8B-Instruct` 위에 다음 두 adapter를 노출한다.

| vLLM model id | adapter path |
| --- | --- |
| `nano-memgpt-llama3-r8` | `outputs/lora_student_r8/final_adapter/` |
| `nano-memgpt-llama3-r16` | `outputs/lora_student_r16/final_adapter/` |

실행 명령은 다음과 같다.

```bash
STUDENT_MODEL_ID=NousResearch/Meta-Llama-3-8B-Instruct \
STUDENT_TOOL_CALL_PARSER=nano_strict_llama \
docker compose -f docker-compose.yaml -f docker-compose.lora.yaml \
  --profile llama up -d --force-recreate llama-vllm
```

Letta가 adapter model handle을 사용할 수 있도록 provider registry도 갱신한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/register_vllm_lora_provider_models.py
```

post-training parser는 학습된 explicit JSON 표면 형식을 OpenAI `tool_calls`로 바꾸는
deterministic adapter다. 현재 `nano_strict_llama`는 다음 정규화를 수행한다.

| 패턴 | 정규화 |
| --- | --- |
| top-level `request_heartbeat` | tool `arguments.request_heartbeat`로 이동 |
| top-level `thinking` 또는 `reason` | tool `arguments.thinking`으로 이동 |
| missing non-`send_message` heartbeat | `request_heartbeat=true`로 기본화 |
| bare `{"message": ...}` | `send_message` tool call로 감쌈 |

이 parser는 모델이 생성한 `name`, `arguments`, `message` 필드를 이용하는 deterministic
adapter이며, 검색 query나 최종 답변 내용을 새로 선택하지 않는다. 따라서 post-training
보고서에서는 `parser_lenient_v3` 조건으로 명시한다.

## 9. Post-training DMR 결과

`r=8`과 `r=16` adapter를 모두 `parser_lenient_v3` 조건으로 vLLM/Letta에 연결한 뒤,
Experiment 1과 동일한 500-row DMR protocol을 실행했다.

| 조건 | 정상 loop | behavioral failure | ROUGE-L recall | containment | search rate | GPT-4.1 judge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA `r=8` | `489/500` | `11` | `0.5489` | `0.2474` | `0.7280` | `0.4765` |
| LoRA `r=16` | `497/500` | `3` | `0.5515` | `0.2656` | `0.7948` | `0.4809` |

두 adapter의 semantic judge accuracy는 거의 비슷하다. 그러나 operational metric에서는
`r=16`이 더 안정적이다. `r=16`은 format failure가 더 적고, `conversation_search`
호출률과 deterministic containment가 모두 높다. 따라서 현재 결과는 rank capacity가
최종 답변 의미 판정보다는 tool-use stability와 retrieval action frequency에 더 직접적인
영향을 준다는 쪽으로 해석하는 것이 안전하다.

## 10. Query-chain + evidence-answer SFT 실패 실험

failure audit 이후 query selection 병목을 더 직접적으로 겨냥하기 위해, 기존 full
teacher trajectory SFT를 세 부분으로 다시 export했다.

| Dataset | Source | Target | Step 수 |
| --- | --- | --- | ---: |
| Query-chain SFT | teacher의 search chain context | `conversation_search` call | `1,180` |
| Evidence-grounded answer SFT | probe + teacher search evidence | final answer text | `398` |
| Query + answer combined SFT | 위 두 dataset 결합 | search call 또는 answer text | `1,578` |

생성 artifact는 다음과 같다.

```text
data/trajectories/gpt41_paper_substring_scaled_query_chain_sft.jsonl
data/trajectories/gpt41_paper_substring_scaled_evidence_answer_sft.jsonl
data/trajectories/gpt41_paper_substring_scaled_query_answer_sft.jsonl
```

combined dataset은 `max_length=8192` 기준 2개 long query step을 제외하고
`1,576`개 record로 학습되었다.

| 조건 | 값 |
| --- | ---: |
| adapter | `outputs/lora_query_answer_r16/final_adapter/` |
| train records | `1,502` |
| eval records | `74` |
| optimizer step | `282/282` |
| epoch | `3` |
| train runtime | 약 `1시간 54분 30초` |
| train loss | `0.9346` |
| final eval loss | `0.8583` |
| final eval mean token accuracy | `0.7859` |

표면적인 SFT proxy metric은 기존 r16보다 좋아졌다. 기존 full-trajectory r16의 final eval
loss는 `0.8694`, token accuracy는 `0.7599`였고, query+answer r16은 eval loss `0.8583`,
token accuracy `0.7859`를 기록했다.

그러나 end-to-end Letta loop smoke에서는 크게 실패했다. vLLM에는
`nano-memgpt-llama3-query-answer-r16`로 정상 노출되었고, Letta provider에도
`vllm/nano-memgpt-llama3-query-answer-r16`로 등록되었지만, DMR 실행 초반
`35`개 row 중 `34`개가 `No tool calls found in response, model must make a tool call`
에러로 종료되었다.

| Metric | Query+answer r16 partial DMR |
| --- | ---: |
| attempted rows | `35` |
| completed rows | `1` |
| behavioral failures | `34` |
| tool-call format failures | `34` |
| containment | `0.0` |
| search rate among completed rows | `1.0` |

따라서 이 조건은 풀 500-row 평가를 중단했다. 이 결과는 중요한 negative result다.
query/answer target을 한 adapter에 섞으면 token-level 학습은 좋아질 수 있지만,
Letta가 매 step 요구하는 structured tool-call contract는 오히려 무너질 수 있다.

다음 학습에서는 search policy와 final answer policy를 같은 generation surface에 섞지
않고, 아래처럼 분리하는 것이 더 타당하다.

| 다음 후보 | 이유 |
| --- | --- |
| Query-only LoRA | `conversation_search` call만 target으로 두어 tool-call shell 안정성을 보존 |
| Answer-only adapter 또는 prompt | evidence가 주어진 후 `send_message`/assistant answer만 따로 학습 |
| Tool-call skeleton constrained decoding | 모델은 query string만 채우고 `name`, `arguments`, heartbeat는 deterministic wrapper가 고정 |
| Multi-stage adapter routing | search step에는 query adapter, answer step에는 answer adapter를 선택적으로 적용 |

## 11. Post-training DMR 상세 결과

### 11.1 r16 상세 결과

| 항목 | 값 |
| --- | ---: |
| 평가 행 | `500` |
| 정상 loop 완료 | `497` |
| behavioral failure | `3` |
| infrastructure error | `0` |
| ROUGE-L recall | `0.5515` |
| deterministic containment | `0.2656` |
| `conversation_search` 호출률 | `0.7948` |
| GPT-4.1 judge accuracy | `0.4809` |

실패 후보 분포는 다음과 같다.

| 후보 | 건수 |
| --- | ---: |
| `incorrect_answer` | `365` |
| `retrieval_miss_candidate` | `102` |
| `retrieval_hallucination_candidate` | `9` |
| `tool_call_format_failure` | `3` |

이 결과는 학습 전 strict-template Llama baseline의 500-row DMR 결과와 비교하면 loop
stability가 개선되었음을 보여 준다. strict-template baseline은 `481/500` loop 완료,
`19`건의 tool-call format failure, deterministic containment `0.1954`였다. 반면 r16
LoRA는 `497/500` loop 완료, `3`건의 format failure, containment `0.2656`을 보였다.

GPT-4.1 judge는 deterministic containment보다 관대한 semantic 판정이다. 따라서
`0.4809`를 containment 수치와 직접 같은 metric으로 비교하면 안 된다. 다만 lexical
match가 낮게 보이는 답변 중 상당수가 의미적으로는 정답으로 인정된다는 점은 분명하다.
보조 sanity check로 `gpt-4-turbo` judge도 실행했으며 `0.4789`로 GPT-4.1 판정과 거의
같았다.

현재 핵심 artifact는 다음 경로다.

```text
data/evaluation/post_lora_dmr_r16_lenient_v3/
logs/post_lora_dmr_r16_lenient_v3.log
logs/post_lora_dmr_r16_lenient_v3_judge_gpt41.log
```

주요 파일은 다음과 같다.

```text
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.jsonl
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.summary.json
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.judged.gpt41.jsonl
data/evaluation/post_lora_dmr_r16_lenient_v3/vllm-nano-memgpt-llama3-r16-offset-0-limit-500.judged.gpt41.summary.json
```

해석은 두 단계로 나누는 것이 안전하다.

1. `r=16` LoRA는 small model의 MemGPT tool-use surface와 loop stability를 상당히
   복구했다.
2. 아직 retrieval query 품질, retrieved evidence grounding, 최종 답변 압축에서는
   오류가 많이 남아 있다.

### 11.2 r8 상세 결과

`r=8` adapter는 같은 protocol에서 다음 결과를 보였다.

| 항목 | 값 |
| --- | ---: |
| 평가 행 | `500` |
| 정상 loop 완료 | `489` |
| behavioral failure | `11` |
| infrastructure error | `0` |
| ROUGE-L recall | `0.5489` |
| deterministic containment | `0.2474` |
| `conversation_search` 호출률 | `0.7280` |
| GPT-4.1 judge accuracy | `0.4765` |

실패 후보 분포는 다음과 같다.

| 후보 | 건수 |
| --- | ---: |
| `incorrect_answer` | `368` |
| `retrieval_miss_candidate` | `133` |
| `retrieval_hallucination_candidate` | `10` |
| `tool_call_format_failure` | `11` |

주요 artifact는 다음 경로다.

```text
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.jsonl
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.summary.json
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.judged.gpt41.jsonl
data/evaluation/post_lora_dmr_r8_lenient_v3/vllm-nano-memgpt-llama3-r8-offset-0-limit-500.judged.gpt41.summary.json
logs/post_lora_dmr_r8_lenient_v3.log
logs/post_lora_dmr_r8_lenient_v3_judge_gpt41.log
```

## 12. Query-only SFT 진단 결과

위 negative result를 바탕으로 answer target을 제거하고 `conversation_search` target만 남긴
query-only LoRA를 학습했다. 자세한 분석은 `docs/query_only_lora_report.md`에 따로 정리했다.

| 항목 | 값 |
| --- | ---: |
| dataset | `data/trajectories/gpt41_paper_substring_scaled_query_chain_sft.jsonl` |
| adapter | `outputs/lora_query_only_r16/final_adapter/` |
| train records | `1,125` |
| eval records | `53` |
| dropped overlength | `2` |
| optimizer step | `213/213` |
| train runtime | 약 `1시간 45분 59초` |
| train loss | `0.9618` |
| final eval loss | `0.8244` |
| final token accuracy | `0.7703` |

Proxy metric은 기존 full-trajectory r16과 query+answer r16보다 좋았다. 그러나 20-row DMR
smoke에서는 첫 5개 row가 모두 behavioral failure로 종료되어 full evaluation을 중단했다.

| Metric | Query-only r16 smoke |
| --- | ---: |
| attempted rows | `5` |
| completed rows | `0` |
| behavioral failures | `5` |
| tool-call format failures | `5` |
| mean provider attempts per behavioral failure | `14.4` |

중요한 점은 이 실패가 완전한 no-tool failure는 아니라는 것이다. Raw provider response에서
각 row는 초반에 여러 번 정상 `tool_calls` 채널로 `conversation_search`를 생성했다. 예를
들어 5개 실패 row의 valid tool-call message 수는 각각 `9`, `21`, `10`, `10`, `7`이었다.
하지만 multi-turn 검색이 길어지면서 같은 tool-call JSON을 assistant `content` 문자열로
출력했고, 이때 provider response는 `tool_calls=[]`, `finish_reason="stop"`이 되었다. Letta는
tool call이 필요한 step에서 이를 `No tool calls found in response`로 거절했다.

따라서 query-only 결과는 다음처럼 해석한다.

```text
query-only SFT는 tool-call intent를 일부 강화하지만,
OpenAI tool_calls 채널을 multi-turn loop 전체에서 안정적으로 유지하지 못한다.
```

다음 실험은 full query-only DMR이 아니라 parser-rescue 또는 deterministic tool skeleton
진단이 우선이다. Assistant `content` 안의 schema-valid JSON을 `tool_calls`로 변환했을 때
성능이 회복되는지 보면, 남은 병목이 channel/transport 문제인지 query content 문제인지 더
깨끗하게 분리할 수 있다.

이후 vLLM parser layer에 `nano_rescue_llama`를 추가해 6-row smoke를 실행했다. 결과는
`1/6` loop completion, `5/6` tool-call format failure였다. 따라서 vLLM parser-rescue만으로는
불충분하며, 다음 진단은 Letta-side 또는 endpoint-proxy rescue가 더 직접적이다.

Endpoint proxy-rescue도 실행했다. `scripts/vllm_tool_rescue_proxy.py`는 vLLM과 Letta 사이에서
assistant `content` JSON을 `tool_calls`로 변환한다. Proxy 로그에는 `rescued tool call` event가
`70`회 기록되었지만, 20-row smoke 결과는 `2/20` loop completion, `18/20` format failure,
containment `0.0`이었다. 완료된 두 row도 정답이 아니라 planning leakage였다.

따라서 query-only adapter는 end-to-end agent로는 부적합하다. 이후 방향은 query-only를
search phase 전용으로 쓰고, final answer phase는 full LoRA/base/answer adapter로 라우팅하는
구조다.

## 13. Phase-routed query-only 진단

위 방향을 diagnostic으로 검증했다. Letta loop 대신 deterministic controller가 search phase와
answer phase를 나누었다.

| 항목 | 값 |
| --- | --- |
| Search phase | `nano-memgpt-llama3-query-only-r16` |
| Answer phase | `nano-memgpt-llama3-r16` |
| Search execution | local substring recall |
| Dataset | MSC DMR offset `0`, limit `100` |
| Max searches | `3` |
| Answer context | retrieved evidence only |

결과:

| Metric | Value |
| --- | ---: |
| completed | `100/100` |
| answer containment | `0.11` |
| retrieved-reference rate | `0.09` |
| mean searches | `2.22` |
| mean retrieved messages | `1.10` |
| rows with zero retrieved messages | `64/100` |
| `UNKNOWN` answers | `67/100` |

해석은 분명하다. Phase routing은 query-only adapter의 stop-and-answer failure를 제거하지만,
검색어 자체가 answer-bearing utterance를 찾는 능력은 낮다. 따라서 다음 학습은 query-call JSON
전체를 더 모방하는 방향보다, query string이 실제 reference-containing message를 retrieve하는지
직접 평가하고 최적화하는 방향으로 가야 한다.

자세한 분석은 `docs/phase_routed_dmr_report.md`에 정리했다.

## 14. Deterministic query skeleton 결과

Tool-call shell을 완전히 고정하고 모델이 query string만 생성하는 조건도 실행했다.

| Query generator | Rows | Completion | Retrieved-reference rate | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query-only r16, tool-call phase-routed | `100` | `100/100` | `0.09` | `0.11` | `1.10` |
| Query-only r16, skeleton | `100` | `100/100` | `0.23` | `0.23` | `3.70` |
| Full trajectory r16, skeleton | `100` | `100/100` | `0.16` | `0.17` | `2.86` |
| Base Llama skeleton smoke | `20` | `20/20` | `0.20` | `0.20` | `1.75` |

이 결과는 query-only LoRA를 다르게 해석하게 만든다. End-to-end agent로는 실패했지만, query
string만 생성하게 하면 full trajectory r16보다 좋은 retrieval hit를 보인다. 따라서 query-only
SFT에는 query-policy signal이 들어 있으며, 앞으로는 이 adapter를 constrained query generator로
다루는 것이 맞다.

다음 학습 방향은 query-call JSON imitation이 아니라 retrieval-supervised query objective다.
예를 들어 query가 실제 reference-containing message를 retrieve하면 positive, distractor만
retrieve하거나 결과가 없으면 negative로 두는 pairwise/ranking objective를 고려한다.

자세한 분석은 `docs/query_skeleton_dmr_report.md`에 정리했다.

## 15. Teacher query skeleton replay 결과

Query-only skeleton이 어느 정도 의미 있는지 보기 위해, GPT-4.1 teacher가 실제로 사용한 query도
같은 deterministic skeleton으로 replay했다. 이 조건은 teacher query가 agent loop를 통과한다는
뜻이 아니라, query string 자체가 substring recall에서 reference-bearing message를 얼마나 잘
찾는지 보는 upper reference다.

| Query source | Subset | Rows | Retrieved-reference rate | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher max-3 query | approved | `398` | `0.367` | `0.342` | `3.25` |
| Teacher max-3 query | teacher-search only | `302` | `0.483` | `0.450` | `4.28` |
| Query-only r16 skeleton | same approved | `398` | `0.276` | `0.249` | `3.76` |
| Query-only r16 skeleton | teacher-search subset | `302` | `0.272` | `0.238` | `3.76` |

이 결과는 query-only LoRA에 signal이 있지만 아직 teacher query 수준은 아니라는 점을 보여 준다.
Approved subset에서는 teacher max-3 containment의 약 `73%`까지 따라오지만, teacher-search subset
기준으로는 약 `53%`에 머문다. 또한 query-only는 더 많은 message를 retrieve하면서도 reference
hit가 낮으므로, 단순 recall 폭 증가보다 distractor를 줄이는 query objective가 필요하다.

다음 학습은 `query text imitation`만으로 보기보다 retrieval-supervised query objective로 설계하는
것이 더 타당하다. 예를 들어 teacher query와 query-only query를 row-level로 비교해
`teacher-correct / student-wrong` row를 hard example로 만들고, reference-containing message를
retrieve하는 query를 positive로 두는 ranking loss를 고려할 수 있다.

자세한 분석은 `docs/teacher_query_skeleton_report.md`에 정리했다.

## 16. Query preference dataset export

Teacher max-3 query skeleton과 query-only r16 skeleton의 row-level gap에서 hard-negative query
dataset을 export했다. 자세한 분석은 `docs/query_preference_dataset_report.md`에 정리했다.

기본 export 명령:

```bash
python3 scripts/export_query_preference_dataset.py
```

기본 조건은 teacher-search subset 중 `teacher_only` retrieval row만 사용한다. Teacher trace에서
reference-containing message를 처음 retrieve한 query를 `chosen`으로 두고, query-only r16의
non-hit query를 `rejected`로 둔다. 가능하면 result가 0개인 rejected query를 우선 선택한다.

| Output | Records | Purpose |
| --- | ---: | --- |
| `data/trajectories/query_hard_negative_preferences.jsonl` | `72` | DPO/ORPO/ranking-style preference 학습 |
| `data/trajectories/query_hard_positive_sft.jsonl` | `72` | 기존 LoRA SFT trainer로 positive query만 학습 |
| `data/trajectories/query_hard_negative_preferences_zero_only.jsonl` | `51` | 더 깨끗한 zero-result-only negative preference subset |
| `data/trajectories/query_hard_positive_sft_zero_only.jsonl` | `51` | zero-result-only subset의 positive SFT |

SFT trainer prepare check:

```bash
docker compose exec -T nano-memgpt-dev python scripts/train_lora_sft.py \
  --trajectories data/trajectories/query_hard_positive_sft.jsonl \
  --output-dir outputs/query_hard_positive_sft_prepare_check \
  --max-length 4096 \
  --rank 16 \
  --alpha 32 \
  --prepare-only
```

결과는 `72` records, train/eval split `68/4`, overlength drop `0`이다. 따라서 기존 SFT trainer로
바로 읽을 수 있다.

다음 pilot 후보:

```bash
docker compose exec -T nano-memgpt-dev python scripts/train_lora_sft.py \
  --trajectories data/trajectories/query_hard_positive_sft.jsonl \
  --output-dir outputs/lora_query_hard_positive_r16 \
  --max-length 4096 \
  --rank 16 \
  --alpha 32 \
  --learning-rate 1.0e-5 \
  --epochs 1.0 \
  --gradient-accumulation-steps 8
```

Dataset이 작기 때문에 이 실험은 성능 개선을 보장하는 full training이라기보다, hard-positive query
SFT가 skeleton retrieval hit를 올리는지 보는 작은 pilot으로 해석해야 한다.

## 17. Hard-positive query SFT pilot 결과

위 pilot을 실행했다. 자세한 결과는 `docs/query_hard_positive_lora_report.md`에 정리했다.

학습 결과:

| Item | Value |
| --- | ---: |
| train steps | `9` |
| train loss | `3.622` |
| eval loss | `3.899` |
| eval mean token accuracy | `0.2917` |

Skeleton raw500 결과:

| Query generator | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only r16 skeleton | `0.244` | `0.216` | `3.744` |
| Hard-positive r16 skeleton | `0.184` | `0.180` | `2.882` |

학습 hard set 72개 내부에서는 약간의 개선이 있었다.

| Metric on 72 hard rows | Query-only r16 | Hard-positive r16 |
| --- | ---: | ---: |
| Retrieved-reference | `0/72` | `7/72` |
| Containment | `1/72` | `7/72` |

따라서 이 pilot은 전체적으로 negative result다. Positive-only SFT는 hard row 일부를 맞히게
만들지만, 전체 query distribution을 너무 좁혀 no-result/UNKNOWN을 늘린다.

## 12. Query Preference DPO Pilot

Positive-only SFT의 부작용을 확인한 뒤, `query_hard_negative_preferences_zero_only.jsonl`로 DPO
pilot을 실행했다. 이 dataset은 teacher query가 reference retrieval에 성공하고, student query는
zero-result로 실패한 `51`개 preference pair만 포함한다.

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
| preference records | `51` |
| train/eval split | `47/4` |
| train steps | `6` |
| train loss | `0.691` |
| eval loss | `0.689` |
| eval reward accuracy | `0.75` |

Evaluation summary:

| Query generator | Raw500 retrieved-reference | Raw500 containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only r16 skeleton | `0.244` | `0.216` | `3.744` |
| Hard-positive r16 skeleton | `0.184` | `0.180` | `2.882` |
| Preference zero-DPO r16 skeleton | `0.166` | `0.158` | `2.564` |

Hard set 내부에서는 작은 회복이 있었다.

| Metric on 72 hard rows | Query-only r16 | Hard-positive r16 | Preference zero-DPO r16 |
| --- | ---: | ---: | ---: |
| Retrieved-reference | `0/72` | `7/72` | `8/72` |
| Containment | `1/72` | `7/72` | `7/72` |

결론은 hard-positive SFT와 같다. Teacher query signal은 분명히 있지만, small hard-set LoRA로는
일반 query policy를 개선하지 못한다. 자세한 결과는
`docs/query_preference_dpo_report.md`에 정리했다.

다음 단계는 query dataset을 더 오래 학습하는 것이 아니라, candidate query generation + local
retrieval reranking으로 넘어가는 것이다. 모델이 여러 후보를 만들고, 실제 retrieval 결과를 보고
선택하게 해야 한다.

후속 candidate reranking 실험은 이 해석을 지지했다. 단순 result-count reranker는 raw500에서는
query-only보다 낮았지만 hard72를 `20/72` retrieval까지 복구했다. 이후 lexical evidence overlap과
broad-query penalty를 넣은 target-5 reranker는 raw500 retrieved-reference/containment를
`0.246/0.224`까지 올려 query-only skeleton `0.244/0.216`을 소폭 넘었고, hard72에서는
retrieved-reference `25/72`, containment `22/72`를 기록했다. 자세한 결과는
`docs/query_candidate_rerank_report.md`에 정리했다.

마지막 evidence filtering diagnostic에서는 lexical target-5의 mean retrieved를 `4.742`에서
`3.426`으로 줄이면서 raw500 containment `0.224`를 유지했다. 이 결과는
`docs/evidence_filter_report.md`에 정리했다. 따라서 v1에서는 추가 LoRA 학습을 중단하고,
paper writing과 example analysis로 넘어간다.
