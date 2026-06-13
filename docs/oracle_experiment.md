# Oracle 실험 실행 가이드

## 1. 목적

Oracle 실험은 소형 모델의 실패 원인을 더 직접적으로 분해한다. 강한 teacher가 동일한
MemGPT DMR loop에서 수행한 검색 호출과 검색 결과를 기록한 뒤, frozen student에게
teacher가 회수한 근거를 제공한다. student는 자체 도구 호출 없이 최종 답변만 만든다.

이 조건은 다음 질문을 답한다.

> 소형 모델이 검색과 chaining을 대신 제공받았을 때 정답률이 얼마나 회복되는가?

점수가 크게 회복되면 병목의 상당 부분은 지식 부족보다 검색 행동과 tool chaining에
있다. 여전히 실패하는 행은 answer construction 또는 teacher evidence 자체의 한계를
확인해야 한다.

## 2. 두 종류의 Oracle

| 조건 | student에게 제공하는 정보 | 목적 | 지위 |
| --- | --- | --- | --- |
| `teacher_trace` | teacher가 실제 회수한 tool output | 검색 행동을 제거한 주요 Oracle | 제안서 핵심 조건 |
| `full_history` | MSC 세션 1-5 전체 원문 | 검색 자체를 제거한 상한 진단 | 보조 upper bound |

`full_history`는 빠르고 API key가 필요 없지만 제안서의 핵심 Oracle을 대체하지 않는다.
실제 MemGPT loop에서 teacher가 어떤 근거를 찾았는지는 `teacher_trace`로 평가해야 한다.

## 3. Teacher 모델 provenance

[원본 MemGPT 논문](https://arxiv.org/abs/2310.08560)은 GPT-4 Turbo 결과에
`gpt-4-1106-preview`를 사용했다. OpenAI
[deprecation log](https://platform.openai.com/docs/deprecations)에 따르면 해당 endpoint는
2026-03-26 종료되어 동일 snapshot의 직접 재현은 불가능하다.

이 프로젝트는 고정된 현대 teacher snapshot을 사용한다.

| 항목 | 값 |
| --- | --- |
| 환경 변수 | `OPENAI_TEACHER_MODEL=gpt-4.1-2025-04-14` |
| Letta model id | `openai/gpt-4.1-2025-04-14` |
| 원 논문 대비 차이 | retired GPT-4 Turbo 대신 GPT-4.1 snapshot 사용 |

`.env` 예시는 다음과 같다. API key 값은 문서나 로그에 기록하지 않는다.

```bash
OPENAI_API_KEY=...
OPENAI_TEACHER_MODEL=gpt-4.1-2025-04-14
OPENAI_JUDGE_MODEL=gpt-4.1-2025-04-14
```

## 4. 중요한 실행 주의사항

현재 scaled teacher collection이 실행 중인 동안에는
`nano-memgpt-dev` container를 recreate하지 않는다. `.env`를 바꾼 뒤 container에
반영하려면 일반적으로 다음 명령이 필요하지만, 장기 수집 중에는 실행하지 않는다.

```bash
docker compose up -d --force-recreate nano-memgpt-dev letta-server
```

장기 수집이 끝난 뒤 teacher preflight를 다시 확인하려면 다음을 실행한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/check_openai_teacher_preflight.py

docker compose exec -T nano-memgpt-dev \
  curl -fsS http://letta-server:8283/v1/models/
```

## 5. Paper-faithful recall 계약

초기 GPT-4.1 pilot 감사에서 maintained Letta의 도구 설명과 로컬 PostgreSQL 실행 경로가
다름을 확인했다.

| 층 | 초기 상태 |
| --- | --- |
| 모델에게 보인 설명 | text와 semantic similarity를 함께 사용하는 hybrid search처럼 읽힘 |
| 로컬 PostgreSQL 경로 | 대소문자 무시 SQL substring matching |
| 논문 시기 기본 DMR recall | 대소문자 무시 string matching |

현재 Docker 설정은 좁은 docstring override를 mount하고 harness가
`paper_substring` metadata를 적용한다. SQL 구현을 바꾸는 것이 아니라 모델에게 실제
실행 계약을 정확히 알려 준다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/configure_dmr_recall_contract.py --contract paper_substring
```

## 6. Teacher trajectory 수집

### 6.1 한 행 smoke

유료 scaled collection 전에는 한 행 smoke를 수행한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/eval_vanilla_dmr.py \
  --model openai/gpt-4.1-2025-04-14 \
  --model-source-note "OpenAI GPT-4.1 2025-04-14 modern teacher; original paper used retired gpt-4-1106-preview" \
  --output-dir data/evaluation/oracle_teacher_dmr_smoke \
  --limit 1 \
  --capture-provider-traces
```

### 6.2 20행 quality-gate pilot

신규 API account는 GPT-4.1 TPM limit가 낮을 수 있다. 한 DMR 행은 여러 chained call을
거치며 prompt token을 많이 소비하므로 현재는 worker 1개와 65초 cooldown을 사용한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/eval_vanilla_dmr.py \
  --model openai/gpt-4.1-2025-04-14 \
  --model-source-note "OpenAI GPT-4.1 2025-04-14 paper-substring contract pilot" \
  --output-dir data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_pilot \
  --limit 20 \
  --workers 1 \
  --max-steps 16 \
  --row-delay-seconds 65 \
  --infrastructure-retry-delay-seconds 65 \
  --capture-provider-traces \
  --resume
```

### 6.3 500행 scaled 수집

현재 실행 중인 scaled collection은 다음 조건이다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/eval_vanilla_dmr.py \
  --model openai/gpt-4.1-2025-04-14 \
  --model-source-note "OpenAI GPT-4.1 2025-04-14 mounted paper-substring contract scaled trajectory collection max_steps=16; original paper used retired gpt-4-1106-preview" \
  --output-dir data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled \
  --limit 500 \
  --workers 1 \
  --max-steps 16 \
  --row-delay-seconds 65 \
  --infrastructure-retry-delay-seconds 65 \
  --capture-provider-traces \
  --resume
```

`--max-steps 16`은 비용을 제한하기 위해 pilot에서 검증한 명시적 상한이다. harness의
기본값은 Letta와 같은 `50`이다. 예산이 허용되면 별도 sensitivity run에서 기본값
상한을 사용해 ceiling 차이를 확인한다.

진행 상황은 다음 명령으로 확인한다.

```bash
wc -l data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.jsonl
tail -n 20 logs/gpt41_paper_substring_scaled.log
```

## 7. Judge, 필터링, 데이터셋 export

teacher도 항상 정답을 내지는 않는다. 따라서 raw trajectory를 곧바로 Oracle replay나
LoRA 학습에 사용하지 않는다. 전체 수집 후 GPT-4.1 judge 판정을 수행하고 승인된 행만
두 형식으로 export한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/judge_dmr_answers.py \
  --input-jsonl data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.jsonl \
  --model gpt-4.1-2025-04-14 \
  --resume

docker compose exec -T nano-memgpt-dev \
  python scripts/filter_teacher_trajectories.py \
  --input-jsonl data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.jsonl \
  --judge-jsonl data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.judged.jsonl \
  --output-jsonl data/trajectories/gpt41_paper_substring_scaled_approved_rows.jsonl

docker compose exec -T nano-memgpt-dev \
  python scripts/extract_teacher_oracle_traces.py \
  --input-jsonl data/trajectories/gpt41_paper_substring_scaled_approved_rows.jsonl \
  --output-jsonl data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl

docker compose exec -T nano-memgpt-dev \
  python scripts/export_teacher_sft_trajectories.py \
  --input-jsonl data/trajectories/gpt41_paper_substring_scaled_approved_rows.jsonl \
  --output-jsonl data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl
```

| export | 용도 |
| --- | --- |
| `approved_rows.jsonl` | judge 통과 raw teacher 행 보존 |
| `approved_oracle.jsonl` | student answer-only replay용 teacher call/output sequence |
| `approved_sft.jsonl` | LoRA 학습용 context-complete trajectory step |

SFT step은 전체 provider request context, teacher function call, heartbeat 여부, 대응하는
function output을 보존한다.

## 8. Student teacher-trace replay

Llama replay 예시는 다음과 같다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/eval_dmr_oracle_replay.py \
  --model NousResearch/Meta-Llama-3-8B-Instruct \
  --oracle-mode teacher_trace \
  --teacher-traces data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl \
  --output-dir data/evaluation/oracle_dmr_teacher_trace \
  --limit 500 \
  --workers 8 \
  --resume
```

Mistral을 평가할 때는 serving model을 `mistralai/Mistral-7B-Instruct-v0.3`로 전환하고
같은 명령의 `--model`을 바꾼다. 두 replay가 끝나면 기본 개발 상태를 위해 Llama
serving으로 복구한다.

## 9. Full-history upper bound

전체 history를 직접 주입하는 보조 상한은 다음 명령으로 실행한다.

```bash
docker compose exec -T nano-memgpt-dev \
  python scripts/eval_dmr_oracle_replay.py \
  --model NousResearch/Meta-Llama-3-8B-Instruct \
  --oracle-mode full_history \
  --output-dir data/evaluation/oracle_dmr_full_history \
  --limit 500 \
  --workers 8 \
  --resume
```

## 10. 비용 계획

교정 pilot은 raw 20행에 `$0.8372`, 행당 평균 `$0.0419`를 사용했다.

| 목표 | 단순 선형 추정 | 주의점 |
| --- | ---: | --- |
| raw 500행 | 약 `$20.93` | retry와 judge 비용 제외 |
| 승인 SFT step 2,000개 | raw 약 580행 | pilot 분포 기준 |
| 승인 SFT step 2,500개 | raw 약 725행 | 분산 고려 필요 |
| 승인 SFT step 3,000개 | raw 약 870행 | 추가 예산 필요 |

## 11. 현재 상태

2026-06-01 기준 direct preflight와 smoke는 성공했다. 교정된 20행 pilot은 GPT-4.1 judge
`17/20`(`0.8500`)을 통과하고 context-complete SFT step 69개를 export했다. 승인된
17개 teacher trace의 replay는 Llama `14/17`(`0.8235`), Mistral
`15/17`(`0.8824`)이다.

500행 scaled collection과 후처리가 완료되었다.

| 항목 | 결과 |
| --- | ---: |
| raw teacher trajectory | `500`행 |
| GPT-4.1 judge 승인 | `398/500` (`0.7960`) |
| 승인 Oracle trace | `398`행 |
| 승인 SFT step | `1,664`개 |
| Llama scaled Teacher-Trace Oracle | `292/398` (`0.7337`) |
| Mistral scaled Teacher-Trace Oracle | `349/398` (`0.8769`) |

Oracle 평가가 끝난 뒤 vLLM serving은 기본 Llama-3-8B 상태로 복구했다. 다음 단계는
context-complete SFT step을 parser 계약에 맞게 검증하고 LoRA 학습 launcher를 확정하는
것이다.
