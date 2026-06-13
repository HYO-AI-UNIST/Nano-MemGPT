# Nano-MemGPT 연구 문서 안내

이 디렉터리는 제안서
**Diagnosing and Recovering MemGPT's Function-Calling Failures in Small Open-Source LLMs
via Knowledge Distillation**의 재현 실험, 진단 결과, 실행 절차를 기록한다.

원본 제안서는 [`customproposal.pdf`](customproposal.pdf)이며, Markdown 문서는 실험을
진행하면서 확인된 구현 세부사항과 재현 조건을 보완한다. 특히 원 논문의 공개 코드와 현재
유지보수 중인 Letta 사이에는 도구 schema와 실행 경로 차이가 있으므로, 결과를 읽을 때
`paper-faithful`, `local proxy`, `diagnostic upper bound` 조건을 구분해야 한다.

## 권장 읽기 순서

| 순서 | 문서 | 목적 |
| ---: | --- | --- |
| 1 | [`proposal_summary.md`](proposal_summary.md) | 연구 질문, 중심 가설, 평가 축을 빠르게 파악한다. |
| 2 | [`research_plan.md`](research_plan.md) | 전체 연구 단계와 산출물, 현재 진행 상태를 확인한다. |
| 3 | [`vanilla_dmr_protocol.md`](vanilla_dmr_protocol.md) | DMR 데이터 적재, recall 검색, metric 정의를 재현한다. |
| 4 | [`vanilla_memgpt_pilot.md`](vanilla_memgpt_pilot.md) | 초기 tool-call compatibility gate가 왜 필요했는지 이해한다. |
| 5 | [`experiment_1_report.md`](experiment_1_report.md) | Vanilla 및 strict-template baseline 결과를 확인한다. |
| 6 | [`oracle_experiment.md`](oracle_experiment.md) | GPT-4.1 teacher trajectory 수집과 Oracle replay를 실행한다. |
| 7 | [`oracle_experiment_report.md`](oracle_experiment_report.md) | Oracle pilot 결과와 현재 해석을 읽는다. |
| 8 | [`document_qa_proxy.md`](document_qa_proxy.md) | 20M Wikipedia index 부재 시 사용하는 Document-QA proxy의 한계를 확인한다. |
| 9 | [`lora_training.md`](lora_training.md) | 승인 trajectory를 이용한 LoRA distillation 실행 절차를 확인한다. |
| 10 | [`post_lora_evaluation_report.md`](post_lora_evaluation_report.md) | LoRA adapter의 end-to-end DMR 평가 결과를 확인한다. |
| 11 | [`failure_audit_report.md`](failure_audit_report.md) | LoRA 이후 남은 정확도 병목을 query/evidence/answer 단계로 분해한다. |
| 12 | [`teacher_evidence_ablation_report.md`](teacher_evidence_ablation_report.md) | teacher evidence가 주어졌을 때 LoRA adapter가 얼마나 회복되는지 확인한다. |
| 13 | [`teacher_query_ablation_report.md`](teacher_query_ablation_report.md) | teacher query chain hint만으로 end-to-end loop가 얼마나 회복되는지 확인한다. |
| 14 | [`query_only_lora_report.md`](query_only_lora_report.md) | query-only LoRA가 왜 end-to-end agent로 실패했는지 확인한다. |
| 15 | [`phase_routed_dmr_report.md`](phase_routed_dmr_report.md) | search/answer phase를 분리했을 때 query-only adapter의 한계를 확인한다. |
| 16 | [`query_skeleton_dmr_report.md`](query_skeleton_dmr_report.md) | tool-call shell을 고정하고 query string만 생성하게 했을 때의 회복 폭을 확인한다. |
| 17 | [`teacher_query_skeleton_report.md`](teacher_query_skeleton_report.md) | teacher query 자체를 같은 skeleton으로 replay해 query upper reference를 확인한다. |
| 18 | [`query_skeleton_gap_report.md`](query_skeleton_gap_report.md) | teacher max-3 query와 query-only query의 row-level gap을 확인한다. |
| 19 | [`query_preference_dataset_report.md`](query_preference_dataset_report.md) | teacher-only gap row로 만든 hard-negative/query-positive dataset을 확인한다. |
| 20 | [`query_hard_positive_lora_report.md`](query_hard_positive_lora_report.md) | hard-positive query SFT pilot의 negative result와 hard-set 내부 개선을 확인한다. |
| 21 | [`query_preference_dpo_report.md`](query_preference_dpo_report.md) | zero-result hard negative DPO pilot의 negative result와 hard-set 내부 회복을 확인한다. |
| 22 | [`query_candidate_rerank_report.md`](query_candidate_rerank_report.md) | candidate query generation + local reranking의 raw500 및 hard-set 복구 결과를 확인한다. |
| 23 | [`evidence_filter_report.md`](evidence_filter_report.md) | 마지막 evidence filtering 실험과 v1 실험 freeze 기준을 확인한다. |
| 24 | [`archive/final_paper_draft.md`](archive/final_paper_draft.md) | LaTeX 논문 이전의 기존 논문형 draft를 참고한다. |
| 25 | [`archive/final_paper_korean_full.md`](archive/final_paper_korean_full.md) | 모든 실험을 초심자도 따라갈 수 있게 풀어 쓴 v1 통합본을 참고한다. |
| 26 | [`latex/final_paper.tex`](latex/final_paper.tex) | ACL 형식 최종 논문 LaTeX 초안을 확인한다. |
| 27 | [`artifact_policy.md`](artifact_policy.md) | GitHub 공개 시 추적할 파일과 제외할 산출물을 확인한다. |

## 실험 조건 용어

| 용어 | 의미 |
| --- | --- |
| `Raw Vanilla` | 모델 가중치, prompt adapter, parser adapter를 추가하지 않은 원시 serving 조건 |
| `Strict Template Adapter` | 가중치는 고정하고, 명시적인 단일 schema-valid tool call만 OpenAI `tool_calls` 형식으로 변환하는 조건 |
| `Full-History Upper Bound` | retrieval을 제거하고 과거 대화 전체를 student context에 직접 주입하는 진단 조건 |
| `Teacher-Trace Oracle` | 승인된 GPT teacher의 tool call과 tool output을 student에게 evidence로 주고 최종 답변만 생성시키는 조건 |
| `Paper-Substring Contract` | paper-era DMR recall semantics에 맞춘 case-insensitive substring 검색 계약 |
| `Document-QA Proxy` | paper의 20M-passage archival retrieval을 대체하지 않는 로컬 context-pack 진단 조건 |

## 현재 핵심 상태

2026-06-05 기준으로 corrected GPT-4.1 20-row pilot은 judge accuracy `17/20`
(`0.8500`)을 기록했다. 승인된 17개 teacher trace를 replay한 결과는 Llama-3-8B
`14/17` (`0.8235`), Mistral-7B `15/17` (`0.8824`)이다.

500-row GPT-4.1 trajectory 수집, judge 필터링, 두 student의 scaled Teacher-Trace
Oracle replay가 완료되었다.

| 항목 | 결과 |
| --- | ---: |
| GPT-4.1 teacher judge accuracy | `398/500` (`0.7960`) |
| 승인 teacher trajectory | `398`행 |
| LoRA용 context-complete SFT step | `1,664`개 |
| Llama-3-8B scaled Oracle judge accuracy | `292/398` (`0.7337`) |
| Mistral-7B scaled Oracle judge accuracy | `349/398` (`0.8769`) |
| Llama-3-8B LoRA `r=8` final eval loss | `0.9009` |
| Llama-3-8B LoRA `r=8` final token accuracy | `0.7546` |
| Llama-3-8B LoRA `r=8` post-training loop completion | `489/500` |
| Llama-3-8B LoRA `r=8` post-training containment | `0.2474` |
| Llama-3-8B LoRA `r=8` post-training GPT-4.1 judge accuracy | `0.4765` |
| Llama-3-8B LoRA `r=16` final eval loss | `0.8694` |
| Llama-3-8B LoRA `r=16` final token accuracy | `0.7599` |
| Llama-3-8B LoRA `r=16` post-training loop completion | `497/500` |
| Llama-3-8B LoRA `r=16` post-training containment | `0.2656` |
| Llama-3-8B LoRA `r=16` post-training GPT-4.1 judge accuracy | `0.4809` |
| LoRA `r=8` + teacher trace GPT-4.1 judge accuracy | `343/398` (`0.8618`) |
| LoRA `r=16` + teacher trace GPT-4.1 judge accuracy | `345/398` (`0.8668`) |
| LoRA `r=16` + teacher-query hint search rate | `0.9594` |
| LoRA `r=16` + teacher-query hint GPT-4.1 judge accuracy | `248/394` (`0.6294`) |
| Query-only r16 phase-routed retrieved-reference rate | `0.09` |
| Query-only r16 phase-routed containment | `0.11` |
| Query-only r16 deterministic skeleton retrieved-reference rate | `0.23` |
| Query-only r16 deterministic skeleton containment | `0.23` |
| Query-only r16 deterministic skeleton raw-500 retrieved-reference rate | `0.244` |
| Query-only r16 deterministic skeleton raw-500 containment | `0.216` |
| Full trajectory r16 deterministic skeleton retrieved-reference rate | `0.16` |
| Full trajectory r16 deterministic skeleton containment | `0.17` |
| Teacher max-3 query skeleton approved retrieved-reference rate | `0.367` |
| Teacher max-3 query skeleton approved containment | `0.342` |
| Teacher max-3 query skeleton teacher-search retrieved-reference rate | `0.483` |
| Teacher max-3 query skeleton teacher-search containment | `0.450` |
| Hard-positive query SFT raw-500 retrieved-reference rate | `0.184` |
| Hard-positive query SFT raw-500 containment | `0.180` |
| Preference zero-DPO query raw-500 retrieved-reference rate | `0.166` |
| Preference zero-DPO query raw-500 containment | `0.158` |
| Candidate lexical rerank query raw-500 retrieved-reference rate | `0.246` |
| Candidate lexical rerank query raw-500 containment | `0.224` |
| Candidate lexical rerank query hard72 retrieved-reference rate | `0.347` |
| Candidate lexical rerank query hard72 containment | `0.306` |
| Evidence filter top-6 raw-500 retrieved-reference rate | `0.234` |
| Evidence filter top-6 raw-500 containment | `0.224` |
| Evidence filter top-6 raw-500 mean retrieved | `3.426` |
| Evidence filter top-6 hard72 containment | `0.306` |
| Query skeleton gap teacher-search teacher-only retrieval rows | `72/302` |
| Query skeleton gap teacher-search student-only retrieval rows | `8/302` |
| Query hard-negative preference records | `72` |
| Query zero-result-only clean preference records | `51` |
| Query hard-positive SFT prepare check | `72` records, `0` overlength drop |
| Query hard-positive r16 skeleton raw-500 retrieved-reference rate | `0.184` |
| Query hard-positive r16 skeleton raw-500 containment | `0.180` |
| Query hard-positive r16 hard-set retrieval gain | `7/72` |

원본 raw trajectory와 승인 데이터셋은 다음 명령으로 행 수를 확인할 수 있다.

```bash
wc -l data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.jsonl
wc -l data/trajectories/gpt41_paper_substring_scaled_approved_sft.jsonl
```

현재 vLLM serving은 LoRA adapter mount 조건으로 `nano-memgpt-llama3-r8`와
`nano-memgpt-llama3-r16`을 노출한다. 기본 base model도 함께 노출되어 있다.
