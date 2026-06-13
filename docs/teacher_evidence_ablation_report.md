# LoRA Teacher-Evidence Ablation Report

## 1. 목적

이 실험의 목적은 post-LoRA DMR 실패의 원인을 더 명확히 분리하는 것이다. End-to-end
LoRA 평가에서는 student가 직접 `conversation_search` query를 만들고, 검색 결과를 읽고,
최종 답변을 생성한다. 반면 이 ablation에서는 GPT-4.1 teacher가 실제로 호출했던 tool call과
tool output을 그대로 student prompt에 replay한다.

따라서 이 조건은 순수한 Teacher-query ablation은 아니다. Retrieval 실행까지 student loop에
맡기는 조건이 아니라, teacher가 얻은 evidence를 student에게 직접 제공하는 direct vLLM
answer-from-evidence 조건이다. 보고서에서는 이를 `LoRA Teacher-Trace Replay` 또는
`teacher-evidence ablation`이라고 부른다.

핵심 질문은 다음이다.

```text
LoRA student가 end-to-end에서는 틀리더라도,
teacher evidence가 주어지면 정답을 만들 수 있는가?
```

만약 이 조건에서 성능이 크게 회복되면, post-LoRA의 낮은 DMR 정확도는 answer-generation
능력 부족보다 query/evidence acquisition 병목에서 비롯되었다고 해석할 수 있다.

## 2. Protocol

| 항목 | 값 |
| --- | --- |
| Script | `scripts/eval_dmr_oracle_replay.py` |
| Oracle mode | `teacher_trace` |
| Teacher traces | `data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl` |
| Approved rows | `398` |
| Student models | `nano-memgpt-llama3-r8`, `nano-memgpt-llama3-r16` |
| Serving | vLLM direct chat completion |
| Max tokens | `128` |
| Judge | `gpt-4.1-2025-04-14` |

Teacher trace는 GPT-4.1이 DMR 환경에서 실제로 만든 `conversation_search` call과 tool
output을 포함한다. Student는 이 evidence를 보고 최종 답변을 생성한다. 즉 이 실험은
Letta agent loop의 parser, heartbeat, search query generation, recall search execution을
평가하지 않는다.

## 3. Results

### 3.1 LoRA Teacher-Trace Replay

| Model | Rows | ROUGE-L recall | containment | GPT-4.1 judge |
| --- | ---: | ---: | ---: | ---: |
| LoRA r8 + teacher trace | `398` | `0.7490` | `0.4874` | `343/398` (`0.8618`) |
| LoRA r16 + teacher trace | `398` | `0.7603` | `0.4899` | `345/398` (`0.8668`) |

두 LoRA adapter 모두 teacher evidence가 주어지면 `0.86`대 semantic accuracy까지 회복된다.
이는 post-LoRA end-to-end DMR의 `0.48`대 accuracy와 매우 큰 차이다.

### 3.2 End-to-End LoRA와 비교

| Condition | Evidence source | Student controls search? | Rows | containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: |
| LoRA r8 end-to-end | student retrieval | Yes | `489` completed | `0.2474` | `0.4765` |
| LoRA r8 + teacher trace | teacher evidence | No | `398` approved | `0.4874` | `0.8618` |
| LoRA r16 end-to-end | student retrieval | Yes | `497` completed | `0.2656` | `0.4809` |
| LoRA r16 + teacher trace | teacher evidence | No | `398` approved | `0.4899` | `0.8668` |

가장 중요한 차이는 search control이다. 같은 LoRA adapter라도 student가 직접 검색 query를
만들어야 하는 조건에서는 semantic accuracy가 `0.48` 근처에 머문다. 반면 teacher가 얻은
evidence를 직접 제공하면 accuracy가 `0.86` 이상으로 올라간다.

### 3.3 Frozen Student Oracle과 비교

| Condition | Model | Rows | containment | GPT-4.1 judge |
| --- | --- | ---: | ---: | ---: |
| Teacher-trace Oracle | base Llama-3-8B | `398` | `0.4246` | `292/398` (`0.7337`) |
| Teacher-trace Oracle | Mistral-7B | `398` | `0.4598` | `349/398` (`0.8769`) |
| LoRA r8 + teacher trace | Llama-3-8B LoRA r8 | `398` | `0.4874` | `343/398` (`0.8618`) |
| LoRA r16 + teacher trace | Llama-3-8B LoRA r16 | `398` | `0.4899` | `345/398` (`0.8668`) |

흥미로운 점은 LoRA Llama가 teacher evidence 조건에서 base Llama Oracle보다 훨씬 높고,
Mistral Teacher-Trace Oracle에 근접한다는 것이다. 이는 trajectory distillation이 단순한
tool-call format뿐 아니라 evidence를 짧고 적절한 최종 답변으로 바꾸는 행동도 어느 정도
학습시켰다는 신호다.

## 4. Interpretation

이 ablation은 post-LoRA failure audit의 해석을 강하게 지지한다.

첫째, LoRA adapter는 answer-from-evidence 능력을 상당히 갖고 있다. Teacher evidence가
주어졌을 때 r8과 r16은 모두 `0.86`대 judge accuracy를 기록했다. 따라서 end-to-end DMR
정확도가 낮다고 해서 adapter가 최종 답변 생성 자체를 못 배웠다고 보기는 어렵다.

둘째, end-to-end gap은 query/evidence acquisition에서 주로 발생한다. r16 LoRA는
end-to-end에서 `497/500` loop completion과 search rate `0.7948`을 기록했지만, semantic
accuracy는 `0.4809`였다. 즉 검색을 아예 하지 않는 문제는 상당히 줄었지만, 검색 query가
answer-bearing utterance를 안정적으로 끌어오지 못한다.

셋째, LoRA rank를 키우는 것만으로는 충분하지 않다. r16은 r8보다 end-to-end loop stability와
search rate가 좋지만, teacher evidence 조건에서는 둘 다 거의 같은 수준이다. 이는 다음
학습이 adapter capacity 확대보다 query selection objective와 evidence-grounding objective를
분리하는 방향이어야 함을 시사한다.

## 5. Teacher-Query Hint 결과와 연결

이 결과 이후 teacher query chain만 hint로 제공하는 ablation을 실행했다.

```text
Student에게 teacher search query chain만 hint로 제공하고,
retrieval execution과 final answer는 Letta end-to-end loop에서 수행한다.
```

이 조건은 pure forced-query가 아니라 prompt hint 기반이라는 caveat가 있지만, r16 LoRA에서
`248/394` (`0.6294`) GPT-4.1 judge accuracy를 기록했다. 이는 end-to-end r16 `0.4809`보다
높고, teacher evidence replay `0.8668`보다 낮다.

따라서 query selection은 실제 병목이지만 query만으로 전부 해결되지는 않는다. Query 이후의
검색 결과 선택, tool-result interpretation, final answer boundary도 별도 병목이다.

학습 방향은 두 가지로 나뉜다.

| 방향 | Target | 줄이고 싶은 오류 |
| --- | --- | --- |
| Query SFT | probe와 memory contract에서 teacher `conversation_search` query 예측 | `no_search`, `searched_wrong_or_insufficient_evidence` |
| Evidence-grounded Answer SFT | probe + retrieved evidence에서 concise answer 생성 | `evidence_found_but_not_used`, surface leakage |

현재 수치만 놓고 보면 Query SFT와 Evidence-grounded Answer SFT를 분리하는 것이 가장
타당하다. Query hint가 search rate를 `0.9594`까지 올렸지만, planning/tool-output leakage가
남았기 때문이다.

## 6. Artifacts

```text
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r8-teacher_trace-offset-0-limit-500.jsonl
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r8-teacher_trace-offset-0-limit-500.summary.json
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r8-teacher_trace-offset-0-limit-500.judged.gpt41.jsonl
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r8-teacher_trace-offset-0-limit-500.judged.gpt41.summary.json
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r16-teacher_trace-offset-0-limit-500.jsonl
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r16-teacher_trace-offset-0-limit-500.summary.json
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r16-teacher_trace-offset-0-limit-500.judged.gpt41.jsonl
data/evaluation/oracle_dmr_lora_teacher_trace/nano-memgpt-llama3-r16-teacher_trace-offset-0-limit-500.judged.gpt41.summary.json
logs/oracle_replay_lora_r8_teacher_trace.log
logs/oracle_replay_lora_r8_teacher_trace_judge_gpt41.log
logs/oracle_replay_lora_r16_teacher_trace.log
logs/oracle_replay_lora_r16_teacher_trace_judge_gpt41.log
```
