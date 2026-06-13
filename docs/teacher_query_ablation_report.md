# LoRA Teacher-Query Hint Ablation Report

## 1. 목적

이 실험은 post-LoRA DMR gap을 한 단계 더 분해한다. 이전 teacher-evidence ablation에서는
GPT-4.1 teacher가 얻은 tool output까지 student에게 직접 제공했다. 그 조건에서 r16 LoRA는
`0.8668` judge accuracy까지 회복되었다. 하지만 이 결과만으로는 query selection이 전부인지,
아니면 query 이후의 search execution, tool-result interpretation, final-answer boundary도
문제인지 알기 어렵다.

이번 ablation은 teacher가 사용한 `conversation_search` query chain만 student에게 hint로
준다. Student는 여전히 Letta end-to-end loop 안에서 직접 tool call을 만들고, 검색 결과를
읽고, 최종 답변을 생성해야 한다.

핵심 질문은 다음이다.

```text
teacher query chain만 알려주면 end-to-end LoRA가 얼마나 회복되는가?
```

## 2. 중요한 Caveat

이 조건은 완전한 forced-query ablation이 아니다. Letta server 내부 tool call을 가로채서
student의 query를 teacher query로 강제 치환한 것이 아니라, probe 앞에 teacher query chain을
hint로 제공했다.

따라서 이 실험은 아래 조건으로 해석해야 한다.

| 항목 | 해석 |
| --- | --- |
| What is controlled | teacher가 사용한 search query chain이 prompt hint로 제공됨 |
| What is not controlled | student가 반드시 그 query를 tool call로 실행한다고 보장하지 않음 |
| Leakage risk | query string 자체가 answer-like token을 포함할 수 있음 |
| Remaining evaluation target | hinted query를 실제 tool call로 실행하고, tool result를 final answer로 바꾸는 능력 |

이 caveat 때문에 결과를 `pure Teacher-query upper bound`라고 부르면 안 된다. 보고서에서는
`Teacher-query hint ablation`으로 부른다.

## 3. Protocol

| 항목 | 값 |
| --- | --- |
| Script | `scripts/eval_vanilla_dmr.py` |
| Model | `vllm/nano-memgpt-llama3-r16` |
| Teacher hints | `data/trajectories/gpt41_paper_substring_scaled_approved_oracle.jsonl` |
| Query policy | `all` teacher `conversation_search` queries |
| Candidate rows | approved teacher rows `398` |
| Letta max steps | `50` |
| Workers | `2` |
| Judge | `gpt-4.1-2025-04-14` |

Prompt hint format은 아래와 같다.

```text
[Research ablation: teacher search-query hints]
Use the following literal strings only as conversation_search queries.
They are not answers. Search memory first, then answer only from retrieved conversation evidence.
Suggested search queries:
1. ...
2. ...

User probe: ...
```

## 4. Results

### 4.1 r16 Teacher-Query Hint

| Metric | Value |
| --- | ---: |
| Rows | `398` |
| Completed | `394/398` |
| Behavioral failure | `4` |
| Search rate | `0.9594` |
| ROUGE-L recall | `0.6437` |
| containment | `0.3452` |
| GPT-4.1 judge | `248/394` (`0.6294`) |

Teacher query hint는 search behavior를 크게 회복시킨다. r16 end-to-end post-LoRA의 search
rate는 `0.7948`이었지만, query hint 조건에서는 `0.9594`까지 올라간다. Format/control
failure도 `4/398`으로 낮다.

하지만 semantic accuracy는 teacher-evidence replay 수준까지 올라가지 않는다. r16
end-to-end `0.4809`보다는 높지만, teacher trace evidence를 직접 제공한 `0.8668`에는 크게
못 미친다.

### 4.2 Three-Point Gap Decomposition

| Condition | Search source | Evidence source | Rows judged | Search rate | containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LoRA r16 end-to-end | student | student retrieval | `497` | `0.7948` | `0.2656` | `0.4809` |
| LoRA r16 + teacher-query hint | teacher query hint + student execution | student retrieval | `394` | `0.9594` | `0.3452` | `0.6294` |
| LoRA r16 + teacher trace | teacher trace | teacher evidence replay | `398` | n/a | `0.4899` | `0.8668` |

이 표가 현재 가장 중요한 결과다. Query hint는 end-to-end LoRA를 `0.4809`에서 `0.6294`로
올린다. 즉 query selection은 실제 병목이다. 그러나 teacher evidence 직접 제공 조건과의
차이도 `0.2374`p 남는다. 즉 query string만 알려주는 것으로는 충분하지 않고, 검색 결과를
고르고 해석해 final answer로 압축하는 단계가 별도 병목이다.

## 5. Failure Pattern

자동 surface audit으로 아래 두 유형을 확인했다.

| Pattern | Count | Wrong among pattern | Interpretation |
| --- | ---: | ---: | --- |
| Planning leakage | `56` | `35` | "Let me search...", "Trying query..." 같은 internal/tool-planning 문장이 final answer로 새어 나옴 |
| Tool-output leakage | `15` | `14` | "Showing 1 results:", "No results found.", "arguments" 같은 tool output surface가 final answer가 됨 |

예시:

```text
Reference:
  The Rolling Stones!

Answer:
  Showing 1 results:
```

이 유형은 query selection만 고쳐서는 해결되지 않는다. 모델이 tool result를 user-facing final
answer로 바꾸는 boundary를 더 명확히 배워야 한다.

## 6. Interpretation

첫째, query selection은 중요한 병목이다. Teacher query hint만 제공해도 r16 judge accuracy가
`0.4809`에서 `0.6294`로 오른다. Search rate도 `0.7948`에서 `0.9594`로 오른다.

둘째, query selection만으로는 충분하지 않다. Teacher query hint 조건이 teacher-evidence
replay `0.8668`에 도달하지 못한다는 점은, search result interpretation과 evidence-grounded
final answer generation이 별도 병목임을 보여 준다.

셋째, 다음 학습은 두 objective를 분리해야 한다.

| Objective | Input | Target |
| --- | --- | --- |
| Query-chain SFT | probe + memory contract | teacher `conversation_search` query sequence |
| Evidence-grounded Answer SFT | probe + tool result snippets | concise final answer |

단일 trajectory imitation은 tool-call format과 query policy와 final answer를 한꺼번에
섞는다. 이번 ablation은 그 혼합 objective가 왜 충분하지 않은지 보여 준다.

## 7. Artifacts

```text
data/evaluation/teacher_query_hint_r16_all/vllm-nano-memgpt-llama3-r16-teacher-query-all-offset-0-limit-500.jsonl
data/evaluation/teacher_query_hint_r16_all/vllm-nano-memgpt-llama3-r16-teacher-query-all-offset-0-limit-500.summary.json
data/evaluation/teacher_query_hint_r16_all/vllm-nano-memgpt-llama3-r16-teacher-query-all-offset-0-limit-500.judged.gpt41.jsonl
data/evaluation/teacher_query_hint_r16_all/vllm-nano-memgpt-llama3-r16-teacher-query-all-offset-0-limit-500.judged.gpt41.summary.json
logs/teacher_query_hint_r16_all.log
logs/teacher_query_hint_r16_all_judge_gpt41.log
```
