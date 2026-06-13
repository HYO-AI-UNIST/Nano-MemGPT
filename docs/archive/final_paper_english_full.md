# Why Do Small Open-Source LLMs Fail at MemGPT-Style Memory Retrieval?

**A Diagnostic Study of Tool-Call Control, Query Policy, Teacher-Trajectory Distillation, Candidate Reranking, and Evidence Filtering**

> Working paper draft. This document consolidates the Nano-MemGPT v1 experiments into a paper-style English manuscript. It is intended as the source draft for a later LaTeX submission.

## Abstract

Long-term memory agents require more than the ability to generate an answer from a provided context. A memory-augmented language model must decide when to search, formulate an effective memory query, call the retrieval tool with a valid schema, interpret noisy retrieval results, decide whether additional search is needed, and finally produce an evidence-grounded answer. Frontier proprietary models often appear to perform this behavior naturally in MemGPT-style systems, but it remains unclear whether small open-source instruction models can reliably perform the same control loop.

We study this question on MSC Deep Memory Retrieval (DMR) using Llama-3-8B and Mistral-7B in a Letta/MemGPT-style memory loop. We find that raw small models fail before retrieval quality can be measured: both models fail the tool-call control contract in a 5-row vanilla pilot. A strict template adapter allows Llama-3-8B to complete `481/500` DMR loops, but answer containment remains low (`0.1954`), showing that tool-call surface repair alone does not recover memory retrieval.

To separate answer generation from memory acquisition, we collect GPT-4.1 teacher trajectories under a local paper-substring recall contract. GPT-4.1 completes 500 rows with judge accuracy `398/500` (`0.7960`) and containment `206/500` (`0.4120`). We do not treat these trajectories as gold oracle traces: a containment mismatch audit shows that the 194 judge-correct but containment-failing rows include clean paraphrases, no-search answers, memory patches, and lenient semantic matches. We therefore use the teacher traces as judge-filtered supervision rather than gold-optimal memory policies. When the approved teacher evidence is replayed to frozen students, Llama-3-8B reaches `292/398` judge accuracy and Mistral-7B reaches `349/398`, indicating that much of the end-to-end failure lies in evidence acquisition rather than answer-from-evidence capacity.

We then fine-tune Llama-3-8B with LoRA on judge-filtered teacher trajectories. The best rank-16 adapter completes `497/500` DMR loops and reaches GPT-4.1 judge accuracy `0.4809`. However, the same adapter reaches `0.8668` judge accuracy when teacher evidence is provided, and `0.6294` when only teacher queries are hinted. These ablations identify memory-query selection as the dominant remaining bottleneck. Query-only LoRA contains useful query signal, but fails as an end-to-end agent due to tool-channel and stop-answer instability. When tool-call shells are removed and the model generates only query strings in a deterministic skeleton, raw500 retrieved-reference improves to `0.244`, but remains below teacher max-3 query replay (`0.367` on the approved subset).

Finally, we test whether direct query-policy updates can close this gap. Hard-positive SFT and zero-result DPO recover a few hard rows but reduce raw distribution performance. In contrast, search-time candidate query generation with lexical retrieval-feedback reranking slightly improves the raw distribution (`0.246` retrieved-reference, `0.224` containment) and recovers `25/72` teacher-only hard retrieval failures. A final non-oracle evidence filter preserves raw containment while reducing mean retrieved evidence from `4.742` to `3.426`.

Overall, small open-source LLMs can imitate parts of MemGPT behavior, but robust memory retrieval does not emerge from trajectory distillation alone. It requires explicitly separating tool-call channel control, query selection, retrieval-feedback reranking, evidence filtering, and evidence-grounded answering.

## 1. Introduction

Conversational agents that persist across sessions must remember facts that are not present in the current context window. A user may ask, days or weeks later, "Where did I say I work on weekends?" or "What artist did I say I could get into?" Answering such questions requires more than general world knowledge. The agent must retrieve a specific previous utterance and ground its response in that memory.

MemGPT-style systems address this problem by treating the language model as a memory controller. Instead of always answering immediately, the model can call a tool such as `conversation_search`, inspect retrieved messages, and continue searching until it has enough evidence. In principle, this turns long-term recall into an agentic control problem: decide what to search, execute the retrieval call, read the results, and produce an answer.

This control problem is easy to underestimate. A successful DMR memory agent must solve at least five subproblems:

1. **Tool-channel control**: produce a schema-valid tool call in the correct channel.
2. **Query selection**: translate an indirect user probe into a literal string likely to appear in past dialogue.
3. **Retrieval feedback use**: revise queries when previous searches return no useful evidence.
4. **Evidence filtering**: select answer-bearing messages among distractors.
5. **Answer grounding**: answer concisely without leaking search intent or hallucinating beyond evidence.

Large proprietary models often appear to satisfy these requirements. Small open-source instruction models, however, are much less reliable. Their failures can be misleading if evaluated only through end-to-end accuracy. A wrong answer might arise because the model lacks answer-generation ability, because it cannot format a tool call, because it searches with the wrong literal phrase, because it retrieves too many distractors, or because it fails to stop searching and answer.

This paper presents a diagnostic study of these failure modes. We focus on Llama-3-8B and Mistral-7B under a local Letta/MemGPT setup on MSC Deep Memory Retrieval (DMR). Rather than asking only whether small models can match GPT-4-level memory agents, we ask where the behavior breaks and which interventions recover which part of the loop.

Our contributions are:

1. **A failure decomposition of small-model MemGPT behavior**. We show that raw small models first fail the tool-call control contract, while format repair alone leaves retrieval quality low.
2. **A teacher-trace replay diagnostic separating answer capacity from evidence acquisition**. Frozen Llama and Mistral answer much better when provided GPT-4.1 teacher evidence, showing that end-to-end failure is not merely answer generation failure.
3. **A cautious audit of teacher supervision quality**. We show that GPT-4.1 teacher traces are useful but not gold: `398/500` rows are judge-approved, while only `206/500` pass exact containment. We audit the mismatch and treat the data as judge-filtered supervision.
4. **A LoRA distillation study showing partial recovery but persistent query-policy failure**. Rank-16 LoRA nearly eliminates loop failure but remains far below teacher-evidence replay, and teacher-query hints substantially improve performance.
5. **A query-policy analysis with deterministic skeletons and teacher-query replay**. Removing tool-call shell burden reveals useful query signal in query-only LoRA, but teacher max-3 queries still retrieve substantially more reference evidence.
6. **Negative and positive interventions for query policy**. Hard-positive SFT and zero-DPO overfit hard rows and hurt raw performance, while candidate generation with retrieval-feedback reranking recovers hard failures more effectively.
7. **A final evidence-filtering diagnostic**. Non-oracle lexical filtering preserves containment while reducing retrieved evidence volume, suggesting that retrieval recall and answer-time context efficiency should be handled separately.

## 2. Background and Related Work

### 2.1 Memory-Augmented LLM Agents

MemGPT introduced the idea of virtual context management for LLMs, drawing an analogy to operating systems that move information between fast and slow memory tiers [Packer et al., 2023]. In multi-session chat, the model can query a recall memory to retrieve previous dialogue. This motivates a view of long-term conversation as an active memory-management problem rather than a static long-context prompt.

Several later systems explore memory architectures beyond raw conversational retrieval. Zep and Graphiti use a temporal knowledge graph to organize agent memory and report strong DMR performance relative to MemGPT [Rasmussen et al., 2025]. Memory-R1 studies reinforcement learning for memory management, training agents to store, update, delete, and use memories [Yan et al., 2025]. These works highlight that memory behavior can be structured and learned, but they generally emphasize memory-system performance rather than isolating small-model failures inside a MemGPT/Letta tool loop.

### 2.2 Long-Term Conversational Memory Benchmarks

DMR evaluates whether an agent can recover personal facts from multi-session chat. The original MemGPT evaluation showed that recall memory improves over fixed-summary baselines on DMR [Packer et al., 2023]. More recent evaluations report very high DMR scores for GPT-4-class or GPT-4o-mini-class systems, suggesting that DMR may be relatively easy for large modern models when the full conversation or a strong memory layer is available [Rasmussen et al., 2025].

Other benchmarks stress longer and more diverse memory behavior. LongMemEval evaluates long-term interactive memory across information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention [Wu et al., 2024]. LoCoMo evaluates very long-term conversational memory over conversations with up to 35 sessions and 300 turns, showing that even strong LLMs remain far below human performance [Maharana et al., 2024]. These benchmarks reinforce the broader importance of memory, but our work asks a narrower and mechanistic question: why do small open-source models fail inside a MemGPT-style DMR tool loop?

### 2.3 Tool Use, Distillation, and Query Policy

Tool-augmented agents require models to produce structured actions and to maintain control over multi-step loops. Teacher-trajectory distillation is a natural approach: collect trajectories from a stronger model and fine-tune a smaller model to imitate them. However, our results show that trajectory imitation mixes several behaviors: tool-call formatting, search-phase query generation, stop-answer decisions, and final answer grounding. Improving token-level trajectory loss does not necessarily improve the query policy that controls retrieval success.

This paper therefore treats query policy as a separate object of study. A query is not merely a summary of the user question. Under substring recall, it must be a literal phrase likely to occur in past dialogue. This makes query generation a retrieval-control problem, where the model must learn broad-to-specific refinement and avoid plausible but wrong answer priors.

## 3. Task, Models, and Evaluation

### 3.1 MSC Deep Memory Retrieval

Each DMR row contains previous multi-session dialogue, a current user probe, and a reference answer. The probe often asks indirectly about a previously stated personal fact. For example:

```text
Probe: Hey, remember that time we talked about music?
       What was the artist you mentioned you could get into?

Reference: Taylor Swift!
```

The answer-bearing memory may be a previous utterance such as:

```text
A little bit. I can get into Taylor Swift.
```

The model must retrieve a relevant memory and answer the probe. In our Letta/MemGPT setup, each row creates a fresh agent, captures previous dialogue into recall storage, removes those messages from immediate context, and sends only the probe. This forces the model to use memory tools rather than simply reading the full dialogue in context.

### 3.2 Paper-Substring Recall Contract

During pilot experiments, we found a mismatch between maintained Letta tool descriptions and the local retrieval path. The maintained tool description suggested semantic/hybrid retrieval, while the local PostgreSQL path used case-insensitive substring matching for recall memory. To align our experiments with the paper-era DMR recall behavior, we explicitly define a local **paper-substring contract**:

```text
conversation_search(query):
  return previous messages whose content contains query
  under case-insensitive substring matching.
```

This contract makes query formulation critical. A semantic query such as `audio studio location` may fail if that phrase never appears in memory, while a shorter literal anchor such as `studio`, `California`, or `Santa Barbara` can succeed.

### 3.3 Models

We evaluate:

| Role | Model | Notes |
| --- | --- | --- |
| Student | `NousResearch/Meta-Llama-3-8B-Instruct` | main student and LoRA target |
| Student | `mistralai/Mistral-7B-Instruct-v0.3` | frozen replay comparison |
| Teacher/Judge | `gpt-4.1-2025-04-14` | teacher trajectory generation and semantic judging |

All local 7B/8B model experiments use BF16 serving without quantization. We use LoRA rather than full fine-tuning due to GPU memory constraints.

### 3.4 Metrics

We report:

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Loop completion | Whether the agent run finishes without behavioral/tool failure | control-loop stability |
| Format failure | Tool-call schema or channel failure | tool-channel control |
| Search rate | Fraction of rows with at least one `conversation_search` | search activity, not necessarily quality |
| Retrieved-reference | Whether retrieved evidence contains the reference string | retrieval proxy |
| Containment | Whether final answer contains normalized reference string | strict exact/substring proxy |
| ROUGE-L recall | LCS recall between answer and reference | soft lexical overlap |
| GPT-4.1 judge accuracy | Semantic correctness judged by GPT-4.1 | more flexible but not human ground truth |

Containment and judge accuracy answer different questions. Containment is strict and can reject valid paraphrases; judge accuracy can accept paraphrases but may be lenient. We therefore use both and explicitly audit their disagreement for teacher data.

## 4. Diagnostic Methodology

Our experiments form a diagnostic ladder. Each condition removes or controls one source of failure, allowing us to localize the next bottleneck.

| Stage | Question | Diagnostic |
| --- | --- | --- |
| Vanilla MemGPT | Can the small model enter the tool loop? | raw Letta/MemGPT DMR |
| Strict template | Is the first barrier only tool-call surface form? | rule-based parser/template repair |
| Full history | Can the model answer when search is removed? | full previous dialogue in prompt |
| Teacher trace replay | Can the model answer with teacher evidence? | replay approved GPT-4.1 tool outputs |
| LoRA distillation | Can teacher trajectories teach the loop? | full trajectory LoRA r8/r16 |
| Teacher query hint | How much does better query selection help? | teacher query chain + student execution |
| Query-only LoRA | Can query policy be learned separately? | search-call-only SFT |
| Deterministic skeleton | What if the model only emits query strings? | fixed tool wrapper |
| Teacher query replay | How far is student query quality from teacher query quality? | replay teacher query strings |
| Hard SFT/DPO | Can hard examples repair the gap? | positive SFT and zero-result DPO |
| Candidate reranking | Can search-time retrieval feedback help? | generate multiple queries and rerank |
| Evidence filtering | Can answer context be reduced without losing accuracy? | non-oracle top-k evidence filter |

This design intentionally avoids a single "accuracy race." The goal is to explain which part of the memory-agent behavior is missing.

## 5. Results

### 5.1 Raw Small Models Fail the Tool-Call Contract

In a raw vanilla Letta/MemGPT pilot, both Llama-3-8B and Mistral-7B fail all 5 DMR rows before retrieval quality can be measured.

| Model | Rows | Completed loops | Behavioral failures |
| --- | ---: | ---: | ---: |
| Llama-3-8B | 5 | 0 | 5 |
| Mistral-7B | 5 | 0 | 5 |

This result is not a retrieval-quality measurement. It shows that raw small instruction models do not reliably satisfy the structured tool-call contract required by the agent loop.

### 5.2 Strict Formatting Allows Loops but Does Not Recover Retrieval

A strict-template adapter converts explicit model intent into schema-valid tool calls without changing query content. Under this condition, Llama-3-8B completes most rows but remains inaccurate.

| Condition | Rows | Completed | Format failures | Search rate | Containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw Llama pilot | 5 | 0 | 5 | n/a | n/a |
| Strict-template Llama | 500 | 481 | 19 | 0.8462 | 0.1954 |

Thus, tool-call surface repair removes the first barrier but leaves query selection, retrieval chaining, and answer construction unresolved.

### 5.3 Full-History Context Shows That Retrieval Is a Major Bottleneck

Before teacher collection, we evaluate a no-tool upper-bound proxy by placing all previous MSC sessions directly in context.

| Model | Rows | Completed | Containment |
| --- | ---: | ---: | ---: |
| Llama-3-8B full history | 500 | 500 | 0.5080 |
| Mistral-7B full history | 500 | 500 | 0.4240 |

For Llama, containment rises from `0.1954` in strict-template DMR to `0.5080` when search is removed. This does not mean answer generation is perfect, but it strongly suggests that retrieval selection and tool-loop behavior are major bottlenecks.

### 5.4 GPT-4.1 Teacher Traces Are Useful but Not Gold

We collect 500 GPT-4.1 teacher trajectories under the paper-substring contract. All runs complete without infrastructure error.

| Metric | Value |
| --- | ---: |
| Teacher rows | 500 |
| Completed | 500 |
| GPT-4.1 judge accuracy | `398/500` (`0.7960`) |
| ROUGE-L recall | 0.7393 |
| Containment | `206/500` (`0.4120`) |
| Search rate | 0.7740 |

We use only judge-approved rows for replay and LoRA training. However, we do **not** call these traces gold oracle trajectories. A containment mismatch audit shows substantial heterogeneity:

| Containment | Judge | Rows | Interpretation |
| --- | --- | ---: | --- |
| Pass | Correct | 204 | clean exact and semantic success |
| Pass | Incorrect | 2 | string included but answer contradicted expected meaning |
| Fail | Correct | 194 | semantic paraphrase or lenient acceptance |
| Fail | Incorrect | 100 | teacher failure |

Among the 194 judge-correct but containment-failing rows:

| Category | Rows | Use |
| --- | ---: | --- |
| Clean search paraphrase | 108 | relatively safe for answer/evidence and query supervision |
| No-search correct | 47 | answer-only possible, not query-policy supervision |
| Search with memory patch | 11 | noisy for full trajectory imitation |
| Search noisy or lenient | 28 | manual review or exclusion for high-precision query training |

This audit affects our interpretation. GPT-4.1 provides a strong teacher signal, but teacher trajectories are judge-filtered supervision, not gold-optimal memory policies.

### 5.5 Teacher-Trace Replay Separates Evidence Use from Evidence Acquisition

We replay the 398 judge-approved teacher traces to frozen students. In this condition, the student does not search; it receives the teacher's retrieved evidence and produces the final answer.

| Model | Rows | Completed | Judge accuracy | ROUGE-L recall | Containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-8B | 398 | 398 | `292/398` (`0.7337`) | 0.6597 | 0.4246 |
| Mistral-7B | 398 | 398 | `349/398` (`0.8769`) | 0.7208 | 0.4598 |

Mistral is particularly revealing: it fails the vanilla Letta gate, but answers very well when given teacher evidence. This separates tool-call compatibility from answer-from-evidence ability.

### 5.6 Full Trajectory LoRA Recovers Control but Not Teacher-Level Retrieval

We train Llama-3-8B LoRA adapters on the 398 judge-approved teacher trajectories. The rank-16 adapter slightly improves token-level training metrics relative to rank 8.

| Adapter | Rank | Eval loss | Eval token accuracy |
| --- | ---: | ---: | ---: |
| Full trajectory LoRA r8 | 8 | 0.9009 | 0.7546 |
| Full trajectory LoRA r16 | 16 | 0.8694 | 0.7599 |

End-to-end DMR improves loop stability and semantic accuracy, but remains far below teacher-evidence replay.

| Condition | Loop completion | Format failures | Search rate | Containment | GPT-4.1 judge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict-template Llama | `481/500` | 19 | 0.8462 | 0.1954 | not run |
| LoRA r8 | `489/500` | 11 | 0.7280 | 0.2474 | 0.4765 |
| LoRA r16 | `497/500` | 3 | 0.7948 | 0.2656 | 0.4809 |

When the same LoRA adapters receive teacher evidence, performance sharply increases:

| Condition | Student controls search? | Rows | Containment | GPT-4.1 judge |
| --- | --- | ---: | ---: | ---: |
| LoRA r8 end-to-end | Yes | 489 completed | 0.2474 | 0.4765 |
| LoRA r8 + teacher trace | No | 398 approved | 0.4874 | 0.8618 |
| LoRA r16 end-to-end | Yes | 497 completed | 0.2656 | 0.4809 |
| LoRA r16 + teacher trace | No | 398 approved | 0.4899 | 0.8668 |

This demonstrates that LoRA learns useful answer-from-evidence behavior, but the end-to-end agent still fails to acquire the right evidence.

### 5.7 Teacher Query Hints Identify Query Selection as a Major Bottleneck

We then provide the teacher query chain as a hint while allowing the student to execute retrieval and answer.

| Condition | Search source | Evidence source | Rows judged | Search rate | Containment | GPT-4.1 judge |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LoRA r16 end-to-end | student | student retrieval | 497 | 0.7948 | 0.2656 | 0.4809 |
| LoRA r16 + teacher-query hint | teacher query + student execution | student retrieval | 394 | 0.9594 | 0.3452 | 0.6294 |
| LoRA r16 + teacher trace | teacher query and evidence | teacher evidence replay | 398 | n/a | 0.4899 | 0.8668 |

Teacher query hints recover a large fraction of the gap. The remaining difference between teacher-query hint and full teacher-trace replay reflects retrieval execution, evidence interpretation, and answer boundary losses.

### 5.8 Query-Only LoRA Fails as an Agent but Contains Useful Query Signal

We train a query-only LoRA adapter using only `conversation_search` targets. Token-level metrics are strong, but the adapter fails as an end-to-end Letta agent because it cannot reliably preserve tool calls in the correct channel or transition from search to answer.

| Condition | Proxy metric | End-to-end outcome |
| --- | --- | --- |
| Query+answer LoRA r16 | eval loss 0.8583, token acc 0.7859 | 34/35 early rows fail tool-call format |
| Query-only LoRA r16 | eval loss 0.8244, token acc 0.7703 | first 5 rows all fail end-to-end |

However, raw traces show that query-only LoRA does generate many valid search calls before eventually outputting JSON-like tool calls as normal assistant content. Endpoint rescue confirms that channel repair alone is not enough: the model still lacks a stop-and-answer policy. This motivates separating query generation from tool shell and phase control.

### 5.9 Deterministic Query Skeleton Reveals Query-Policy Signal

In the deterministic skeleton, the model emits only a query string. The wrapper supplies the `conversation_search` shell. This removes tool-call schema burden and stop-answer control from the query model.

| Query generator | Rows | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16, phase-routed tool-call | 100 | 0.09 | 0.11 | 1.10 |
| Query-only r16, skeleton | 100 | 0.23 | 0.23 | 3.70 |
| Full trajectory r16, skeleton | 100 | 0.16 | 0.17 | 2.86 |

On raw500, query-only skeleton obtains retrieved-reference `0.244` and containment `0.216`. This confirms that query-only SFT learned useful query content, but that the agent interface had hidden that signal.

### 5.10 Teacher Query Replay Quantifies the Remaining Gap

We replay GPT-4.1 teacher query strings in the same deterministic skeleton. Full teacher chains and max-3 teacher chains both outperform student query-only skeletons.

| Condition | Subset | Rows | Retrieved-reference | Containment | Mean retrieved |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher full query chain | approved | 398 | 0.442 | 0.405 | 4.36 |
| Teacher full query chain | teacher-search only | 302 | 0.583 | 0.533 | 5.74 |
| Teacher max-3 query | approved | 398 | 0.367 | 0.342 | 3.25 |
| Teacher max-3 query | teacher-search only | 302 | 0.483 | 0.450 | 4.28 |
| Query-only r16 skeleton | same approved | 398 | 0.276 | 0.249 | 3.76 |
| Query-only r16 skeleton | teacher-search only | 302 | 0.272 | 0.238 | 3.76 |

Row-level comparison on the teacher-search subset shows an asymmetric gap:

| Category | Retrieval count | Retrieval rate |
| --- | ---: | ---: |
| Both teacher and student retrieve reference | 74 | 0.245 |
| Teacher only | 72 | 0.238 |
| Student only | 8 | 0.026 |
| Neither | 148 | 0.490 |

The student does not simply differ from the teacher randomly. Teacher-only successes are much more common than student-only successes.

Qualitative analysis reveals three patterns:

1. **Probe phrase copying**: the student copies long probe phrases that do not occur in memory.
2. **Plausible but wrong answer priors**: the student searches for a guessed answer before retrieving evidence.
3. **Missing discriminative literals**: the student finds broad words but misses the final literal anchor that retrieves the answer-bearing message.

### 5.11 Hard-Set SFT and DPO Do Not Generalize

We construct a teacher-only hard set of 72 rows where teacher max-3 retrieves the reference but query-only skeleton does not. Positive SFT on these teacher queries recovers some hard rows but hurts the raw distribution.

| Query generator | Raw500 retrieved-reference | Raw500 containment | Hard72 retrieved-reference | Hard72 containment |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16 skeleton | 0.244 | 0.216 | 0/72 | 1/72 |
| Hard-positive SFT | 0.184 | 0.180 | 7/72 | 7/72 |

We also train a zero-result DPO adapter using 51 preference pairs where teacher queries retrieve the reference and student rejected queries retrieve zero results.

| Query generator | Raw500 retrieved-reference | Raw500 containment | Hard72 retrieved-reference | Hard72 containment |
| --- | ---: | ---: | ---: | ---: |
| Query-only r16 skeleton | 0.244 | 0.216 | 0/72 | 1/72 |
| Preference zero-DPO | 0.166 | 0.158 | 8/72 | 7/72 |

Both objectives contain valid local signal, but with only small hard-case data they over-specialize and narrow the query policy. Direct weight updates are therefore not sufficient in this v1 setting.

### 5.12 Candidate Query Reranking Improves Hard Failures

We next generate multiple candidate queries at each search step and rerank them using non-oracle local retrieval feedback. The lexical reranker scores result count, query specificity, probe/evidence overlap, repeated-query penalties, broad-result penalties, and repeated-evidence penalties. It never uses the reference answer.

On raw500:

| Query strategy | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | 0.244 | 0.216 | 3.744 |
| Hard-positive SFT skeleton | 0.184 | 0.180 | 2.882 |
| Preference zero-DPO skeleton | 0.166 | 0.158 | 2.564 |
| Candidate count rerank, target 3 | 0.210 | 0.202 | 3.218 |
| Candidate lexical rerank, target 5 | 0.246 | 0.224 | 4.742 |

The raw improvement is small, but this is the first non-oracle condition that improves the full raw distribution while also improving the targeted hard class.

On the teacher-only hard 72 rows:

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | 1.000 | 0.847 | 4.125 |
| Query-only skeleton | 0.000 | 0.014 | 2.028 |
| Hard-positive SFT | 0.097 | 0.097 | 1.611 |
| Preference zero-DPO | 0.111 | 0.097 | 1.500 |
| Candidate lexical rerank, target 5 | 0.347 | 0.306 | 4.097 |

The lexical reranker retrieves the reference in `25/72` teacher-only hard rows and raises containment to `22/72`, far above the SFT/DPO pilots.

### 5.13 Evidence Filtering Reduces Context Volume Without Lowering Containment

Candidate lexical reranking increases retrieval volume, creating more distractors for the answer model. We therefore apply a final non-oracle lexical evidence filter before answer generation.

| Condition | Retrieved-reference | Containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | 0.244 | 0.216 | 3.744 |
| Candidate lexical rerank, target 5 | 0.246 | 0.224 | 4.742 |
| Candidate lexical rerank + evidence filter top-6 | 0.234 | 0.224 | 3.426 |

Filtering slightly reduces retrieved-reference but preserves final containment while cutting mean retrieved evidence. On hard subsets, the same pattern holds:

| Subset | Before filter | After filter | Containment |
| --- | --- | --- | --- |
| Hard72 retrieved-reference | 25/72 | 23/72 | 22/72 unchanged |
| Zero51 retrieved-reference | 18/51 | 16/51 | 15/51 unchanged |

The filter is not a new accuracy improvement. It shows that query-time recall and answer-time context efficiency can be separated.

## 6. Discussion

### 6.1 The Main Bottleneck Is Query Policy, Not Answer Knowledge

Several results point to the same conclusion. Full-history prompting improves over strict-template DMR. Teacher-trace replay allows frozen students to answer many rows. LoRA r16 reaches `0.8668` judge accuracy when teacher evidence is provided, but only `0.4809` end-to-end. Teacher query hints raise performance to `0.6294`. Together, these results show that the model often can answer when evidence is present; it fails because it does not reliably acquire the evidence.

### 6.2 Tool Use and Query Selection Are Different Problems

Strict-template repair and LoRA distillation improve loop completion, but not enough to reach high retrieval accuracy. Query-only LoRA shows the opposite failure: it learns useful query strings, but cannot remain stable as a full agent. This suggests that small memory agents should not force a single small model adapter to handle tool-call transport, query selection, phase transition, and answer generation simultaneously.

### 6.3 Teacher Trajectories Help but Must Be Audited

The teacher containment mismatch audit is central. GPT-4.1 is a strong teacher, but its trajectories contain paraphrases, no-search answers, memory patches, and leniently judged rows. Treating such data as gold can overstate claims and contaminate query-policy training. The right framing is **judge-filtered teacher supervision**, with stricter subsets for query-specific objectives.

### 6.4 Search-Time Control May Be More Robust Than Small Hard-Set Fine-Tuning

Hard-positive SFT and zero-DPO recover a few memorized hard rows but hurt raw performance. Candidate reranking, by contrast, uses the retrieval system itself as feedback at inference time. This more directly targets the problem: the model can propose several imperfect queries, and the controller can choose the one whose local results look most useful. The gains are still modest, but the direction is more stable.

### 6.5 Retrieval Recall Creates an Evidence-Volume Problem

The lexical candidate reranker improves hard recall by retrieving more evidence. This also increases distractors. The final evidence filter shows that some of this volume can be reduced without lowering containment, but lexical filtering is imperfect. A learned evidence sufficiency model or semantic reranker is a natural next step.

## 7. Limitations

1. **Local substring recall is not the full MemGPT retrieval stack.** We intentionally use a paper-substring contract to match the local recall path, but this does not reproduce all archival or embedding-based retrieval conditions.
2. **GPT-4.1 differs from the original MemGPT teacher model.** The original MemGPT paper used GPT-4-era models. We use `gpt-4.1-2025-04-14` for accessibility and consistency.
3. **Teacher traces are not gold.** The teacher audit shows that judge-approved data includes noisy trajectories. Our claims should be read as claims about judge-filtered teacher supervision, not optimal memory policies.
4. **GPT-4.1 judging is automatic.** Semantic judging is more flexible than containment but not a substitute for human evaluation. Manual auditing of representative rows remains necessary.
5. **Mistral is not fine-tuned.** Mistral is evaluated in frozen full-history and teacher-trace replay conditions, but LoRA and query-policy experiments focus on Llama-3-8B.
6. **The final reranker and evidence filter are lexical heuristics.** They do not use reference answers, but they may miss paraphrases or indirect clues.
7. **The hard-set SFT/DPO pilots are small.** The negative result should not be read as a universal failure of preference learning for memory queries. Larger, more diverse preference datasets may behave differently.

## 8. Reproducibility and Ethical Considerations

All experimental claims in this draft are tied to local artifacts under `docs/`, `data/evaluation/`, `data/trajectories/`, and `outputs/`. The main evaluation scripts are listed in Appendix A. The DMR recall contract, model checkpoints, LoRA adapters, teacher filtering, and reranking/evidence-filtering diagnostics are documented in the corresponding reports. The most important reproducibility caveat is that the teacher collection uses a fixed GPT-4.1 snapshot and a local paper-substring memory contract; reproducing the exact numeric values requires matching both.

The experiments use conversational memory data from a benchmark derived from multi-session chat. This setting is privacy-sensitive by nature, even when benchmark data are synthetic or anonymized. A memory agent that retrieves personal facts must avoid exposing unrelated private details and must distinguish evidence-grounded answers from guesses. Our evidence-filtering and judge-audit analyses are partly motivated by this concern: a useful memory system should retrieve enough context to answer, but not indiscriminately surface irrelevant personal history.

## 9. Conclusion

Small open-source LLMs can imitate parts of MemGPT-style memory behavior, but robust long-term memory retrieval does not emerge from trajectory distillation alone. Raw models first fail the tool-call contract. Strict formatting repairs the surface but leaves retrieval weak. Teacher-trace replay shows that the models can often answer when evidence is present. LoRA distillation improves loop stability and evidence-grounded answering, but the end-to-end agent remains bottlenecked by query selection.

The strongest evidence for this conclusion comes from teacher-query diagnostics. Query-only SFT contains useful query signal, but it must be evaluated in a deterministic skeleton to reveal it. Teacher max-3 queries still retrieve far more reference evidence, and row-level analysis shows a systematic teacher-only hard class. Direct hard-set SFT and DPO do not generalize; search-time candidate generation with retrieval-feedback reranking is more effective at recovering hard failures. Evidence filtering then reduces the context burden created by higher-recall search.

The practical lesson is that small memory agents should be designed as modular systems. Tool-call channel control, query selection, retrieval-feedback reranking, evidence filtering, and final answer grounding are distinct capabilities. Treating them as one behavior to be copied from a teacher trajectory is too coarse. Future small-model memory agents should train and evaluate these components separately.

## References

- Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., and Gonzalez, J. E. (2023). **MemGPT: Towards LLMs as Operating Systems**. arXiv:2310.08560. https://arxiv.org/abs/2310.08560
- Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., and Yu, D. (2024). **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory**. arXiv:2410.10813. https://arxiv.org/abs/2410.10813
- Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. (2024). **Evaluating Very Long-Term Conversational Memory of LLM Agents**. arXiv:2402.17753. https://arxiv.org/abs/2402.17753
- Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and Chalef, D. (2025). **Zep: A Temporal Knowledge Graph Architecture for Agent Memory**. arXiv:2501.13956. https://arxiv.org/abs/2501.13956
- Yan, S., Yang, X., Huang, Z., Nie, E., Ding, Z., Li, Z., Ma, X., Schutze, H., Tresp, V., and Ma, Y. (2025). **Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning**. arXiv:2508.19828. https://arxiv.org/abs/2508.19828

## Appendix A. Artifact Map

| Component | Artifact |
| --- | --- |
| Vanilla and strict-template baseline | `docs/experiment_1_report.md` |
| DMR protocol | `docs/vanilla_dmr_protocol.md` |
| Teacher trajectory and replay | `docs/oracle_experiment_report.md` |
| Teacher quality audit | `docs/teacher_containment_mismatch_audit.md` |
| LoRA training | `docs/lora_training.md` |
| Post-LoRA evaluation | `docs/post_lora_evaluation_report.md` |
| Failure audit | `docs/failure_audit_report.md` |
| Teacher evidence ablation | `docs/teacher_evidence_ablation_report.md` |
| Teacher query hint | `docs/teacher_query_ablation_report.md` |
| Query-only LoRA | `docs/query_only_lora_report.md` |
| Phase-routed diagnostic | `docs/phase_routed_dmr_report.md` |
| Deterministic query skeleton | `docs/query_skeleton_dmr_report.md` |
| Teacher query replay | `docs/teacher_query_skeleton_report.md` |
| Query gap analysis | `docs/query_skeleton_gap_report.md` |
| Hard-positive SFT | `docs/query_hard_positive_lora_report.md` |
| Zero-result DPO | `docs/query_preference_dpo_report.md` |
| Candidate reranking | `docs/query_candidate_rerank_report.md` |
| Evidence filtering | `docs/evidence_filter_report.md` |

## Appendix B. Main Result Tables

### B.1 End-to-End and Replay Conditions

| Condition | Model | Rows | Loop completion | Containment | Judge accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw vanilla pilot | Llama-3-8B | 5 | 0/5 | n/a | n/a |
| Strict-template DMR | Llama-3-8B | 500 | 481/500 | 0.1954 | not run |
| Full-history | Llama-3-8B | 500 | 500/500 | 0.5080 | not judged |
| Full-history | Mistral-7B | 500 | 500/500 | 0.4240 | not judged |
| Teacher-trace replay | Llama-3-8B | 398 | 398/398 | 0.4246 | 0.7337 |
| Teacher-trace replay | Mistral-7B | 398 | 398/398 | 0.4598 | 0.8769 |
| LoRA r8 end-to-end | Llama-3-8B | 500 | 489/500 | 0.2474 | 0.4765 |
| LoRA r16 end-to-end | Llama-3-8B | 500 | 497/500 | 0.2656 | 0.4809 |
| LoRA r16 + teacher query hint | Llama-3-8B | 398 judged | 394 completed | 0.3452 | 0.6294 |
| LoRA r16 + teacher trace | Llama-3-8B | 398 | n/a | 0.4899 | 0.8668 |

### B.2 Query Policy Conditions

| Query strategy | Raw500 retrieved-reference | Raw500 containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Query-only skeleton | 0.244 | 0.216 | 3.744 |
| Hard-positive SFT | 0.184 | 0.180 | 2.882 |
| Preference zero-DPO | 0.166 | 0.158 | 2.564 |
| Candidate count rerank, target 3 | 0.210 | 0.202 | 3.218 |
| Candidate lexical rerank, target 5 | 0.246 | 0.224 | 4.742 |
| Candidate lexical rerank + evidence filter top-6 | 0.234 | 0.224 | 3.426 |

### B.3 Teacher-Only Hard Set

| Condition | Hard72 retrieved-reference | Hard72 containment | Mean retrieved |
| --- | ---: | ---: | ---: |
| Teacher max-3 query | 72/72 | 61/72 | 4.125 |
| Query-only skeleton | 0/72 | 1/72 | 2.028 |
| Hard-positive SFT | 7/72 | 7/72 | 1.611 |
| Preference zero-DPO | 8/72 | 7/72 | 1.500 |
| Candidate lexical rerank | 25/72 | 22/72 | 4.097 |
| Candidate lexical rerank + evidence filter | 23/72 | 22/72 | 3.014 |
