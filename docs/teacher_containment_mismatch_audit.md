# GPT-4.1 Teacher Containment Mismatch Audit

## 1. 감사 목적

이 문서는 GPT-4.1 teacher trajectory 중 `contains_reference = false`이지만 GPT-4.1 judge가 `correct = true`로 판정한 row를 따로 점검한 기록이다.

문제의 핵심은 다음과 같다.

```text
teacher raw 500:
  judge accuracy = 398/500 = 0.796
  containment    = 206/500 = 0.412

judge-approved 398 중:
  containment pass = 204
  containment fail = 194
```

따라서 judge-approved trajectory를 그대로 "gold oracle"로 부르면 위험하다. 이 감사의 목적은 containment는 실패했지만 judge가 맞다고 본 194개가 실제로 학습 supervision으로 쓸 만한지, 특히 query-policy 학습에 적합한지 확인하는 것이다.

사용한 원본 파일은 다음과 같다.

```text
data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.jsonl
data/evaluation/oracle_teacher_dmr_gpt41_paper_substring_scaled/openai-gpt-4-1-2025-04-14-offset-0-limit-500.judged.jsonl
```

## 2. 전체 Cross-Tab

`contains_reference`와 `judge_correct`를 교차하면 다음과 같다.

| Containment | Judge | Rows | 해석 |
| --- | --- | ---: | --- |
| Pass | Correct | `204` | exact/reference string도 있고 semantic judge도 맞다고 본 가장 깨끗한 정답 |
| Pass | Incorrect | `2` | 정답 문자열은 들어갔지만 문맥상 오답 또는 contradiction 포함 |
| Fail | Correct | `194` | 이번 감사 대상. 정답 문자열 exact match는 없지만 judge는 의미상 정답으로 봄 |
| Fail | Incorrect | `100` | teacher 실패 또는 retrieval/answer 실패 |

이 표만 보면 containment가 너무 낮아 보이지만, 실제로는 `Fail + Correct` 194개 중 상당수가 단순 paraphrase, 단복수 차이, 철자 차이, 짧은 reference 문장의 의미 변형이다.

예를 들면:

| Row | Reference | Teacher answer 핵심 | 판단 |
| ---: | --- | --- | --- |
| `5` | `I have a cat.` | `I have several cats` | 의미상 정답. containment는 exact phrase 부재로 실패 |
| `267` | `Grey!` | `Barnaby is gray` | 미국식/영국식 철자 차이 |
| `420` | `Gray!` | `favorite color is grey` | 철자 차이 |
| `458` | `Flamingo dance!` | `flamenco dancer` | dataset typo 또는 의미 교정 가능성 |

## 3. 194개 Mismatch Row의 구조적 분류

194개를 tool trace와 답변 형태 기준으로 다시 나누면 다음과 같다.

| 분류 | Rows | 의미 | 권장 처리 |
| --- | ---: | --- | --- |
| Clean search paraphrase | `108` | 검색을 수행했고, memory patch 없이, 답변도 대체로 명확한 paraphrase | answer/evidence supervision으로 사용 가능. query supervision에도 비교적 안전 |
| No-search correct | `47` | teacher가 `conversation_search`를 호출하지 않고 core/persona/immediate context에서 답함 | answer-only supervision으로는 가능하지만 query-policy 학습에는 부적합 |
| Search with memory patch | `11` | 검색 후 `memory_apply_patch`를 호출함 | full trajectory SFT에는 노이즈. query/answer만 분리 추출하는 것이 안전 |
| Search noisy or lenient | `28` | hedging, "couldn't find exact", judge lenient language, 추론/부분 정답 포함 | high-quality teacher set에서는 보류 또는 수동 검토 권장 |

즉 194개 전체를 버릴 필요는 없다. 하지만 `judge=True`라는 이유만으로 194개를 모두 같은 품질의 trajectory로 취급하면 안 된다.

## 4. Query-Policy 관점의 핵심 결론

이 194개는 "teacher가 의미상 맞게 답했는가"와 "teacher trajectory가 query-policy 학습에 좋은가"를 분리해서 봐야 한다.

### 4.1 의미상 정답으로는 대체로 쓸 수 있음

대부분의 containment mismatch는 reference answer의 exact string이 답변에 없어서 생긴다. 예를 들어 `I was three!`와 `three years old`, `I have 3 dogs!`와 `three dogs`, `Gray!`와 `grey`는 containment에는 실패하지만 의미상 정답이다.

따라서 논문에서 semantic accuracy를 볼 때는 judge-approved subset을 사용하는 것이 타당하다. 다만 이 subset을 gold label로 부르지 않고 `judge-filtered teacher trajectory`라고 부르는 것이 안전하다.

### 4.2 Query 학습용으로는 필터링 필요

`no_search_correct` 47개는 teacher가 검색을 잘해서 맞힌 것이 아니다. 이 row들은 teacher가 core memory나 immediate context에서 바로 답한 경우이므로, query-policy 학습 데이터로 쓰면 "검색하지 않아도 되는 row"와 "검색해야 하는 row"가 섞인다.

따라서 query-only SFT, retrieval-supervised objective, hard-positive query dataset에는 다음을 기본 필터로 쓰는 것이 좋다.

```text
use_for_query_policy =
  judge_correct
  and has_conversation_search
  and not has_memory_apply_patch
  and not noisy_or_lenient
```

이번 194개 mismatch 안에서는 이 조건을 만족하는 row가 `108`개다.

### 4.3 Full trajectory SFT에는 memory patch가 특히 위험함

`memory_apply_patch`가 포함된 11개 row는 answer 자체가 맞아도 full trajectory imitation에는 조심해야 한다. DMR의 목표는 과거 memory를 검색해 답하는 것이지, teacher가 persona/core memory를 수정하는 behavior까지 student에게 학습시키는 것이 아니다.

따라서 v2 학습 데이터에서는 full trajectory target에서 `memory_apply_patch` step을 제거하거나, 이 row들을 answer-only/evidence-only 형태로 재가공하는 것이 좋다.

## 5. 보류 또는 수동 검토가 필요한 대표 Row

아래 row들은 judge가 맞다고 봤지만, strict한 연구용 supervision으로는 보류하거나 수동 검토하는 것이 좋다.

| Row | Reference | Teacher answer 핵심 | 우려 | 권장 |
| ---: | --- | --- | --- | --- |
| `14` | `He was a manager at Home Depot.` | father worked for Home Depot | `manager` 역할 누락 | no-search row. query 학습 제외, answer label은 보류 |
| `38` | no specific dressing mentioned | searched many dressing terms, then says no exact dressing | negative fact는 맞지만 hedged answer | query-positive로 쓰지 않기 |
| `71` | meditate before marathon | meditation 포함, ritual details 과다, memory patch 포함 | answer는 맞지만 trajectory noisy | full trajectory 제외 |
| `75` | town name not mentioned | many town guesses 후 name not written | negative fact는 맞지만 query chain이 guess-heavy | query gold로 쓰지 않기 |
| `79` | factory product not mentioned | "I don't have a factory job" | expected와 살짝 다른 부정 설명 | 보류 또는 제외 |
| `104` | mini pony to children's hospital | hospital outing 정답, memory patch 4회 | answer는 좋지만 patch-heavy | full trajectory 제외 |
| `161` | fries, cheese curds, gravy | poutine inference, exact message 못 찾음 | 좋은 의미 추론이지만 hedged | high-quality set에서는 보류 |
| `190` | favorite color inferred red | "probably red" | inference + uncertainty | query gold로 쓰지 않기 |
| `192` | dogs would freak out | exact message 못 찾고 추측 | judge가 paraphrase로 인정했지만 supervision은 약함 | 제외 권장 |
| `316` | no young band; Eagles, Beatles, ACDC | no young band, Eagles/Beatles mention | ACDC 누락 | answer-only는 가능, strict set은 보류 |
| `379` | Robots | robot as best guess | ambiguity + best guess | 제외 권장 |
| `380` | Mostly cardio | wrestling routine + cardio | cardio가 핵심인지 불명확 | 보류 |
| `398` | public service employee | human services field | semantic equivalence가 넓음 | 보류 또는 제외 |
| `423` | Adult coloring books | coloring books | `adult` qualifier 누락, no-search | 보류 |
| `451` | army | served in WWII | army 직접 언급 없음 | 보류 |
| `457` | teach tennis | professional tennis player | 현재 직업을 잘못 좁힘 | 제외 권장 |
| `459` | hair half Rebels/half Empire | split-color Luke/Sith hair | 핵심 구조는 맞지만 details hallucinated | strict set 제외 |
| `493` | local bunny rescue | Bunny Rescue, exact full name not given | negative clarification + hedging | query gold로 쓰지 않기 |

이 표의 row들이 모두 완전 오답이라는 뜻은 아니다. 핵심은 `teacher trajectory imitation` 데이터로 쓰기에는 신호가 깨끗하지 않다는 것이다.

## 6. 권장 Dataset Policy

앞으로 학습 데이터와 논문 보고를 다음처럼 분리하는 것이 가장 안전하다.

### 6.1 Teacher 품질 보고

논문에서는 다음처럼 보고한다.

```text
GPT-4.1 teacher generated 500 trajectories.
398 were accepted by GPT-4.1 judge.
However, only 204 of the accepted trajectories contained the exact reference string.
We therefore treat the teacher data as judge-filtered supervision, not gold oracle trajectories.
```

### 6.2 Full trajectory LoRA

기존 r8/r16 LoRA는 `judge-approved 398`로 학습했으므로, 약간의 noisy trajectory가 포함되어 있다. 이 점은 한계로 명시한다.

v2에서는 다음 필터를 권장한다.

```text
full_trajectory_sft_v2 =
  judge_correct
  and no_memory_apply_patch
  and not severe_manual_review
```

### 6.3 Query-only / Retrieval-supervised 학습

query-policy 학습에는 더 엄격한 필터를 쓴다.

```text
query_policy_sft_v2 =
  judge_correct
  and has_conversation_search
  and no_memory_apply_patch
  and answer is not hedged
  and judge reason does not rely on loose semantic alignment
```

이번 194개 mismatch 중 이 조건에 가까운 clean search paraphrase row는 `108`개다. 여기에 containment-pass/judge-pass row 중 clean search row를 더하면 더 높은 precision의 query dataset을 만들 수 있다.

### 6.4 Answer-only / Evidence-only 학습

answer-only 또는 evidence-only 학습에는 no-search row도 일부 쓸 수 있다. 다만 `teacher가 검색을 통해 evidence를 찾았다`는 signal은 없으므로, query-policy 분석에는 포함하지 않는다.

## 7. 최종 판단

`containment=False, judge=True` 194개를 전수 점검한 결과, 이들을 모두 폐기할 필요는 없다. 상당수는 exact string 기준이 너무 엄격해서 생긴 정상 paraphrase다.

하지만 이 194개는 모두 같은 품질이 아니다. 약 `108/194`는 검색 기반 paraphrase trajectory로 비교적 깨끗하고, `47/194`는 검색 없이 맞힌 row이며, `39/194`는 memory patch, hedging, lenient judge, partial semantic match 때문에 query-policy 학습용으로는 보류하는 것이 좋다.

따라서 현재 연구의 해석은 다음처럼 수정하는 것이 가장 안전하다.

```text
GPT-4.1 teacher trajectory is useful but not gold.
Judge-approved trajectories provide a strong supervision source,
but query-policy learning should use a stricter subset that excludes
no-search, memory-patch, and hedged/lenient rows.
```

한국어 논문 표현:

```text
본 연구는 GPT-4.1 trajectory를 완전한 oracle로 간주하지 않는다.
대신 judge가 승인한 trajectory를 강한 teacher supervision으로 사용하되,
query-policy 분석과 학습에는 검색 수행 여부, memory patch 여부, 답변의 불확실성 표현을 기준으로
더 엄격한 subset을 구성해야 함을 확인했다.
```
