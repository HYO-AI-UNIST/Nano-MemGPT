# Document-QA Context-Pack Proxy

## 목적과 적용 범위

원 제안서는 MemGPT 논문과 동일한 Document-QA 실험을 수행한다. paper-faithful 조건은
NaturalQuestions-Open 50문항과 Wikipedia 20M passage embedding index를 PostgreSQL 및
pgvector HNSW indexing으로 검색하는 것이다.

현재 공개 Hugging Face dataset인 `MemGPT/wikipedia_embeddings`에는 index 파일이 없고,
stored bytes도 0으로 보고된다. 원 논문 repository와 현재 Letta clone에서도 동일한
20M-passage artifact를 복원할 수 있는 별도 경로를 찾지 못했다.

따라서 `scripts/eval_document_qa_context_pack.py`는 archival retrieval stack을 재현하는
대신, 로컬에서 반복 가능한 context-pack 진단 실험을 제공한다.

> 이 결과는 원 논문의 20M-passage vector retrieval 재현 결과가 아니다. 보고서에서는
> 반드시 `Document-QA Context-Pack Proxy`로 표기한다.

## 원 논문 조건과 Proxy 조건 비교

| 항목 | Paper-Faithful Document QA | Local Context-Pack Proxy |
| --- | --- | --- |
| 질문 집합 | NaturalQuestions-Open 50문항 | `MemGPT/qa_data` prefix 50문항 |
| passage pool | Wikipedia 20M passages | 질문별 최대 30 local contexts |
| retrieval | pgvector HNSW 기반 archival search | 사전 제공 context를 prompt에 직접 packing |
| 평가 질문 | retrieval depth 증가에도 agent가 답을 유지하는가? | 주어진 context 안에서 student가 답을 추출하는가? |
| 결과 용도 | 논문 Figure 5와 비교 | local extraction 및 context-length 진단 |

## Local Dataset 구조

다운로드한 `MemGPT/qa_data`에는 18,585개 질문이 있다. 각 질문은 최대 30개 local context를
포함한다.

```text
Question row:
  question: NaturalQuestions-Open factoid question
  contexts[0]: injected annotated-gold passage
  contexts[1:]: DPR-ranked passages, 최대 29개
```

evaluator는 다음 두 mode를 제공한다.

| mode | 포함 context | 해석 |
| --- | --- | --- |
| `gold_plus_dpr` | annotated-gold passage 1개 + DPR passages | 정답 passage가 주어졌을 때의 extraction upper bound |
| `dpr_only` | DPR passages만 사용 | 제한적인 retrieval-conditioned proxy |

`K=40`을 요청해도 local context 수가 30개이므로 effective `K`는 더 작다.

| mode | Requested `K` | Maximum Effective `K` |
| --- | ---: | ---: |
| `gold_plus_dpr` | 40 | 30 |
| `dpr_only` | 40 | 29 |

evaluator는 requested `K`와 effective `K`를 row별 artifact에 모두 기록한다.

## 실행 방법

### Llama-3-8B

```bash
docker compose exec nano-memgpt-dev python scripts/eval_document_qa_context_pack.py \
  --model NousResearch/Meta-Llama-3-8B-Instruct \
  --output-dir data/evaluation/experiment_1/document_qa_proxy \
  --limit 50 \
  --retrieved-k 5 10 20 40 \
  --modes gold_plus_dpr dpr_only \
  --resume
```

### Mistral-7B

현재 vLLM service를 Mistral로 교체한 뒤 실행한다.

```bash
STUDENT_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 \
docker compose -f docker-compose.yaml -f docker-compose.mistral.yaml \
  --profile llama up -d --force-recreate llama-vllm

docker compose exec nano-memgpt-dev python scripts/eval_document_qa_context_pack.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output-dir data/evaluation/experiment_1/document_qa_proxy_mistral \
  --limit 50 \
  --retrieved-k 5 10 20 40 \
  --modes gold_plus_dpr dpr_only \
  --resume
```

evaluator는 각 case 이후 JSONL과 aggregate summary를 checkpoint한다.

## Metric 해석

| metric | 의미 | 주의점 |
| --- | --- | --- |
| Exact Match | normalize한 answer가 reference와 정확히 일치하는 비율 | 설명형 답변에는 엄격함 |
| Reference Containment | 생성 답변이 reference를 포함하는 비율 | verbosity에 비교적 관대함 |
| Effective `K` | 실제 prompt에 들어간 passage 수 | requested `K`와 구분해야 함 |

`gold_plus_dpr`와 `dpr_only`의 격차가 크다면, answer extraction보다 retrieval quality가
더 큰 병목일 가능성이 있다. 다만 이 proxy는 실제 MemGPT archival tool chaining을 포함하지
않으므로, 해당 결론은 local diagnostic 범위에서만 사용한다.

## 후속 과제

paper-faithful Document-QA 결과를 보고하려면 다음 중 하나가 필요하다.

1. 원 논문의 20M Wikipedia passage embedding artifact를 확보한다.
2. 동일 Wikipedia snapshot과 embedding model을 확인하고 index를 재구축한다.
3. PostgreSQL + pgvector HNSW retrieval과 MemGPT archival search paging을 연결한다.
4. retrieved `K`별 accuracy를 원 논문 Figure 5와 동일하게 평가한다.
