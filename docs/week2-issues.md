# Week 2 Issues - Retrieval, Few-Shot, And Ablation

This issue pack continues from the week-1 Tier A baseline:

```text
validated VLSP split + flattened LawDB
  -> Qdrant text retrieval for legal articles
  -> structured VLM answer/citation parser
  -> smoke Text-RAG pipeline
  -> week-2 retrieval, few-shot, fusion, ablation, and demo inspection
```

Week 2 should improve the retrieval and prompting quality of the week-1
pipeline. It should not implement QLoRA, OCR, or a full traffic-sign detector.
Whole-image retrieval is allowed; cropped-sign retrieval remains stretch unless
all required week-2 acceptance gates are already green.

Replace `M1`-`M4` with real member names before creating GitHub issues.

## Entry Gate From Week 1

Before merging any week-2 PR, the team should confirm:

- LawDB preprocessing produces 402 article records.
- The train/validation split is grouped by `image_id`.
- Qdrant text retrieval returns `Evidence` objects from `src/schemas.py`.
- VLM output parser returns validated `Prediction` objects.
- Evaluation can report retrieval Precision/Recall/F2 and QA Accuracy.
- A small smoke pipeline writes `PipelineResult`-compatible JSONL.

If one of these is missing, keep the week-2 issue open but mark it blocked by
the corresponding week-1 issue instead of working around the missing contract.

## W2-01 - Cached Text And Whole-Image Embeddings

Labels: `week-2`, `P0`, `retrieval`

Milestone: `W2 - Retrieval and few-shot`

Owner/reviewer: `M2` / `M1`

Branch: `feat/w2-01-cached-embeddings`

PR title: `feat(retrieval): add cached text and image embeddings`

Depends on: `W1-02`, `W1-03`, `W1-04`

### Description

Add a reusable embedding layer for week-2 retrieval experiments. This issue
extends the week-1 text retrieval foundation by supporting cached query/article
text embeddings and whole-image embeddings. It should not add sign detection,
OCR, or QLoRA.

### Project Files To Change

- `src/vision.py`
- `src/retrieval.py`
- `src/utils.py` if a small hashing/cache helper is needed
- `configs/config.yaml`
- `tests/test_vision.py` (new)
- `tests/test_retrieval_embeddings.py` (new)

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/text_embedding.py`: learn
  the wrapper pattern for one text embedding model.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/image_feature_extraction.py`:
  learn whole-image feature extraction and image resizing assumptions.
- `../LexiSignVQA-main/src/deps/embeddings.py`: learn how LexiSignVQA hides
  model-specific embedding calls behind one dependency interface.

### Implementation Notes

- Keep the public interface model-agnostic, for example `embed_texts(texts)`
  and `embed_images(paths)`.
- Unit tests must use fake embedders and tiny synthetic images; CI must not
  download models.
- Cache local embeddings under an ignored directory such as
  `data/outputs/embeddings/`.
- Cache metadata must include model name, modality, source file hash or split
  hash, embedding dimension, and created timestamp.
- Normalize vectors before storing/querying unless the selected model requires
  otherwise.
- The implementation should work on CPU; GPU support can be opportunistic.

### Expected Output

- A text embedding adapter usable by LawDB article retrieval and example
  retrieval.
- A whole-image embedding adapter usable by image/example retrieval.
- A cache manifest that prevents accidentally mixing embeddings from different
  models or dataset versions.
- Tests that prove the retrieval code can be exercised without real models.

### Acceptance Criteria

- [ ] Text and image embedding functions accept batches and return
  deterministic `list[list[float]]` or `numpy.ndarray` values.
- [ ] Fake embedders can be injected into retrieval code.
- [ ] Cache reads fail clearly when model name, dimension, or data hash does
  not match.
- [ ] Image preprocessing handles RGB conversion and missing files with helpful
  errors.
- [ ] Unit tests run without GPU, Docker, or model downloads.
- [ ] No generated embedding files are committed.

### Tests

- Unit test text embedding adapter with a fake model.
- Unit test image embedding adapter with a generated tiny image.
- Unit test vector normalization.
- Unit test cache metadata mismatch.
- Unit test missing image path error.

### Verification

```bash
python -m pytest tests/test_vision.py tests/test_retrieval_embeddings.py -q
python -m src.vision --help
git status --ignored --short data/outputs/embeddings || true
```

## W2-02 - Leakage-Safe Multimodal Example Index

Labels: `week-2`, `P0`, `retrieval`, `data`

Milestone: `W2 - Retrieval and few-shot`

Owner/reviewer: `M2` / `M1`

Branch: `feat/w2-02-example-index`

PR title: `feat(retrieval): add leakage-safe multimodal example search`

Depends on: `W1-03`, `W1-04`, `W2-01`

### Description

Build a retrieval index over training examples so the VLM can receive similar
solved examples in week-2 few-shot prompting. The index must be leakage-safe:
validation answers and validation images must never be available as retrieved
examples.

### Project Files To Change

- `src/retrieval.py`
- `src/data_utils.py` if helper loading functions are needed
- `configs/config.yaml`
- `tests/test_example_retrieval.py` (new)
- `scripts/index.sh`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/index_qdrant.ipynb`:
  learn payload design for indexed training examples.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  learn separate text, image, and sequential retrieval experiments.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: learn how
  image paths and relevant articles are attached to examples.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/db/qdrant.py`: learn the idea
  of named vectors, but keep the week-2 implementation minimal and testable.

### Implementation Notes

- Create a separate Qdrant collection for training examples, for example
  `traffic_qa_examples`.
- Payload should include `sample_id`, `image_id`, `question`, `question_type`,
  `choices`, `answer`, `relevant_articles`, `image_path`, and `split`.
- Index only the train split by default.
- Query modes should include `text`, `image`, and `fusion`.
- Fusion can be a simple weighted score combination using
  `retrieval.text_weight` and `retrieval.image_weight`.
- If the query sample has a known `image_id`, retrieval must exclude examples
  with the same `image_id` and all validation examples.
- Tests should use a fake vector store.

### Expected Output

- `python -m src.retrieval --mode index-examples --split train` indexes
  training examples only.
- `python -m src.retrieval --mode retrieve-examples --sample-id train_1 --top-k 3`
  returns similar examples with scores and payloads.
- Retrieval output can be consumed by the few-shot prompt builder in W2-04.

### Acceptance Criteria

- [ ] Example index never includes validation samples.
- [ ] Query results never include the same `image_id` group as the query when
  the query image is known.
- [ ] Text-only, image-only, and fusion modes are supported behind one
  interface.
- [ ] Results are deterministic for a fixed fake vector store.
- [ ] Payload preserves answer and relevant articles for training examples.
- [ ] Unit tests do not require Docker or real embeddings.

### Tests

- Unit test train-only indexing.
- Unit test same-image exclusion.
- Unit test validation-answer leakage prevention.
- Unit test top-k order and score fusion with synthetic vectors.
- Unit test Qdrant payload round-trip into a prompt-ready example object.

### Verification

```bash
python -m pytest tests/test_example_retrieval.py -q
make qdrant-up
python -m src.retrieval --mode index-examples --split train
python -m src.retrieval --mode retrieve-examples --sample-id train_1 --top-k 3
```

## W2-03 - Fuse Legal Evidence From Direct Retrieval And Example Votes

Labels: `week-2`, `P0`, `retrieval`, `data`

Milestone: `W2 - Retrieval and few-shot`

Owner/reviewer: `M1` / `M4`

Branch: `feat/w2-03-evidence-fusion`

PR title: `feat(retrieval): fuse retrieved law evidence and example citations`

Depends on: `W1-02`, `W1-04`, `W2-02`

### Description

Improve legal evidence selection by combining direct LawDB text retrieval from
week 1 with citation votes from retrieved training examples. This issue should
produce the final top-k `Evidence` list that is passed into the VLM.

### Project Files To Change

- `src/retrieval.py`
- `src/data_utils.py`
- `configs/config.yaml`
- `tests/test_law_lookup.py` (new)
- `tests/test_evidence_fusion.py` (new)

### Reference Files To Study

- `../LexiSignVQA-main/src/core/query_signs.py`: learn how retrieved visual
  matches are converted into `relevant_articles`.
- `../LexiSignVQA-main/src/core/lawdb/ingest_lawdb_signs.py`: learn how payloads
  preserve law/article identifiers.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  learn the idea of sequentially combining retrieval signals.

### Implementation Notes

- Direct LawDB retrieval should remain the baseline.
- Example citation votes should be rank-weighted and deduplicated.
- Final evidence must resolve every `law_id` and `article_id` to title and
  content from the processed LawDB corpus.
- Unknown articles should be reported in diagnostics; do not replace them with
  a common default such as article 22.
- Preserve score components in metadata, for example direct score, example vote
  score, fusion score, and source mode.
- Return final evidence as `list[Evidence]`.

### Expected Output

For each query, the retrieval layer can return:

```json
{
  "direct_law": ["QCVN 41:2024/BGTVT#22"],
  "example_votes": ["QCVN 41:2024/BGTVT#22", "QCVN 41:2024/BGTVT#B.4"],
  "evidence": [
    {
      "law_id": "QCVN 41:2024/BGTVT",
      "article_id": "22",
      "retrieval_method": "fusion",
      "rank": 1
    }
  ]
}
```

### Acceptance Criteria

- [ ] Final evidence contains unique law/article UIDs.
- [ ] Every final citation resolves to title and content.
- [ ] Unknown article IDs are returned in an error/diagnostic list.
- [ ] Fusion ranking is deterministic and configurable.
- [ ] No fallback article is inserted silently.
- [ ] The output validates with `Evidence` and `PipelineResult`.

### Tests

- Unit test LawDB lookup for known and unknown article IDs.
- Unit test rank-weighted voting from retrieved examples.
- Unit test direct retrieval and example votes are deduplicated.
- Unit test fused evidence validates against `src/schemas.py`.
- Unit test hallucinated/unknown citations are reported, not hidden.

### Verification

```bash
python -m pytest tests/test_law_lookup.py tests/test_evidence_fusion.py tests/test_schemas.py -q
python -m src.retrieval --mode retrieve-fusion --sample-id train_1 --top-k 5
```

## W2-04 - Retrieval-Based Top-3 Few-Shot Prompting

Labels: `week-2`, `P0`, `vlm`

Milestone: `W2 - Retrieval and few-shot`

Owner/reviewer: `M3` / `M2`

Branch: `feat/w2-04-few-shot-prompting`

PR title: `feat(vlm): add retrieved multimodal few-shot prompting`

Depends on: `W1-06`, `W2-02`, `W2-03`

### Description

Add a B3 prompt variant that includes the query image/question, fused legal
evidence, and up to three retrieved solved training examples. This builds on
the week-1 structured output parser and must keep the same
`Prediction`/`PipelineResult` contract.

### Project Files To Change

- `src/prompts.py`
- `src/vlm.py`
- `configs/config.yaml`
- `tests/test_few_shot_prompt.py` (new)
- `tests/test_vlm_output.py` if parser behavior needs extension

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/vlm_answer.ipynb`: learn
  how retrieved examples are placed into a multimodal few-shot message.
- `../LexiSignVQA-main/src/prompts/answer_prompt.py`: learn the final-answer
  constraint, but keep our stricter JSON output.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn question-type branching,
  image loading, and answer extraction.

### Implementation Notes

- Prompt variants should be named, for example `zero_shot`, `text_rag`, and
  `few_shot_rag`.
- Few-shot examples must come only from the train split.
- Each example should include image reference or placeholder, question, choices
  if present, answer, and relevant citations.
- The validation sample's gold answer must never appear in the prompt.
- Enforce a bounded example count and image budget. Start with `top_examples=3`.
- Keep the required model output as JSON with `answer`, `citations`,
  `explanation`, `confidence`, and `abstained`.
- Do not store unrestricted chain-of-thought.

### Expected Output

- A prompt builder can produce a B3 retrieved few-shot prompt from:
  query, retrieved examples, and fused legal evidence.
- The VLM wrapper can run or be mocked with the same parser used in week 1.
- The returned prediction validates against retrieved evidence.

### Acceptance Criteria

- [ ] Prompt includes no validation answer leakage.
- [ ] Prompt includes at most three examples by default.
- [ ] Prompt contains the query image, question, choices, legal evidence, and
  clear JSON output instructions.
- [ ] Multiple choice and Yes/No formats are both covered.
- [ ] Parser still rejects invalid answers and hallucinated citations.
- [ ] Offline tests do not require model weights.

### Tests

- Unit test prompt construction for Multiple choice.
- Unit test prompt construction for Yes/No.
- Unit test examples are omitted when none are retrieved.
- Unit test validation gold answer is not included in the prompt.
- Unit test parser still handles markdown-fenced JSON.

### Verification

```bash
python -m pytest tests/test_few_shot_prompt.py tests/test_vlm_output.py tests/test_schemas.py -q
python -m src.vlm --mode build-prompt --variant few_shot_rag --sample-id train_1
```

## W2-05 - Retrieval And Prompt Ablation Matrix

Labels: `week-2`, `P0`, `evaluation`, `retrieval`, `vlm`

Milestone: `W2 - Retrieval and few-shot`

Owner/reviewer: `M4` / `M3`

Branch: `experiment/w2-05-ablation-matrix`

PR title: `experiment: compare retrieval and few-shot settings`

Depends on: `W1-05`, `W1-07`, `W2-02`, `W2-03`, `W2-04`

### Description

Create a controlled experiment matrix for week 2. The goal is to compare the
week-1 baseline with direct retrieval, fused evidence, and retrieved few-shot
prompting on the same locked validation split.

### Project Files To Change

- `src/evaluate.py`
- `src/pipeline.py`
- `configs/experiments/` (new)
- `docs/experiments.md`
- `scripts/evaluate.sh`
- `tests/test_experiment_config.py` (new)

### Reference Files To Study

- `../LexiSignVQA-main/src/eval/sub_task_1.py`: retrieval Precision/Recall/F2.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: QA Accuracy and per-type
  breakdown.
- `../LexiSignVQA-main/README.md`: learn how to present experiment tables.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  learn top-k and modality ablation patterns.

### Implementation Notes

- Use the same validation split and seed for every run.
- Suggested week-2 configs:
  - `B1_zero_shot`: image + question only, no legal evidence.
  - `B2_text_rag`: week-1 direct LawDB text retrieval.
  - `B3_fused_rag`: direct LawDB retrieval + example citation fusion.
  - `B4_few_shot_rag`: fused evidence + top-3 retrieved examples.
- Compare `top_k` values such as 1, 3, and 5 when time allows.
- If a real VLM run is not available, allow a mocked smoke run but mark it as
  `mock=true` and do not report it as final model accuracy.
- Store latency, invalid prediction IDs, retrieval metrics, QA metrics, config
  name, seed, and split hash.

### Expected Output

- Experiment config files under `configs/experiments/`.
- A metrics artifact per run under `data/outputs/experiments/`.
- An ablation table in `docs/experiments.md` with retrieval and QA metrics.
- Failed/invalid sample IDs are listed for later error analysis.

### Acceptance Criteria

- [ ] All configs use the same locked validation split.
- [ ] Retrieval metrics are reported for direct/fused evidence when gold
  `relevant_articles` are available.
- [ ] QA Accuracy is reported overall and by question type.
- [ ] Invalid predictions are counted separately.
- [ ] Latency is recorded per sample and summarized.
- [ ] Experiment docs include commands, config names, and output paths.

### Tests

- Unit test experiment config loading.
- Unit test metrics artifact schema.
- Unit test invalid prediction accounting.
- Unit test two configs cannot accidentally use different split paths unless
  explicitly overridden.

### Verification

```bash
python -m pytest tests/test_experiment_config.py tests/test_evaluate.py -q
python -m src.pipeline --mode benchmark --config configs/experiments/w2_b2_text_rag.yaml --limit 5
python -m src.evaluate --predictions data/outputs/experiments/w2_b2_text_rag.jsonl
```

## W2-06 - Streamlit Evidence Inspection Demo V1

Labels: `week-2`, `P1`, `demo`, `retrieval`

Milestone: `W2 - Retrieval and few-shot`

Owner/reviewer: `M4` / `M2`

Branch: `feat/w2-06-demo-inspection`

PR title: `feat(demo): add Streamlit evidence inspection mode`

Depends on: `W1-07`, `W2-03`

### Description

Create a small Streamlit demo mode for inspecting retrieval evidence and, when
configured, VLM answers. This is not the final polished demo. It exists so the
weekly report can show the pipeline visually and so the team can debug failed
retrieval cases faster.

### Project Files To Change

- `app/streamlit_app.py`
- `app/__init__.py` if needed
- `src/pipeline.py` if a thin demo-facing function is needed
- `README.md`
- `tests/test_demo_contract.py` (new)

### Reference Files To Study

- `../LexiSignVQA-main/src/ui/inspect_subtask1.py`: learn how to show images,
  retrieved results, and legal evidence in a review UI.
- `../LexiSignVQA-main/src/ui/inspect_lawdb.py`: learn how to inspect LawDB
  articles.
- `../LexiSignVQA-main/src/core/sub_task_1.py`: learn pipeline handoff between
  retrieval stages and UI-facing artifacts.

### Implementation Notes

- Demo should support selecting a validation sample by ID before supporting
  arbitrary uploads.
- Show image, question, choices, retrieved evidence, scores, citations, and
  final prediction if available.
- If model/API credentials are missing, demo should still run in retrieval
  inspection mode.
- Do not show secrets, raw API keys, or unrestricted internal reasoning.
- Cache expensive resources with Streamlit caching.
- Keep styling simple.

### Expected Output

`streamlit run app/streamlit_app.py` opens a page where a teammate can:

- select a sample;
- view the traffic image and question;
- inspect top-k legal evidence;
- optionally run the configured VLM;
- copy cited law/article IDs for debugging.

### Acceptance Criteria

- [ ] App starts without requiring model credentials.
- [ ] Retrieval inspection works from local processed/indexed data.
- [ ] The UI clearly distinguishes retrieved evidence from model explanation.
- [ ] Citation IDs and scores are visible.
- [ ] Missing data/index/model errors are displayed as helpful messages.
- [ ] No secrets or raw local paths are exposed unnecessarily.

### Tests

- Unit test the demo-facing pipeline response contract.
- Unit test missing model credentials fall back to retrieval-only mode.
- Manual screenshot for the week-2 report.

### Verification

```bash
python -m pytest tests/test_demo_contract.py -q
streamlit run app/streamlit_app.py
```

## W2-07 - Week 2 Integration And Report

Labels: `week-2`, `P0`, `documentation`, `evaluation`

Milestone: `W2 - Retrieval and few-shot`

Owner/reviewer: `M4` / `M3`

Branch: `release/w2-retrieval-few-shot`

PR title: `release: integrate week 2 retrieval and few-shot pipeline`

Depends on: `W2-01`, `W2-02`, `W2-03`, `W2-04`, `W2-05`

### Description

Integrate week-2 retrieval and few-shot work into one reproducible release
candidate. The release should prove that the team can compare week-1 Text-RAG
against week-2 fused retrieval and few-shot prompting on the same validation
split.

### Project Files To Change

- `src/pipeline.py`
- `Makefile`
- `README.md`
- `docs/report.md`
- `docs/experiments.md`
- `scripts/index.sh`
- `scripts/evaluate.sh`
- release notes if the team keeps them

### Reference Files To Study

- `../LexiSignVQA-main/src/core/sub_task_1.py`: learn retrieval orchestration.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn answer-generation
  orchestration.
- `../LexiSignVQA-main/README.md`: learn concise experiment reporting.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/README.md`: learn how to present
  low-cost baseline components.

### Implementation Notes

- Add Makefile targets only when they wrap commands that are already working.
- Keep generated metrics small if committed; keep large artifacts under
  ignored output directories.
- The weekly report must separate real VLM metrics from mock/parser-only smoke
  metrics.
- Do not mark W2-06 as blocking if the retrieval/prompt/eval pipeline is ready
  but demo polish is not.

### Expected Output

One documented flow can:

```text
preprocess LawDB
  -> build LawDB text index
  -> build train-example index
  -> run B2/B3/B4 benchmark configs
  -> evaluate retrieval and QA
  -> update week-2 report
```

### Acceptance Criteria

- [ ] `make preprocess` still works.
- [ ] `make index` or documented commands build both required week-2 indexes.
- [ ] One benchmark command runs on at least 5 validation samples.
- [ ] Metrics artifacts include retrieval P/R/F2, QA Accuracy, invalid count,
  latency, config name, seed, and split hash.
- [ ] `docs/report.md` includes week-2 progress, metrics table, blockers,
  errors, and member contributions.
- [ ] `docs/experiments.md` includes the ablation table and exact commands.
- [ ] Release PR does not commit raw data, embeddings, Qdrant storage, model
  weights, or bulky experiment outputs.

### Tests

- Unit test pipeline integration with fake retriever and fake VLM.
- Unit test generated result JSONL validates with `PipelineResult`.
- Manual 5-sample smoke benchmark.
- Manual review of weekly report for source/metric consistency.

### Verification

```bash
make preprocess
make qdrant-up
make index
python -m src.retrieval --mode index-examples --split train
python -m src.pipeline --mode benchmark --config configs/experiments/w2_b4_few_shot_rag.yaml --limit 5
python -m src.evaluate --predictions data/outputs/experiments/w2_b4_few_shot_rag.jsonl
python -m pytest tests/test_pipeline.py tests/test_schemas.py -q
git diff --check
```

## Suggested Parallel Work Order

```text
W2-01 -> W2-02 -> W2-03 -> W2-04 -> W2-05 -> W2-07
             \                \                 /
              \----------------> W2-06 --------/
```

- `M2` owns embeddings and example retrieval: W2-01, W2-02.
- `M1` owns citation lookup and evidence fusion: W2-03.
- `M3` owns few-shot prompting and parser compatibility: W2-04.
- `M4` owns experiment matrix, demo integration, and report: W2-05, W2-06,
  W2-07.

M2 and M1 should agree on the exact payload shape before W2-02 and W2-03 are
merged. M3 should build W2-04 against fake retrieved examples first, then wire
real retrieval after W2-03. M4 should not publish final week-2 metrics until
the locked validation split and config names are confirmed.

## Week 2 Definition Of Done

- The project has reusable cached text and whole-image embeddings.
- A leakage-safe training-example index can retrieve top-k similar examples.
- Direct LawDB retrieval and example citation votes can be fused into valid
  legal `Evidence`.
- The VLM can receive a retrieved few-shot prompt without leaking validation
  answers.
- The team can compare B1/B2/B3/B4 on the same validation split.
- A retrieval inspection demo or screenshot is available for the weekly report.
- Week-2 report records metrics, failure cases, blockers, and each member's
  contribution.
