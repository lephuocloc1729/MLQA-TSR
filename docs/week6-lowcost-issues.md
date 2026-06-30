# Week 6 Issues - Low-Cost Retrieval Replication And Benchmark Lift

This issue pack is focused on improving VLSP post-submission scores by
replicating the simpler low-cost approach before adding heavier sign-aware
logic.

Current post-submission evidence:

```text
Submission 1: Task 1 F2 = 0.04, Task 2 Accuracy = 0.51
Submission 2: Task 1 F2 = 0.33, Task 2 Accuracy = 0.50
Hybrid candidate: Task 1 from submission 2 + Task 2 from submission 1
```

The main lesson is that Task 1 and Task 2 should be optimized independently.
Do not replace the current Task 2 best run while experimenting with Task 1.

The low-cost reference project reports a strong Task 1 result by indexing
training examples, retrieving similar train samples for each test sample, and
copying/unioning their `relevant_articles`. This should be implemented before
the heavier LexiSignVQA sign-detection/filtering pipeline.

Replace `M1`-`M4` with real names before creating GitHub issues.

## Entry Gate

Before starting these issues, confirm:

- The current hybrid submission has been saved and its Codabench score is
  recorded.
- `make qdrant-up` works locally.
- `python -m src.competition_submission --help` works in the project `.venv`.
- Raw train/public/private data exists under `data/raw/`.
- GPU budget is approved only for feature extraction and real VLM runs; all
  Qdrant indexing, top-k sweeps, packaging, and analysis should run locally
  after features are cached.

## W6-01 - Low-Cost Feature Cache For Text, Whole Image, And Object Features

Labels: `week-6`, `P0`, `retrieval`, `benchmark`, `needs-gpu`

Milestone: `W6 - Low-cost benchmark lift`

Owner/reviewer: `M2` / `M1`

Branch: `feat/w6-01-lowcost-feature-cache`

PR title: `feat(retrieval): cache low-cost multimodal features`

Depends on: `W5-02`

### Description

Implement a reusable feature extraction layer that matches the low-cost
reference as closely as possible:

```text
text:        jinaai/jina-embeddings-v3
whole image: nvidia/C-RADIOv2-B summary feature
objects:     OWLv2 detected object crops encoded with C-RADIOv2-B
```

This issue should only create cached features and metadata. It should not
perform Qdrant search or create submissions.

### Project Files To Change

- `src/lowcost_features.py` (new)
- `src/vision.py` if shared image preprocessing helpers are useful
- `src/utils.py` if cache/hash helpers need a small extension
- `configs/experiments/lowcost_retrieval.yaml` (new)
- `scripts/lowcost_features.sh` (new)
- `requirements.txt` only if a direct dependency is truly missing
- `tests/test_lowcost_features.py` (new)

Generated locally and ignored by Git:

- `data/outputs/lowcost_features/train_features.jsonl`
- `data/outputs/lowcost_features/public_test_features.jsonl`
- `data/outputs/lowcost_features/private_test_features.jsonl`
- `data/outputs/lowcost_features/*.manifest.json`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/text_embedding.py`:
  copy the Jina embedding model choice and wrapper idea.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/image_feature_extraction.py`:
  copy the `C-RADIOv2-B` whole-image feature extraction logic and resizing
  assumptions.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/object_detection.py`:
  copy the OWLv2 label set and detection threshold idea.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/detect_image_object.ipynb`:
  learn how object detections are generated for images.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/extract_image_feature.ipynb`:
  learn the feature cache shape used before indexing.

### Implementation Notes

- Support `--set-name train`, `public_test`, `private_test`, and `all`.
- Support `--limit` for smoke runs.
- Support `--resume` so an interrupted GPU run does not recompute existing
  images.
- For train samples, cache `sample_id`, `image_id`, `image_path`,
  `question_type`, `question`, `choices`, `answer`, and `relevant_articles`.
- For public/private test samples, cache the same query fields that exist in
  the input. Do not require gold labels.
- Build text input exactly like the low-cost notebook:

```text
Question: ...
Options:
A: ...
B: ...
```

or:

```text
Question: ...
Options:
Đúng
Sai
```

- Cache each row with:
  - `text_vector`
  - `image_general_feature_vector`
  - `image_object_feature_list_vector`
  - `object_boxes`
  - `object_scores`
  - `object_labels`
- If OWLv2 detects no objects, store one zero vector with the C-RADIO
  dimension, matching the low-cost notebook behavior.
- Metadata must include model names, feature dimensions, set name, source file
  hash, image directory hash or file list hash, threshold, created timestamp,
  and commit hash when available.
- Unit tests must use fake models and generated tiny images. CI must not
  download Jina, C-RADIO, or OWLv2.
- Do not commit feature files, model caches, or downloaded weights.

### Expected Output

Example commands:

```bash
python -m src.lowcost_features \
  --config configs/experiments/lowcost_retrieval.yaml \
  --set-name train \
  --limit 5 \
  --output-dir data/outputs/lowcost_features

python -m src.lowcost_features \
  --config configs/experiments/lowcost_retrieval.yaml \
  --set-name all \
  --resume \
  --output-dir data/outputs/lowcost_features
```

Each JSONL row should contain:

```json
{
  "id": "train_1",
  "image_id": "train_1_3",
  "question_type": "Multiple choice",
  "text_vector": [0.1],
  "image_general_feature_vector": [0.2],
  "image_object_feature_list_vector": [[0.3]],
  "object_boxes": [],
  "object_scores": [],
  "object_labels": [],
  "relevant_articles": []
}
```

### Acceptance Criteria

- [ ] Feature extraction can run for train, public test, and private test
  inputs.
- [ ] The generated train feature rows preserve `relevant_articles`.
- [ ] Test feature extraction does not require `answer` or gold
  `relevant_articles`.
- [ ] Empty OWLv2 detections produce a zero object vector instead of crashing.
- [ ] Cache manifests prevent mixing features from different models or input
  files.
- [ ] `--resume` skips already completed images and reports skipped counts.
- [ ] Unit tests run without GPU/model downloads.
- [ ] No generated feature files are committed.

### Tests

- Unit test low-cost text formatting for Multiple choice and Yes/No.
- Unit test fake text/image/object feature extraction row shape.
- Unit test empty object detection fallback vector.
- Unit test resume behavior skips existing rows.
- Unit test manifest mismatch raises a clear error.
- Manual GPU smoke on 5 train and 5 private samples.

### Verification

```bash
python -m pytest tests/test_lowcost_features.py -q
python -m src.lowcost_features --help
python -m src.lowcost_features \
  --config configs/experiments/lowcost_retrieval.yaml \
  --set-name train \
  --limit 5 \
  --output-dir data/outputs/lowcost_features
git status --ignored --short data/outputs/lowcost_features
git diff --check
```

## W6-02 - Low-Cost Qdrant Named-Vector Train Example Index

Labels: `week-6`, `P0`, `retrieval`, `benchmark`

Milestone: `W6 - Low-cost benchmark lift`

Owner/reviewer: `M2` / `M4`

Branch: `feat/w6-02-lowcost-qdrant-index`

PR title: `feat(retrieval): index low-cost train examples with named vectors`

Depends on: `W6-01`

### Description

Create a Qdrant index over train examples using the low-cost named-vector
design:

```text
text_vector
image_general_feature_vector
image_object_feature_list_vector
```

The collection payload must preserve train `relevant_articles`, because Task 1
will copy legal citations from retrieved train examples.

### Project Files To Change

- `src/lowcost_retrieval.py` (new)
- `src/retrieval.py` only if shared Qdrant helpers should be reused
- `configs/experiments/lowcost_retrieval.yaml`
- `scripts/lowcost_index.sh` (new)
- `tests/test_lowcost_index.py` (new)

Generated locally and ignored by Git:

- Qdrant collection `traffic_train_examples_lowcost`
- optional index manifest under `data/outputs/lowcost_features/index_manifest.json`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/db/qdrant.py`: named vector
  configuration and multivector setup.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/index_qdrant.ipynb`:
  exact train-example payload and vector insertion pattern.
- `src/retrieval.py`: current project Qdrant adapters and testability pattern.

### Implementation Notes

- Create/recreate a separate collection, for example
  `traffic_train_examples_lowcost`.
- Use cosine distance for all vectors.
- Use multivector `MAX_SIM` behavior for `image_object_feature_list_vector`
  when Qdrant client supports it.
- Payload must include at least:
  - `sample_id`
  - `image_id`
  - `image_path`
  - `question`
  - `question_type`
  - `choices`
  - `answer`
  - `relevant_articles`
  - `split`
- Index only train samples by default. Never index validation or test labels.
- Fail clearly if feature files are missing or dimension metadata does not
  match config.
- Tests should use fake vector stores; CI must not require Docker.

### Expected Output

```bash
python -m src.lowcost_retrieval \
  --config configs/experiments/lowcost_retrieval.yaml \
  --mode index \
  --features data/outputs/lowcost_features/train_features.jsonl
```

The command creates/recreates `traffic_train_examples_lowcost` and prints:

```text
indexed_examples=421
collection=traffic_train_examples_lowcost
vectors=text_vector,image_general_feature_vector,image_object_feature_list_vector
```

### Acceptance Criteria

- [ ] Collection uses the three low-cost named vectors.
- [ ] Object features are indexed as a multivector or documented fallback.
- [ ] Payload preserves train `relevant_articles`.
- [ ] Index command refuses to index rows without `relevant_articles`.
- [ ] Feature dimension mismatch raises a clear error.
- [ ] Unit tests do not require Qdrant or model downloads.
- [ ] No Qdrant storage or feature files are committed.

### Tests

- Unit test vector config construction.
- Unit test payload conversion from feature row.
- Unit test missing `relevant_articles` fails for train indexing.
- Unit test dimension mismatch fails.
- Unit test fake vector store receives the expected named vectors.

### Verification

```bash
python -m pytest tests/test_lowcost_index.py -q
make qdrant-up
python -m src.lowcost_retrieval \
  --config configs/experiments/lowcost_retrieval.yaml \
  --mode index \
  --features data/outputs/lowcost_features/train_features.jsonl
git diff --check
```

## W6-03 - Low-Cost Task 1 Runner And Top-K Ablation

Labels: `week-6`, `P0`, `retrieval`, `submission`

Milestone: `W6 - Low-cost benchmark lift`

Owner/reviewer: `M4` / `M2`

Branch: `feat/w6-03-lowcost-task1-runner`

PR title: `feat(retrieval): run low-cost Task 1 example retrieval`

Depends on: `W6-01`, `W6-02`, `W5-01`

### Description

Implement the low-cost Task 1 prediction path. For each public/private test
sample, query similar train examples and output the union of their
`relevant_articles`.

This issue is the main expected Task 1 score lift. It should not call the VLM.

### Project Files To Change

- `src/lowcost_retrieval.py`
- `src/pipeline.py` only if a `--mode lowcost-task1` entrypoint is preferred
- `configs/experiments/lowcost_task1_text_image_object.yaml` (new)
- `configs/experiments/lowcost_task1_text_image.yaml` (new)
- `scripts/evaluate.sh`
- `docs/experiments.md`
- `tests/test_lowcost_task1.py` (new)

Generated locally and ignored by Git:

- `data/outputs/competitions/private_task1_lowcost_*.jsonl`
- `data/outputs/submissions/submission_lowcost_task1_*.zip`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  copy the nested Qdrant `Prefetch` strategy and union of retrieved
  `relevant_articles`.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: learn the
  `TestDataset.index_result()` output shape.
- `src/competition_submission.py`: package generated Task 1 rows into the
  VLSP submission format.

### Implementation Notes

- Support a mode like:

```bash
python -m src.lowcost_retrieval --mode task1 \
  --set-name private_test \
  --features data/outputs/lowcost_features/private_test_features.jsonl \
  --output data/outputs/competitions/private_task1_lowcost.jsonl
```

- Implement nested query options equivalent to the low-cost notebook:

```text
text_vector prefetch limit = text_limit
image_general_feature_vector prefetch limit = image_limit
image_object_feature_list_vector final limit = object_limit
```

- Make limits configurable:
  - `text_limit`: default `10`
  - `image_limit`: default `5`
  - `object_limit`: default `3`
  - `max_articles`: optional cap for output articles
- Deduplicate citations by `{law_id, article_id}`.
- Preserve input order.
- Do not include `answer` in Task 1 output rows.
- Add a validation ablation command over the locked validation split so top-k
  choices can be selected before private submission.
- Keep the current best Task 2 artifact unchanged while testing Task 1.

### Expected Output

```bash
python -m src.lowcost_retrieval \
  --config configs/experiments/lowcost_task1_text_image_object.yaml \
  --mode task1 \
  --set-name private_test \
  --features data/outputs/lowcost_features/private_test_features.jsonl \
  --output data/outputs/competitions/private_task1_lowcost_t10_i5_o3.jsonl
```

Each output row must be compatible with W5 packager:

```json
{
  "id": "private_test_1",
  "image_id": "private_test_1_1",
  "question": "...",
  "relevant_articles": [
    {"law_id": "QCVN 41:2024/BGTVT", "article_id": "B.13"}
  ]
}
```

### Acceptance Criteria

- [ ] Task 1 runner uses retrieved train examples, not direct LawDB text
  retrieval.
- [ ] Runner supports public and private test inputs without gold labels.
- [ ] Output rows preserve original `id`, `image_id`, and `question`.
- [ ] Output `relevant_articles` is deduplicated and non-empty.
- [ ] Top-k limits are configurable without code changes.
- [ ] Validation ablation reports macro Precision/Recall/F2 for each setting.
- [ ] Generated Task 1 output can be packaged with the current best Task 2
  artifact.
- [ ] Unit tests do not require Docker or real embeddings.

### Tests

- Unit test union of train `relevant_articles` from retrieved examples.
- Unit test deduplication preserves stable order.
- Unit test configurable `text_limit`, `image_limit`, and `object_limit`.
- Unit test output row never contains `answer`.
- Unit test validation ablation chooses the best F2 setting from synthetic
  predictions.
- Manual Codabench A/B: keep Task 2 fixed and submit only Task 1 variants.

### Verification

```bash
python -m pytest tests/test_lowcost_task1.py tests/test_competition_submission.py -q
make qdrant-up
python -m src.lowcost_retrieval \
  --config configs/experiments/lowcost_task1_text_image_object.yaml \
  --mode task1 \
  --set-name public_test \
  --limit 5 \
  --features data/outputs/lowcost_features/public_test_features.jsonl \
  --output data/outputs/competitions/public_task1_lowcost_smoke.jsonl
python -m src.competition_submission \
  --set-name public_test \
  --task task1 \
  --task1-predictions data/outputs/competitions/public_task1_lowcost_smoke.jsonl \
  --allow-missing \
  --dry-run
git diff --check
```

## W6-04 - Low-Cost Answer-Only Multimodal Few-Shot Prompt

Labels: `week-6`, `P0`, `vlm`, `benchmark`, `needs-gpu`

Milestone: `W6 - Low-cost benchmark lift`

Owner/reviewer: `M3` / `M4`

Branch: `feat/w6-04-lowcost-answer-only-fewshot`

PR title: `feat(vlm): add low-cost answer-only few-shot prompt`

Depends on: `W6-02`, `W6-03`, `W3-03`

### Description

Implement a Task 2 prompt variant that matches the low-cost reference more
closely than the current legal JSON prompt.

The current project prompt asks for answer, citations, explanation,
confidence, and abstention. That is useful for the product demo, but Task 2
only scores the final label. This issue adds an answer-only benchmark prompt
that uses retrieved solved examples with their real images.

### Project Files To Change

- `src/prompts.py`
- `src/vlm.py`
- `src/pipeline.py`
- `src/lowcost_retrieval.py` if shared few-shot retrieval helpers are needed
- `configs/experiments/lowcost_task2_qwen_answer_only.yaml` (new)
- `tests/test_lowcost_prompt.py` (new)
- `tests/test_vlm_output.py` if parser behavior is extended

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/vlm_answer.ipynb`:
  copy the answer-only system prompt, `format_text()`, question-type filter,
  and use of retrieved example images.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: preserve NFC normalization and
  valid answer labels.
- `src/prompts.py`: current prompt variants and few-shot helper.
- `src/vlm.py`: current parser and OpenAI-compatible backend.

### Implementation Notes

- Add a prompt variant such as `lowcost_answer_only_fewshot`.
- For Multiple choice, prompt should require exactly `A`, `B`, `C`, or `D`.
- For Yes/No, prompt should require exactly `Đúng` or `Sai`.
- Do not ask for citations or explanation in this benchmark mode.
- Include up to 3 retrieved train examples.
- Each example must include:
  - example question and choices
  - `Choice: <gold answer>`
  - example image as an actual `image_url` message part
- The query message must include:
  - query question and choices
  - `Choice:`
  - query image as an actual `image_url` message part
- Filter retrieved examples by the same `question_type` as the query.
- Prevent leakage:
  - examples must come from train only;
  - never include validation/test labels except train example gold answers;
  - never include the query sample itself or same `image_id`.
- Parser should accept either:
  - raw label text;
  - JSON like `{"choice":"B"}`;
  - XML/tag style like `<answer>B</answer>`.
- Invalid labels must be recorded as invalid, not silently converted, except
  explicit normalization mappings such as `Yes -> Đúng`, `No -> Sai`.

### Expected Output

```bash
python -m src.vlm \
  --mode build-prompt \
  --variant lowcost_answer_only_fewshot \
  --sample-id train_1
```

The prompt/messages should match this idea:

```text
System: Given an image and a question about traffic in Vietnam...
If A/B/C/D are given, choose the letter only.
If Đúng/Sai are given, choose Đúng or Sai only.
No explanation needed.

Few-shot example 1:
Question: ...
Options:
A: ...
Choice: B
<example image>

Query:
Question: ...
Options:
...
Choice:
<query image>
```

### Acceptance Criteria

- [ ] New prompt variant is selectable from config.
- [ ] Few-shot examples include real example image message parts, not only text
  placeholders.
- [ ] Examples are train-only and question-type matched.
- [ ] Parser accepts raw label, `{"choice": ...}`, and `<answer>...</answer>`.
- [ ] Parser rejects labels outside `A/B/C/D` or `Đúng/Sai`.
- [ ] Existing citation-grounded prompt/parser tests still pass.
- [ ] Offline tests do not require real model calls.

### Tests

- Unit test Multiple choice low-cost message construction.
- Unit test Yes/No low-cost message construction.
- Unit test example images are included as `image_url` parts.
- Unit test validation/test answer leakage is rejected.
- Unit test parser handles raw `B`, JSON `{"choice":"B"}`, and
  `<answer>Đúng</answer>`.
- Unit test invalid label `E` fails.

### Verification

```bash
python -m pytest tests/test_lowcost_prompt.py tests/test_vlm_output.py -q
python -m src.vlm \
  --mode build-prompt \
  --variant lowcost_answer_only_fewshot \
  --sample-id train_1
git diff --check
```

## W6-05 - Low-Cost Task 2 Real VLM Runs And Answer Ensemble

Labels: `week-6`, `P0`, `vlm`, `benchmark`, `evaluation`, `needs-gpu`

Milestone: `W6 - Low-cost benchmark lift`

Owner/reviewer: `M4` / `M3`

Branch: `experiment/w6-05-lowcost-task2-vlm-runs`

PR title: `experiment: evaluate low-cost answer-only VLM prompts`

Depends on: `W6-04`, `W5-03`

### Description

Run controlled Task 2 experiments using the answer-only low-cost prompt. The
goal is to beat the current best Task 2 post-submission accuracy of `0.51`.

This issue is an experiment issue. Do not change Task 1 while testing Task 2.

### Project Files To Change

- `configs/experiments/lowcost_task2_qwen_answer_only.yaml`
- `configs/experiments/lowcost_task2_qwen_answer_only_no_examples.yaml` (new)
- `src/evaluate.py` only if answer-only diagnostics need extra fields
- `scripts/evaluate.sh`
- `docs/experiments.md`
- `tests/test_evaluate.py` if artifact fields are extended

Generated locally and ignored by Git:

- `data/outputs/experiments/lowcost_task2_*.jsonl`
- `data/outputs/experiments/lowcost_task2_*_metrics.json`
- `data/outputs/competitions/private_task2_lowcost_*.jsonl`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/vlm_answer.ipynb`:
  staged few-shot answer-only VLM setup.
- `src/pipeline.py`: current real-backend benchmark behavior and error rows.
- `src/evaluate.py`: QA accuracy and invalid prediction accounting.

### Implementation Notes

- Use the same locked validation split for every Task 2 run.
- Keep `temperature: 0.0`.
- Start with Qwen2.5-VL-7B through the OpenAI-compatible endpoint.
- Run in stages:
  1. `--limit 5` connectivity and parser smoke;
  2. `--limit 20` prompt stability smoke;
  3. full validation only if parser success is acceptable;
  4. private test only if full validation is competitive.
- Compare at least:
  - current best `fusion_text` Task 2 artifact;
  - `lowcost_answer_only_no_examples`;
  - `lowcost_answer_only_fewshot`.
- Record:
  - accuracy;
  - accuracy by question type;
  - invalid label count;
  - parse failure count;
  - latency;
  - answer distribution.
- If the answer-only prompt improves validation but private score drops, keep
  the old Task 2 artifact as the default submission component.
- Do not repeatedly tune on private post-submission score without recording
  every attempt.

### Expected Output

```bash
python -m src.pipeline \
  --mode benchmark \
  --config configs/experiments/lowcost_task2_qwen_answer_only.yaml \
  --limit 20

python -m src.evaluate \
  --config configs/experiments/lowcost_task2_qwen_answer_only.yaml \
  --predictions data/outputs/experiments/lowcost_task2_qwen_answer_only.jsonl
```

If validation is competitive:

```bash
python -m src.pipeline \
  --mode vlsp-test \
  --set-name private_test \
  --task task2 \
  --config configs/experiments/lowcost_task2_qwen_answer_only.yaml \
  --output data/outputs/competitions/private_task2_lowcost_answer_only.jsonl
```

### Acceptance Criteria

- [ ] At least one answer-only real VLM validation artifact exists with
  `mock=false`.
- [ ] Metrics report invalid labels separately from wrong answers.
- [ ] Validation comparison includes the current best Task 2 artifact.
- [ ] Private Task 2 generation is run only after a validation smoke passes.
- [ ] Generated private predictions validate with W5 packager.
- [ ] No API keys, model weights, or generated prediction artifacts are
  committed.

### Tests

- Unit test answer-only artifact evaluation.
- Unit test invalid label is counted separately.
- Unit test low-cost prompt config loading.
- Manual GPU smoke with 5 samples.
- Manual full validation before any private run.

### Verification

```bash
python -m pytest tests/test_evaluate.py tests/test_lowcost_prompt.py -q
python -m src.pipeline \
  --mode benchmark \
  --config configs/experiments/lowcost_task2_qwen_answer_only.yaml \
  --limit 5
python -m src.evaluate \
  --config configs/experiments/lowcost_task2_qwen_answer_only.yaml \
  --predictions data/outputs/experiments/lowcost_task2_qwen_answer_only.jsonl
git diff --check
```

## W6-06 - Hybrid Submission Ladder And Codabench Score Ledger

Labels: `week-6`, `P0`, `submission`, `evaluation`, `documentation`

Milestone: `W6 - Low-cost benchmark lift`

Owner/reviewer: `M1` / `M4`

Branch: `experiment/w6-06-hybrid-submission-ledger`

PR title: `experiment: track hybrid VLSP submissions and scores`

Depends on: `W6-03`, `W6-05`, `W5-01`

### Description

Create a disciplined submission ladder so the team can improve one subtask at
a time and avoid confusing which change caused a score movement.

This issue is mostly documentation and packaging safety. It should not run new
models by itself.

### Project Files To Change

- `docs/experiments.md`
- `docs/vlsp-postsubmission-log.md` (new)
- `scripts/evaluate.sh`
- `README.md` if submission commands need updating
- `tests/test_competition_submission.py` only if new packaging helpers are
  added

Generated locally and ignored by Git:

- `data/outputs/submissions/submission_hybrid_*.zip`

### Reference Files To Study

- `src/competition_submission.py`: packager and validation summaries.
- Current generated artifacts under `data/outputs/competitions/`.
- Codabench submission results manually recorded by the team.

### Implementation Notes

- Maintain a table with:
  - submission name;
  - Task 1 artifact path;
  - Task 2 artifact path;
  - zip hash;
  - Codabench Task 1 F2;
  - Codabench Task 2 accuracy;
  - notes and date.
- Submission order should isolate one change at a time:

```text
A. Task1 low-cost + Task2 current best
B. Task1 best known + Task2 low-cost answer-only
C. Task1 low-cost ablation + Task2 best known
```

- Keep `data/outputs/submissions/submission.zip` pointing to the candidate the
  team intends to submit next.
- Never overwrite the old zip without saving a named copy first.
- Do not report private scores as official ranking unless Codabench identifies
  them that way.

### Expected Output

Example ledger row:

```markdown
| candidate | task1 artifact | task2 artifact | sha256 | F2 | Accuracy | notes |
|---|---|---|---|---:|---:|---|
| hybrid_task1_lowcost_task2_best | private_task1_lowcost_t10_i5_o3.jsonl | private_task2_qwen_fusion_text.jsonl | ... | TBD | TBD | isolates Task 1 |
```

### Acceptance Criteria

- [ ] Every submitted zip has a recorded SHA256 hash.
- [ ] Every score is traceable to one pair of Task 1/Task 2 artifacts.
- [ ] The ledger clearly distinguishes internal validation from Codabench
  post-submission scores.
- [ ] Submission candidates change only one subtask at a time unless explicitly
  marked as a combined experiment.
- [ ] `submission.zip` contains exactly the two required files before submit.
- [ ] No generated submission zip is committed.

### Tests

- Manual check that each artifact path exists before recording a candidate.
- Manual check zip entries before upload.
- Unit tests for packager still pass.

### Verification

```bash
python -m pytest tests/test_competition_submission.py tests/test_submission.py -q
python - <<'PY'
from pathlib import Path
import hashlib, zipfile
p = Path("data/outputs/submissions/submission.zip")
print(hashlib.sha256(p.read_bytes()).hexdigest())
with zipfile.ZipFile(p) as z:
    print(z.namelist())
PY
git diff --check
```

## W6-07 - Optional LexiSign Sign-Aware Fallback After Low-Cost Plateau

Labels: `week-6`, `P2`, `retrieval`, `vlm`, `needs-gpu`

Milestone: `W6 - Low-cost benchmark lift`

Owner/reviewer: `M2` / `M3`

Branch: `experiment/w6-07-sign-aware-fallback`

PR title: `experiment: add sign-aware retrieval fallback after low-cost baseline`

Depends on: `W6-03`

### Description

Keep this issue optional. Start it only if the low-cost Task 1 pipeline stops
improving and there is still GPU/API budget.

The goal is to borrow the most useful part of LexiSignVQA: detected sign crops
and parent article rules. It should not replace the low-cost baseline.

### Project Files To Change

- `src/sign_retrieval.py` (new) or `src/lowcost_retrieval.py` if small enough
- `src/vision.py`
- `configs/experiments/sign_aware_task1_fallback.yaml` (new)
- `tests/test_sign_retrieval.py` (new)
- `docs/experiments.md`

### Reference Files To Study

- `../LexiSignVQA-main/src/core/extract_signs.py`: crop sign images.
- `../LexiSignVQA-main/src/core/filter_signs.py`: use VLM to select relevant
  signs only when multiple signs are detected.
- `../LexiSignVQA-main/src/core/query_signs.py`: query sign images and add
  parent/default articles.
- `../LexiSignVQA-main/src/constants.py`: parent article rules:
  `B.* -> 22`, `C.* -> 28`, `D.* -> 32`, `E.* -> 36`.

### Implementation Notes

- Use this as an ensemble source, not the main retrieval source.
- Start with rule-based parent article expansion before adding VLM sign
  filtering.
- If using VLM filtering, log raw decisions and failures.
- Compare:
  - low-cost only;
  - sign-aware only;
  - low-cost plus sign-aware union;
  - low-cost plus sign-aware rerank.
- Stop if validation F2 does not improve.

### Expected Output

```bash
python -m src.sign_retrieval \
  --config configs/experiments/sign_aware_task1_fallback.yaml \
  --set-name private_test \
  --output data/outputs/competitions/private_task1_sign_fallback.jsonl
```

### Acceptance Criteria

- [ ] Sign-aware retrieval is documented as optional/fallback.
- [ ] Parent article expansion is tested.
- [ ] Low-cost baseline remains available and unchanged.
- [ ] Validation comparison shows whether sign-aware retrieval helps.
- [ ] Generated sign crops and predictions are not committed.

### Tests

- Unit test parent article expansion.
- Unit test no detected signs falls back safely.
- Unit test union/rerank deduplicates citations.
- Manual validation F2 comparison before private submission.

### Verification

```bash
python -m pytest tests/test_sign_retrieval.py -q
python -m src.sign_retrieval --help
git diff --check
```
