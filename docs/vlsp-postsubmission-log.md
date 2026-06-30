# VLSP Post-Submission Ledger

This ledger tracks hybrid VLSP submissions one pair of artifacts at a time.
It is intentionally conservative: do not upload a zip unless both task files
were generated from named JSONL artifacts and the zip SHA256 is recorded here.

## Rules

- Keep `data/outputs/submissions/submission.zip` as the next candidate to
  submit, but always save a named copy first, for example
  `submission_hybrid_task1_lowcost_task2_best.zip`.
- Change only one subtask at a time unless the row is explicitly marked as a
  combined experiment.
- Keep Task 1 and Task 2 artifact paths in every row so score movement is
  traceable.
- Record Codabench post-submission scores separately from internal validation
  scores.
- Do not call private post-submission scores official ranking scores unless
  Codabench labels them that way.
- Do not commit generated zips, private predictions, raw backend logs, API
  keys, or model outputs under `data/outputs/`.

## Packaging Command

Use the wrapper below to validate both artifacts, create a named zip, back up
any existing `submission.zip`, and copy the named zip to
`data/outputs/submissions/submission.zip`.

```bash
bash scripts/evaluate.sh hybrid-submission \
  hybrid_task1_lowcost_task2_best \
  data/outputs/competitions/private_task1_lowcost_t10_i5_o3.jsonl \
  data/outputs/competitions/private_task2_best.jsonl \
  private_test
```

The command prints a ledger-ready row with:

- candidate name;
- Task 1 artifact path;
- Task 2 artifact path;
- named zip path;
- SHA256 hash;
- required zip entries.

Before upload, verify the final zip:

```bash
python - <<'PY'
from pathlib import Path
import hashlib, zipfile

p = Path("data/outputs/submissions/submission.zip")
print(hashlib.sha256(p.read_bytes()).hexdigest())
with zipfile.ZipFile(p) as z:
    print(z.namelist())
PY
```

Expected entries:

```text
['submission_task1.json', 'submission_task2.json']
```

## Planned Ladder

| Step | Candidate | Task 1 artifact | Task 2 artifact | Isolated change | Status |
| --- | --- | --- | --- | --- | --- |
| A | `hybrid_task1_lowcost_task2_best` | `data/outputs/competitions/private_task1_lowcost_t10_i5_o3.jsonl` | current best Task 2 artifact, for example `data/outputs/competitions/private_task2_best.jsonl` | Task 1 low-cost retrieval | pending artifacts/hash |
| B | `hybrid_task1_best_task2_lowcost_answer_only` | current best Task 1 artifact | `data/outputs/competitions/private_task2_lowcost_answer_only.jsonl` | Task 2 answer-only prompt | pending validation |
| C | `hybrid_task1_lowcost_ablation_task2_best` | next low-cost Task 1 ablation artifact | current best Task 2 artifact | Task 1 top-k/retrieval ablation | pending ablation |

## Uploaded Ledger

Move a candidate here only after the zip exists and its SHA256 was printed by
the packaging command or the verification snippet above.

| date | candidate | Task 1 artifact | Task 2 artifact | zip path | sha256 | Codabench Task 1 F2 | Codabench Task 2 accuracy | internal validation source | notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| TBD | `hybrid_task1_lowcost_task2_best` | `data/outputs/competitions/private_task1_lowcost_t10_i5_o3.jsonl` | `data/outputs/competitions/private_task2_best.jsonl` | `data/outputs/submissions/submission_hybrid_task1_lowcost_task2_best.zip` | TBD | TBD | TBD | Task 1 validation ablation + current best Task 2 metrics | isolates Task 1; fill only after packaging/upload |

## Score Notes

Record Codabench output exactly as displayed. If Codabench reports only an
overall score or hides per-subtask values, write `unavailable` in the missing
columns and attach a screenshot or copied result note outside Git if needed.

Use this format for manual notes:

```text
YYYY-MM-DD HH:MM ICT
candidate:
zip_sha256:
Task 1 F2:
Task 2 accuracy:
Codabench run ID / URL:
What changed:
Decision:
```

## Decision Log

| date | decision | reason | owner |
| --- | --- | --- | --- |
| 2026-06-30 | Start with candidate A before changing Task 2. | Isolates low-cost Task 1 retrieval from answer-only VLM experiments. | M1/M4 |
