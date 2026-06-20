## Linked issue

Closes #

Plan key: `W?-??`

## What changed

-

## Files and ownership

- Files changed:
- Shared files changed and assigned issue:
- Reference implementations consulted:

## Verification

```bash
python -m pytest -q
```

Evidence or metric artifact:

## Research checks

- [ ] Train and validation remain disjoint by `image_id`.
- [ ] Validation labels were not used in prompts, retrieval examples, or training.
- [ ] Metrics use the locked split and record config, seed, and split hash.
- [ ] Failures are counted and are not silently replaced by a default answer.
- [ ] Citations resolve to real LawDB identifiers when this PR emits citations.

## Repository checks

- [ ] Tests cover behavior changes.
- [ ] No raw data, secrets, weights, embeddings, or local caches are committed.
- [ ] Documentation/configuration is updated where behavior changed.
- [ ] CPU-safe tests pass even if the full path needs a GPU.
- [ ] The PR is focused enough for one reviewer to verify.

## UI changes

Screenshot or recording when applicable:
