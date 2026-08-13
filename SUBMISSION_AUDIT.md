# Submission Audit

Audit date: 2026-08-13. Scope: this clean submission copy only. The working experiment repository was not cleaned or modified in place.

## Final tree

```text
FINAL_SUBMISSION_REPO/
├── src/{face,speech,numerical,fusion,api}/
├── frontend/
├── configs/
├── scripts/
├── results/{face,speech,numerical,fusion}/
├── docs/
├── tests/
├── models/                 # checkpoint placement instructions; binaries excluded
├── README.md
├── requirements.txt
└── .gitignore
```

## Selected models

- Face: ConvNeXt-Tiny real-only, selected at epoch 22 by validation Macro F1.
- Speech: emotion2vec+ with attentive statistics pooling under an actor-independent split.
- Numerical: original-real baseline for the primary real-data result; synthetic benchmarks remain secondary.
- Fusion: gated modality-attention, multi-task fusion over face, speech, and numerical embeddings.

## Authoritative metrics

The sole authority is [results/FINAL_RESULTS.md](results/FINAL_RESULTS.md). Primary test Accuracy / Macro F1: face 0.6896 / 0.6852; speech 0.7583 / 0.7405; numerical original baseline 0.2617 / 0.2245; weakly aligned fusion 0.6633 / 0.5224. Random-split and synthetic-only results are explicitly labeled secondary benchmarks.

## Files intentionally excluded

- Raw FER-style, RAVDESS, and numerical datasets.
- Generated synthetic datasets and augmentation caches.
- Checkpoints and embedding caches that are too large for an ordinary GitHub repository.
- Failed, stopped, historical, and superseded experiment implementations and outputs.
- Dependency folders, build outputs, logs, temporary files, local environment files, and secrets.

## Verification status

- Python syntax/import checks: passed.
- Result consistency and protocol-boundary tests: passed.
- Model output-shape smoke test: passed.
- API health/metrics smoke test: passed.
- Frontend TypeScript production build: passed; Vite reported only a non-fatal chunk-size warning.
- Secret scan: passed; no credential-like values found.
- Broken-path audit: passed for required repository paths and local Markdown links.
- Stale-result audit: passed; no failed-model references, obsolete metrics, training-status flags, or private absolute paths remain.
- Large-file audit: passed; no file exceeds 95 MB.

## Unresolved limitations

- Large trained checkpoints are not committed; users must place them under `models/` or retrain.
- The included API exposes health and authoritative metrics, not the complete uploaded-media inference workflow.
- Fusion is implemented and evaluated, but its inputs were weakly class-conditionally aligned across independent datasets—not participant-paired.
- The frontend demo is not a clinical device, and the reported results do not establish deployment safety or population-level generalization.
- The production JavaScript bundle is about 600 kB before gzip and would benefit from route-level code splitting.
