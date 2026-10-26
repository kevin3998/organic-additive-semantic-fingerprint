# TASK PLAN

## Goal

Maintain one complete, publication-facing organic-additive screening implementation.

## Retained Workflow

1. Verify and import the version-neutral anonymous candidate, anchor, and Stage 1 source bundle.
2. Freeze configuration, code, environment, model, prompts, entity order, and smoke evidence.
3. Encode one baseline, ten anchors, and 200 candidates on each of three axes.
4. Subtract each axis baseline, L2-normalize, and compute candidate-anchor cosines.
5. Average the two highest anchor similarities per axis.
6. Convert axis ranks to percentiles and equally average the three semantic axes.
7. Order by semantic score; use anonymous candidate ID only for exact ties.
8. Publish immutable public results before private target reveal.

## Retained Files

- `project_config/organic_additive_screening.json`
- `semantic_fingerprint/screening/`
- `scripts/run_organic_screening.py`
- `tests/screening/`
- `docs/superpowers/plans/organic-additive-semantic-fingerprint-implementation.md`
- `docs/superpowers/specs/organic-additive-semantic-fingerprint-method.md`
- `environment.yml`

## Frozen Result

- Run ID: `20260724-organic-additive-v4-01`
- Candidate: 3-cyanotetrahydrofuran
- Stage 1 rank: 141
- NCM811 semantic rank: 1
- Lithium-metal semantic rank: 1
- LHCE-solvation semantic rank: 1
- Pure semantic rank: 1
- Final rank: 1

The formal result is complete and must not be rerun or tuned.
