# PROGRESS

## Current State

- Active implementation: organic additive semantic fingerprint v4 only.
- Branch: `main`.
- Formal run: `20260724-organic-additive-v4-01`.
- Formal result: target pure-semantic rank 1; `semantic_top15=true`.
- The locked public and private v4 run directories remain under `runs_v4/` and
  `private_runs_v4/`.

## Completed

- Implemented context-centered ChemDFM hidden-vector embeddings on three axes.
- Implemented Top-2 reference-anchor cosine scoring and equal axis aggregation.
- Fixed the main ranking to pure semantic score, with anonymous candidate ID as
  the deterministic fallback for exact ties.
- Moved the former Morgan/ECFP comparison to an isolated exploratory archive.
- Implemented resumable embedding checkpoints, immutable result locking, and safe
  private reveal.
- Executed and independently verified the first formal v4 run.
- Moved deterministic molecular-card generation into the v4 package.
- Reduced the repository, worktrees, branches, tests, and run products to one
  self-contained v4 implementation.
- Replaced version-coupled provenance with a fixed, version-neutral source bundle.

## Preservation Boundary

- Do not rerun or modify the locked v4 formal result.
- Preserve the frozen candidate, anchor, Stage 1, molecular-card, model, and
  prompt hashes that define the formal v4 run.
- ChemDFM weights remain external at
  `/Users/chenlintao/models/ChemDFM-v2.0-14B`.

## Verification

- Run `conda run -n organic-semantic-fingerprint python -m pytest -q tests/v4`.
- Run `python -m compileall -q project_config semantic_fingerprint/v4 scripts tests/v4`.
- Run `git diff --check` and confirm that only v4 scientific paths exist.
