# Organic Additive Semantic Fingerprint v4 Implementation

## Completed Components

- `project_config/organic_additive_screening_v4.json` freezes the source bundle,
  prompts, context-centered embedding method, semantic ranking, structural tie
  breaker, model revision, tokenizer metadata, and six weight hashes.
- `semantic_fingerprint/v4/molecular_card.py` creates deterministic structure
  cards from canonical SMILES.
- `semantic_fingerprint/v4/backend.py` extracts final-layer prompt embeddings
  without generation or label scoring.
- `semantic_fingerprint/v4/pipeline.py` centers and normalizes embeddings,
  computes top-two anchor cosine scores, ranks the three axes, applies the exact
  semantic-tie rule, and builds anonymous diagnostics.
- `semantic_fingerprint/v4/run_control.py` verifies and copies the frozen source
  bundle, freezes a run, resumes embedding batches, locks outputs, and safely
  reveals the target.
- `scripts/run_organic_screening_v4.py` exposes `import-source`, `freeze`,
  `screen`, and `reveal-target`.
- `tests/v4/` covers configuration boundaries, molecular cards, embedding
  extraction, scoring, import isolation, interruption recovery, immutable locks,
  topology attacks, and end-to-end execution with a deterministic fake backend.

## Execution Contract

1. Prepare public and private source-bundle roots whose files match the frozen
   SHA256 values in the v4 configuration.
2. Run `import-source` into a new public/private run ID. Existing destinations
   are never overwritten.
3. Run the real MPS/float16 smoke test and freeze with batch size 8, or batch
   size 4 only after a documented smoke-stage MPS OOM.
4. Run `screen`. Resume only when manifest, source, model, prompt, shape, entity
   order, and batch metadata match exactly.
5. Verify the 3x211 float32 embeddings, complete bitmap, 200 score rows, 50 Top
   rows, finite diagnostics, manifest, and result lock.
6. Run `reveal-target` only after the public lock exists.

## Formal Run

The retained run is `20260724-organic-additive-v4-01`. Its scientific outputs
are complete and must not be rerun or tuned. The target pure semantic rank and
final rank are both 1.
