# Semantic Screening Results Data

This directory prepares publication-facing source data from the immutable formal
run `20260724-organic-additive-v4-01`.

The processing order is intentional:

1. verify the public manifest, result lock, scores, ranks, anchor cosines and
   Top 50;
2. create all anonymous derived tables and determine the candidate at
   `final_rank = 1`;
3. only then read the private identity record and annotate the selected
   candidate.

Identity information never participates in ranking. Generated public tables use
`is_selected_candidate`; the source field `role=target` is not propagated.

## Commands

Use the repository's scientific Python environment for CSV/JSON processing and
install the isolated workbook dependency when needed:

```bash
python -m pip install -r figures/semantic_screening_results/requirements.txt
python figures/semantic_screening_results/prepare_data.py prepare
python figures/semantic_screening_results/prepare_data.py verify
```

The default output is `figures/semantic_screening_results/data/`. Existing
verified output is treated as an idempotent success. Conflicting generated data
is rejected unless `--replace-generated` is explicitly provided.

CSV and JSON files are the canonical, hashed scientific data. The XLSX workbook
is a formatted Nature Source Data delivery and is validated cell-for-cell
against those canonical tables.

## Data layers

- `source_snapshot/`: exact Stage 1, anchor-cosine and molecular-card inputs
  required to audit the figures; the locked run hashes remain recorded without
  copying its superseded structure-bearing result tables;
- `anonymous/`: pure-semantic ranking, axis, anchor, tie and diagnostic tables without
  chemical identity annotation;
- `annotated/`: the rank-1 candidate annotation and its anchor profile;
- `semantic_screening_source_data.xlsx`: formatted submission workbook;
- `data_manifest.json`: source and output provenance.

The formal run directories, model weights and embeddings are never modified or
copied into this directory.

## Manuscript figure

The submission-ready Results figure is maintained under `manuscript_figure/`:

```bash
python figures/semantic_screening_results/manuscript_figure/prepare_figure_data.py prepare
python figures/semantic_screening_results/manuscript_figure/plot_results.py
```

Panel c contains all 200 candidates and connects the three within-axis
percentiles to their equal-weight semantic score. Panel d reports score
provenance and anchor occurrence only within the same 200-candidate pool.

Earlier layouts are retained under `development_archive/` for internal
traceability and are not part of the submission package. The former
ECFP/Tanimoto comparison remains under `structural_similarity_diagnostic/` as
an exploratory archive and is not read by the formal data or plotting paths.
