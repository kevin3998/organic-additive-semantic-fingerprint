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

## Figure versions

The first complete Nature-style result figure is preserved under `v1/`. It
reads only the derived files under `data/`:

```bash
python figures/semantic_screening_results/v1/plot_results.py
```

Its plotting contract, legend draft, QA record and exports are kept together in
the version directory:

- `v1/figure_contract.md`;
- `v1/figure_legend.md`;
- `v1/qa_notes.md`;
- `v1/images/`.

The v1 exports preserve the earlier visualization design as a historical
snapshot. The current publication data and active figure are pure-semantic.

Version 2 is stored under `v2/` and rebuilds the hero panel as a Stage 1-to-final
rank migration map plus a four-row semantic fingerprint landscape:

```bash
python figures/semantic_screening_results/v2/plot_results.py
```

The v2 TIFF export is fixed at 3200 dpi; SVG and PDF remain the preferred
resolution-independent manuscript files.

The former ECFP/Tanimoto comparison has been moved in full to
`structural_similarity_diagnostic/`. It is an exploratory archive and is not
read by the current data-preparation or plotting paths.

Version 3 is preserved as the last Top-20-matrix Results layout:

```bash
python figures/semantic_screening_results/v3/prepare_v3_data.py prepare
python figures/semantic_screening_results/v3/plot_results.py
```

Its four panels show Stage 1-to-semantic rank migration, the three raw
axis-score distributions, the Top-20 semantic matrix and the rank-1 candidate's
complete anchor profile.

Version 4 is the current Results figure. It replaces the arbitrary Top-20
matrix with an all-candidate equal-weight aggregation view and strengthens the
visibility of the rank and score distributions:

```bash
python figures/semantic_screening_results/v4/prepare_v4_data.py prepare
python figures/semantic_screening_results/v4/plot_results.py
```

Panel c contains all 200 candidates and connects the three within-axis
percentiles to their equal-weight semantic score. Panel d reports score
provenance and anchor occurrence only within the same 200-candidate pool.
