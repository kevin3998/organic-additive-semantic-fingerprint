# Structural-similarity diagnostic

This directory preserves the ECFP-based structure–semantic comparison as an
exploratory diagnostic. It is intentionally separated from the publication
mainline.

Scientific status:

- The main ranking is determined by the equal-weight three-axis semantic score.
- Exact semantic-score ties are resolved deterministically by anonymous
  `candidate_id`.
- Structural similarity does not enter the semantic score, candidate selection,
  Top-20 membership or Top-50 membership.
- The selected candidate is the unique semantic rank 1.
- Removing the historical structural tie-break changes only the internal order
  of some exact-tie groups; it does not change the Top 20 or Top 50 candidate
  sets.

The archived diagnostic retains the historical Morgan/ECFP definition
(`radius=2`, `2048` bits, chirality disabled), the Top-2 anchor aggregation and
the structure–semantic Spearman comparison so the idea can be revisited later.

Generate the standalone archive:

```bash
python figures/semantic_screening_results/structural_similarity_diagnostic/plot_structural_diagnostic.py
```

Outputs:

```text
data/structural_similarity_diagnostic.csv
data/diagnostic_summary.json
images/structural_similarity_diagnostic.svg
images/structural_similarity_diagnostic.pdf
images/structural_similarity_diagnostic.tiff
images/structural_similarity_diagnostic.png
```

These outputs are not inputs to the main manuscript figure.

The plotting script is self-contained within this archive: it reads only
`data/structural_similarity_diagnostic.csv` and
`data/diagnostic_summary.json`. It does not read the current pure-semantic
publication tables.

`historical_design/` preserves the superseded workflow design and its generation
prompt that showed ECFP as a tie-only branch. They are retained for provenance
and must not be used as the current method figure.
