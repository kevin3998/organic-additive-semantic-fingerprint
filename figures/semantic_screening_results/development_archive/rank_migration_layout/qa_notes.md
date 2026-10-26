# Figure QA notes — v2

## Scientific integrity

- Data source: publication-facing derived tables from the locked v4 screening
  result.
- Complete observations: panel a contains all 200 candidates retained from the
  800-candidate Stage 1 screen; panel d contains the same 200 candidates.
- Explicit subsets: panel b contains the final Top 20 and panel c contains the
  deterministic rank-1 candidate across three axes and ten anchors.
- Exclusions: none.
- Rank mapping: `stage1_all.csv.stage1_rank` is joined one-to-one to the
  deterministic final ranking by `candidate_id`.
- Transformations: no smoothing, binning, sampling or stochastic jitter.
- Candidate annotation: chemical identity is attached only after deterministic
  ranking and does not alter the rank migration or semantic score.
- Top-20 identities: PubChem names are connectivity-checked against the ranked
  canonical SMILES and retained in the companion source-data table. Panel b
  uses final ranks 1–20 directly as y-axis ticks and does not repeat material
  names already identified elsewhere. Identity metadata do not participate in
  scoring or ranking.
- Deterministic ties: exact semantic-score ties use anonymous `candidate_id`
  only; no additional chemical score enters the publication ranking.
- Structural diagnostic: moved intact to the separate
  `structural_similarity_diagnostic/` archive and excluded from the main plot.
- Statistics: Spearman correlations are descriptive. No replicate, interval or
  inferential test applies to this deterministic locked run.

## Adaptation record

- Reuse level: panel a was rebuilt; panels b and c inherit v1 style only;
  panel d was rebuilt as a semantic-axis concordance diagnostic.
- The v1 directory and exports remain unchanged.

## Visual and export QA

- Backend: Python/matplotlib exclusively.
- Final canvas: 183 mm × 190 mm.
- Minimum explicit text size: 5 pt.
- Palette: restrained teal, violet and neutral grey; no rainbow mapping and no
  red–green-only encoding.
- Panel labels: lowercase bold a–d.
- SVG: editable text retained; 164 text nodes in the final export.
- PDF: editable TrueType text configuration enabled.
- PNG preview: 2160 × 2244 px at 300 dpi.
- TIFF: 23040 × 23936 px at 3200 dpi with LZW compression.
- Final TIFF size: approximately 59.9 MB.
- Visual inspection: performed at final width; panels a and b share the same
  left boundary, the rank-1 annotation is integrated into panel a, Top-20
  ranks are aligned directly to heatmap rows, a reserved gutter separates
  panels b and c, and panel d contains only semantic-axis information.
- Static source preflight: 14 checks passed, 0 warnings, 0 failures.

## Reproducibility

Run:

```bash
python figures/semantic_screening_results/development_archive/rank_migration_layout/plot_results.py
```

The script validates the 800→200 Stage 1 selection, complete Stage 1 and final
ranks, one-to-one rank mapping, Top-20 membership, selected-candidate placement,
pure-semantic deterministic ordering and all finite plotting values before
rendering.
