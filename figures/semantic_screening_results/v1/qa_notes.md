# Figure QA notes — v1

## Scientific integrity

- Data source: publication-facing derived tables from the locked v4 screening
  result.
- Complete observations: panels a and d contain all 200 Stage-2 candidates.
- Explicit subsets: panel b contains the final Top 20; panel c contains the
  deterministic rank-1 candidate across three axes and ten anchors.
- Exclusions: none.
- Candidate annotation: chemical identity is attached only after deterministic
  ranking and is not used by the plotting script to recompute or change ranks.
- Structural information: structural score is plotted diagnostically and is
  used by the method only inside exact semantic-score ties.
- Statistics: Spearman correlation is descriptive. There are no replicate
  measurements, error bars or inferential tests because the source is one
  deterministic locked run.

## Visual and export QA

- Backend: Python/matplotlib exclusively.
- Final canvas: 183 mm × 160 mm.
- Minimum explicit text size: 5 pt.
- Colour: restrained teal/violet/rose palette; no rainbow map and no
  red–green-only distinction.
- Panel labels: lowercase bold a–d.
- SVG: editable text retained; 158 SVG text nodes in the final export.
- PDF: editable TrueType text configuration enabled.
- PNG preview: 2160 × 1887 px at 300 dpi.
- TIFF: 4320 × 3774 px at 600 dpi with LZW compression.
- Visual inspection: performed at the final 183-mm width; no clipped labels,
  panel collisions or hidden data marks were observed.
- Static source preflight: 14 checks passed, 0 warnings, 0 failures.

## Reproducibility

Run:

```bash
python figures/semantic_screening_results/plot_results.py
```

The script validates row counts, complete final ranks, the Top-20 subset,
selected-candidate placement, semantic-tie membership and finite plotting
values before rendering.
