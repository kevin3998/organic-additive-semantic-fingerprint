# Semantic screening results figure v2

Version 2 replaces the original monotonic rank curve with a two-layer hero
panel:

1. Stage 1 rank versus final semantic rank for all 200 retained candidates;
2. four final-rank-ordered heat strips for the three axis percentiles and their
   equal-weight semantic mean.

The shared publication data remain in `../data/`. Run from the repository root:

```bash
python figures/semantic_screening_results/v2/plot_results.py
```

Panel b uses final ranks 1–20 directly as its y axis and does not repeat
material names already identified in panel a. The complete rank-to-identity
mapping is retained in `data/top20_identity_annotations.csv`; these post-ranking
annotations are not used by the scoring or ranking implementation.

The publication mainline uses the equal-weight semantic score and resolves
exact score ties by anonymous `candidate_id`. The former ECFP diagnostic has
been moved to `../structural_similarity_diagnostic/` and is not read by this
plotting script.

Outputs are written to `v2/images/` as editable SVG, PDF, 3200-dpi TIFF and a
300-dpi PNG preview.
