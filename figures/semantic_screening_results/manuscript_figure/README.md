# Semantic screening results manuscript figure

The manuscript figure presents the locked pure-semantic result as a four-step
evidence chain:

```text
all-candidate rank migration
→ axis-specific score separation
→ all-candidate equal-weight semantic aggregation
→ rank-1 anchor provenance and candidate-pool anchor context
```

Generate and verify the version-local data bundle:

```bash
python figures/semantic_screening_results/manuscript_figure/prepare_figure_data.py prepare
python figures/semantic_screening_results/manuscript_figure/prepare_figure_data.py verify
```

Then render the figure:

```bash
python figures/semantic_screening_results/manuscript_figure/plot_results.py
```

The plotting script reads only `manuscript_figure/data/`. The complete composite
is written to `manuscript_figure/images/`; separately editable panels a–d are
written to `manuscript_figure/images/panels/` from the same plotting functions.

All ranking panels use the complete set of 200 Stage-2 candidates. The main
ranking is pure semantic: exact score ties use anonymous candidate ID. No
structural-similarity quantity is read or displayed.
