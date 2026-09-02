# Semantic screening results figure v4

Version 4 presents the locked pure-semantic result as a four-step evidence
chain:

```text
all-candidate rank migration
→ axis-specific score separation
→ all-candidate equal-weight semantic aggregation
→ rank-1 anchor provenance and candidate-pool anchor context
```

Generate and verify the version-local data bundle:

```bash
python figures/semantic_screening_results/v4/prepare_v4_data.py prepare
python figures/semantic_screening_results/v4/prepare_v4_data.py verify
```

Then render the figure:

```bash
python figures/semantic_screening_results/v4/plot_results.py
```

The plotting script reads only `v4/data/`. The complete composite is written to
`v4/images/`; separately editable panels a–d are written to
`v4/images/panels/` from the same plotting functions.

All ranking panels use the complete set of 200 Stage-2 candidates. The main
ranking is pure semantic: exact score ties use anonymous candidate ID. No
structural-similarity quantity is read or displayed.

