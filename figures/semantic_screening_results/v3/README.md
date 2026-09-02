# Semantic screening results figure v3

Version 3 presents the screening result as a four-step evidence chain:

```text
near-square rank migration
→ axis-specific candidate score separation
→ compact Top-20 semantic profiles
→ rank-1 anchor provenance and global anchor context
```

Generate the version-local data bundle first:

```bash
python figures/semantic_screening_results/v3/prepare_v3_data.py prepare
python figures/semantic_screening_results/v3/prepare_v3_data.py verify
```

Then render the figure:

```bash
python figures/semantic_screening_results/v3/plot_results.py
```

The plotting script reads only `v3/data/`. The complete composite is written to
`v3/images/`; separately editable panels a–d are written to
`v3/images/panels/` from the same plotting functions.
The main ranking is pure semantic: exact score ties use anonymous candidate ID.
No structural-similarity quantity is read or displayed.

The v3 data bundle additionally records the raw rank-1/rank-2 gap for each
semantic axis and the global frequency with which each reference additive enters
the Top-2 anchor pair. These are descriptive views of the locked semantic
result and do not modify ranking.
