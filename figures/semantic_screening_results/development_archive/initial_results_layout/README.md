# Semantic screening results figure v1

This directory preserves the first complete Nature-style visualization of the
semantic-screening results before the rank-transition redesign.

## Contents

- `plot_results.py`: reproducible Python/matplotlib plotting source;
- `figure_contract.md`: scientific claim, evidence hierarchy and panel map;
- `figure_legend.md`: manuscript legend draft;
- `qa_notes.md`: source, export and visual-QA record;
- `images/`: SVG, PDF, 600-dpi TIFF and PNG exports.

The implementation reads only the common publication-facing data stored in
`../data/`. It does not read the formal screening run or private identity files.

Run from the repository root:

```bash
python figures/semantic_screening_results/development_archive/initial_results_layout/plot_results.py
```

The default outputs are written to
`figures/semantic_screening_results/development_archive/initial_results_layout/images/`.
