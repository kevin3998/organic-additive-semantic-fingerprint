# Figure QA notes — v3

- Backend: Python/matplotlib only.
- Reuse level: build anew from the confirmed panel contract; prior output
  contributes style-level palette and typography only.
- Final canvas: 183 mm × 150 mm.
- Primary export: editable SVG; companion PDF, 3200-dpi TIFF and 300-dpi PNG.
- The complete composite and independently editable panels a–d are exported
  from shared plotting functions; the panel files are not raster crops.
- Data counts:
  - panel a: all 200 Stage-2 candidates;
  - panel b: 600 records, corresponding to 200 candidates on each of three axes;
  - panel c: deterministic final ranks 1–20;
  - panel d: one post-ranking selected candidate × three axes × ten anchors,
    plus 30 global anchor-frequency values.
- No observations were sampled or excluded.
- Panel a uses a near-square plotting box and equal physical x/y scaling.
- Panel a includes symmetric rank-space margins so the rank-1 marker is fully
  visible rather than clipped by the upper plotting boundary.
- Panel b uses no KDE, smoothing, uncertainty interval or inferential summary.
  Point displacement is deterministic and affects display only; quantitative
  x positions are unchanged.
- Panel b has separate raw-score ranges for the three context-conditioned axes;
  visual distances should be interpreted within, not across, facets.
- Rank-1 minus rank-2 raw score gaps are 0.0075 (NCM811), 0.0083 (Li metal)
  and 0.0048 (LHCE).
- Panel c uses the full observed percentile range (minimum 0.8342) and labels
  every row with final rank plus anonymous candidate ID.
- Panels a and c share the same left and right boundaries; the c–d gutter is
  6.4 mm at the 183-mm final width.
- Panel d reports all ten selected-candidate similarities and the global Top-2
  anchor frequency across 200 candidates, preventing favourable-anchor-only
  presentation.
- Candidate identity is used only for post-ranking annotation. Panel d contains
  no molecular example card or chemical-structure depiction.
- Structural similarity, ECFP and chemical tie-breaking are absent from the
  data bundle and plotting script.
- The axes are distinct context-conditioned criteria but are strongly
  correlated (Spearman ρ = 0.912–0.950); they are not described as
  statistically independent.
- The rose–blue–violet palette is style-only inheritance from the provided
  scientific reference image. Its original red–blue electrostatic-potential
  semantics are not transferred to these monotonic screening scores.
- Pale helper text was removed from the figure. Candidate count, identity
  timing, cell-text definition and Top-2 outline meaning are documented in the
  legend instead.
- Sequential heatmaps remain interpretable in greyscale through cell text,
  row/column position and outlined Top-2 cells. Cell text colour is selected
  from rendered luminance rather than fixed score thresholds.
- No experimental replicates, uncertainty intervals or inferential tests apply
  to this single deterministic screening run.
