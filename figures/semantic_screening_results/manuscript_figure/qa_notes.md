# Figure QA notes

- Backend: Python/matplotlib only.
- Reuse level: structural adaptation of the v3 quantitative grid.
- Final canvas: 183 mm × 150 mm.
- Primary export: editable SVG; companion PDF, 3200-dpi TIFF and 300-dpi PNG.
- The complete composite and independently editable panels a–d are exported
  from shared plotting functions; panel files are not raster crops.
- Data counts:
  - panel a: all 200 Stage-2 candidates;
  - panel b: 600 records, corresponding to 200 candidates on each of three axes;
  - panel c: all 200 candidate trajectories across three percentiles and their
    equal-weight semantic score;
  - panel d: one post-ranking candidate × three axes × ten anchors, plus 30
    complete candidate-pool anchor-occurrence values.
- No observations were sampled or excluded.
- Panel a uses equal physical x/y scaling and symmetric rank-space margins. The
  selected candidate is shown only as an orange marker without text or leader.
- Ordinary panel-a points are intentionally darker than panel-b points because
  the global rank cloud is primary evidence; both remain subordinate to rank 1.
- Panel b uses no KDE, smoothing, uncertainty interval or inferential summary.
  Deterministic point displacement affects display only; x values are unchanged.
- All three panel-b facets use the common range 0.865–0.985 and identical ticks;
  this prevents facet rescaling from exaggerating within-axis separation.
- Rank-1 minus rank-2 raw score gaps are 0.0075 (NCM811), 0.0083 (Li metal)
  and 0.0048 (LHCE).
- Panel c directly verifies that semantic score is the arithmetic mean of the
  three axis percentiles to absolute tolerance 1e-12. All 200 trajectories are
  drawn; ordinary lines are transparent but not sampled. Rank 1 is encoded as
  a dark-coral line without circular markers, with the same width as every other
  trajectory so colour is the only emphasis channel.
- Panel a transforms both source ranks with `(200 - rank) / 199 × 100%`. Its
  axes therefore share the same increasing direction; the upper-right corner
  denotes stronger ranking in both stages and the pale band marks the final
  Top 20 at percentiles of at least 90.45%.
- Composite panel headers use fixed figure-level baselines with a protected top
  margin; the two titles in each row are vertically aligned and remain inside
  the fixed 183 mm × 150 mm canvas.
- Panel d reports all ten selected-candidate similarities. Its cosine range
  (0.8971–0.9827) lies within the displayed colour scale, so no values clip.
- Panel d uses equal-height 3 × 10 matrices with a compact inter-matrix gap.
  The lower colour bar uses a short, raised `count` title and reserved right
  margin so its title does not overlap the maximum tick label.
- In the composite, the panel-d container shares the exact top and bottom bounds
  of panel c and is widened across the available right column. Both heatmaps use
  the added vertical extent rather than reintroducing inter-matrix whitespace.
- Each panel-d axis has exactly two outlined anchors; their mean reproduces the
  corresponding panel-b axis score.
- The lower panel-d counts sum to 400 per axis, corresponding to two Top-2
  positions for each of 200 candidates. The label explicitly limits the result
  to this candidate pool.
- Candidate identity is used only for post-ranking annotation.
- Structural similarity, ECFP and chemical tie-breaking are absent from the
  data bundle and plotting script.
- The axes are context-conditioned and strongly correlated; they are not
  described as statistically independent.
- No experimental replicates, uncertainty intervals or inferential tests apply
  to this single deterministic screening run.
