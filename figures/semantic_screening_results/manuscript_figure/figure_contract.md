# Figure contract: semantic screening results

Core conclusion:
The three-axis semantic stage substantially reranks the frozen prescreened set,
and the same candidate ranks first on every context-conditioned axis and after
their deterministic equal-weight aggregation.

Figure archetype:
Asymmetric quantitative grid.

Target journal/output:
Nature-family manuscript Results figure; editable SVG and PDF, 3200-dpi TIFF,
and 300-dpi PNG preview.

Backend:
Python (matplotlib) exclusively.

Final size:
183 mm × 150 mm.

Panel map:

- **a:** Near-square Stage 1 ranking percentile versus deterministic final
  semantic ranking percentile for all 200 retained candidates. Both axes run
  from 0% to 100%, so the upper-right corner denotes stronger ranking in both
  stages. The rank-1 observation is identified only by an orange marker.
- **b:** Three axis-specific candidate strips showing all 200 raw Top-2 anchor
  scores, the interquartile range, median, ranks 1 and 2, and their raw score
  gap. A common cosine-score scale is used across all three axes; comparisons
  remain descriptive because the axes are separately context-conditioned.
- **c:** Rotated parallel-coordinate view of all 200 candidates. Each trajectory
  connects the three axis percentiles to their equal-weight mean semantic score.
  The selected rank-1 trajectory uses the same line width as the background
  trajectories and is distinguished only by a dark-coral colour, without markers.
- **d:** The rank-1 candidate's complete 3 × 10 candidate-to-anchor cosine
  profile and the 3 × 10 number of Stage-2 candidates for which each anchor
  entered the Top-2. Both matrices span the panel width.

Evidence hierarchy:

- Panel a establishes the global reranking outcome.
- Panel b establishes rank-1 separation on each raw context-conditioned axis.
- Panel c makes the percentile transformation and equal-weight aggregation
  visible without an arbitrary Top-N cutoff.
- Panel d decomposes the selected candidate's axis scores and places its Top-2
  anchor pattern within the complete 200-candidate screening pool.

Statistics:
Panels a–c contain all 200 Stage-2 candidates. Panel d contains all ten anchors
for the selected candidate and all 400 Top-2 selections per axis. Summaries are
descriptive; no inferential test, uncertainty interval or replicate-based
statistic applies to this single deterministic screening run.

Source data:
The version-local data-preparation script reads only publication-facing
pure-semantic tables in the parent figure-data directory. It does not access
model weights, formal run directories or private identity files.

Image-integrity notes:
No observations are sampled or excluded. Candidate identity is attached only
after anonymous deterministic rank 1 is established. SVG text remains editable.

Delivery:
The complete four-panel composition and four independently editable panels are
exported from the same plotting functions; individual panels are not raster
crops of the composite.

Reviewer risks and controls:

- Do not imply that chemical identity was used during ranking.
- Panel-a percentiles are deterministic transforms of the displayed source
  ranks, using `(200 - rank) / 199 × 100%` on both axes.
- Do not describe the three axes as statistically independent.
- Raw cosine scores are compared only within an axis.
- Panel c uses all 200 candidates and only locked percentiles and semantic score.
- Panel d is score provenance and candidate-pool context, not mechanistic or
  experimental validation.
- “Anchor occurrence” refers only to the current 200-candidate Stage-2 pool;
  it is not a claim about global chemical space or anchor importance.
- Structural similarity, ECFP and chemical post-ranking correction are absent.

Adaptation record:
Structural adaptation of the v3 quantitative grid. Panels a, b and d retain
their data roles with revised emphasis; panel c is rebuilt for the all-candidate
equal-weight aggregation question.
