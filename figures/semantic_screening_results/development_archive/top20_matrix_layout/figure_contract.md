# Figure contract: semantic screening results v3

Core conclusion:
The three-axis semantic stage substantially reranks the frozen prescreened set
and identifies one candidate that ranks first under each of the three
context-conditioned semantic criteria.

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

- **a:** Near-square Stage 1 rank versus deterministic pure-semantic rank for
  all 200 retained candidates. Equal physical x and y scales prevent visual
  distortion of the 141-to-1 rank migration, and symmetric margins keep the
  rank-1 marker fully visible.
- **b:** Three compact, axis-specific candidate strips showing all 200 raw
  Top-2 anchor scores, the interquartile range, median, ranks 1 and 2, and their
  raw score gap. Axis-specific x ranges avoid implying that raw cosine values
  are directly calibrated across contexts.
- **c:** Compact Top-20 three-axis percentile matrix. Rows show final rank and
  anonymous candidate ID; cell text reports within-axis rank.
- **d:** The rank-1 candidate's full 3 × 10 anchor-cosine profile and the global
  3 × 10 frequency with which each anchor enters the Top-2 across all 200
  candidates. Both matrices span the panel width; no repeated molecular example
  card is shown.

Evidence hierarchy:

- Hero evidence: panel a establishes the rank migration caused by the semantic
  stage.
- Primary score evidence: panel b shows that the selected candidate occupies
  rank 1 under all three context-conditioned criteria and quantifies its
  separation from rank 2 without a density model.
- Neighbourhood evidence: panel c shows the cross-axis composition of the
  deterministic Top 20.
- Score provenance and reviewer control: panel d exposes all selected-candidate
  similarities and places its SN/ADN Top-2 pattern within global anchor-use
  frequencies rather than presenting only favourable anchors.

Statistics:
Panels a and b contain all 200 Stage-2 candidates. Summaries are descriptive;
no inferential test, uncertainty interval or replicate-based statistic applies
to this single deterministic screening run.

Source data:
The version-local data-preparation script reads only publication-facing
pure-semantic tables in the parent figure-data directory. It does not access
model weights, formal run directories or private identity files.

Image-integrity notes:
No observations are sampled or excluded. Panel c is the deterministic Top 20;
the panel-d individual profile is added only after the anonymous rank-1
candidate is established. SVG text remains editable.

Delivery:
The complete four-panel composition and the four independently editable panels
are exported from the same plotting functions; the individual panels are not
raster crops of the composite.

Reviewer risks and controls:

- Do not imply that chemical identity was used during ranking.
- Do not describe the three axes as statistically independent; they are
  distinct context-conditioned criteria and are strongly correlated.
- Use raw axis scores in panel b and separate x ranges by axis.
- Show all ten anchors plus global anchor-use frequency in panel d.
- Do not introduce structural similarity, ECFP or any chemical post-ranking
  correction.
- Use the reference image only for low-saturation rose–blue–violet style
  inheritance; do not transfer its signed electrostatic-potential colour
  semantics to monotonic score heatmaps.

Adaptation record:
Build anew from the confirmed scientific contract. The reference image informs
only the restrained palette; its data encoding is not reused. The earlier v3
panel geometry and KDE representation are not retained.
