# Figure contract: semantic screening results v2

Core conclusion:
The three-axis semantic stage substantially reorders the anchor-prescreened
candidate set and consistently prioritizes 3-cyanotetrahydrofuran across the
NCM811, lithium-metal and LHCE axes.

Figure archetype:
Asymmetric quantitative grid with a two-layer hero panel.

Target journal/output:
Nature-family manuscript figure; editable SVG is the primary output, with PDF,
3200-dpi TIFF and 300-dpi PNG companions.

Backend:
Python (matplotlib) only.

Final size:
183 mm × 190 mm.

Panel map:

- a, upper: Stage 1 rank versus deterministic final semantic rank for all 200
  candidates retained by the prescreen. The no-change diagonal and final
  Top-20 region make rank migration explicit.
- a, lower: Four synchronized 200-candidate heat strips ordered by final rank,
  showing the NCM811, lithium-metal and LHCE percentiles and their equal-weight
  semantic mean.
- b: Three-axis percentile matrix for the final Top 20, with final-rank labels
  1–20 placed directly on the heatmap y axis and exact axis ranks in the cells.
  The first row is emphasized without repeating the material name already shown
  in panel a. The complete identity mapping is reported in Source Data.
- c: Three-axis cosine-similarity profile of the rank-1 candidate against all
  ten reference additives; the two anchors used per axis are outlined.
- d: Pairwise Spearman rank concordance among the NCM811, lithium-metal and
  LHCE semantic axes for all 200 candidates.

Evidence hierarchy:

- Hero evidence: Stage 1-to-final rank migration and the complete four-row
  semantic fingerprint landscape.
- Validation evidence: Top-20 axis matrix and rank-1 anchor profile.
- Control/robustness: pairwise agreement among the three semantic axes.

Statistics:
All 200 Stage-2 candidates are shown in panels a and d. Spearman correlations
are descriptive. No inferential test or uncertainty interval applies because
the source is one deterministic locked screening run.

Source data:
Only the publication-facing derived CSV files in the common `data/` directory
are read. The plotting script does not access formal run directories or private
identity files.

Image-integrity notes:
No observations are excluded. Panel b and panel c are explicitly labelled
subsets. Vector text remains editable, and raster outputs retain a white
background.

Reviewer risks addressed:

- Stage 1 is visibly separated from the semantic reranking stage.
- Identity annotation occurs after deterministic ranking.
- Three semantic axes remain equally weighted.
- Exact semantic-score ties are resolved only by anonymous `candidate_id`.
- Structural similarity is isolated in a separate exploratory diagnostic and
  does not enter the publication mainline.

Adaptation record:
Panel a is rebuilt for a new scientific question. Panels b–d inherit the v1
visual language; panel b uses the deterministic final rank as its y axis, and
panel d is rebuilt as a semantic-axis concordance diagnostic.
Panels b and c have a reserved inter-panel gutter that prevents axis labels from
entering adjacent data areas.
