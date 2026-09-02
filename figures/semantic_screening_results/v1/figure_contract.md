# Figure contract: semantic screening results v1

Core conclusion:
Context-centred semantic fingerprints consistently prioritize the same candidate
across the NCM811-interface, lithium-metal-interface and LHCE-solvation axes,
yielding a deterministic rank-1 candidate without relying on the structural
tie-breaker.

Figure archetype:
Asymmetric quantitative grid with one full-width hero ranking panel.

Target journal/output:
Nature-family manuscript figure; editable SVG is the primary output, with PDF,
600-dpi TIFF and 300-dpi PNG companions.

Backend:
Python (matplotlib) only.

Final size:
183 mm × 160 mm.

Panel map:

- a: Complete semantic ranking of all 200 Stage-2 candidates, with the
  rank-1 candidate directly labelled and Top-20/Top-50 regions indicated.
- b: Three-axis percentile matrix for the final Top 20, with the rank-1 row
  emphasized typographically.
- c: Three-axis cosine-similarity profile of the rank-1 candidate against all
  ten reference additives; the two anchors used per axis are outlined.
- d: Structural score versus semantic score for all 200 candidates, separating
  candidates in exact semantic ties from non-tied candidates and showing that
  the rank-1 candidate did not require structural tie-breaking.

Evidence hierarchy:

- Hero evidence: complete 200-candidate semantic ranking.
- Validation evidence: balanced Top-20 three-axis percentiles and the selected
  candidate's anchor-level similarity profile.
- Control/robustness: structure–semantic relationship and exact-tie membership.

Statistics:
All 200 Stage-2 candidates are shown in panels a and d. Spearman correlation is
reported in panel d. No inferential statistical test or uncertainty interval is
applicable because the figure summarizes one deterministic locked screening
run rather than replicate measurements.

Source data:
Only the publication-facing derived CSV files in this directory are read.
Figure code does not access the formal run directories or identity registry.

Image-integrity notes:
No observations are excluded. Top-20 and rank-1 subsets are explicitly labelled
as subsets. SVG text remains editable; raster exports use a white background.

Reviewer risks addressed:

- Candidate identity is annotation after deterministic ranking, not an input.
- All three axes have equal weight in the semantic score.
- ECFP does not enter the semantic score and is used only inside exact semantic
  ties.
- Heatmap colour limits and sample counts are stated in the figure legend.
