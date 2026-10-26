# Organic Additive Semantic Fingerprint Method

## Scope

This repository contains one method for screening neutral organic electrolyte
additives for a LiFSI/DME/TTE localized high-concentration electrolyte in a
high-voltage NCM811||Li-metal full cell.

The method ranks a frozen set of 200 Stage 1 candidates. It does not predict
additive concentration, measured electrochemical performance, or reaction
mechanisms. Exact semantic-score ties are resolved by anonymous candidate ID.

## Frozen Inputs

The version-neutral source bundle contains:

- 800 anonymous candidate IDs and canonical connectivity SMILES;
- 10 reference-additive anchors;
- Stage 1 scores and the fixed 200-candidate selection;
- Stage 1 embeddings retained for provenance;
- deterministic molecular cards for all candidates;
- a private target registry stored outside the public run directory.

The configuration fixes the SHA256 of every source file. `import-source`
rejects missing, modified, linked, overlapping, or incomplete source bundles.
No earlier screening implementation or ranking is an input to the retained method.

## Molecular Representation

Each candidate and anchor is represented by a deterministic molecular card
derived from canonical non-stereochemical SMILES. The card contains formula,
molecular weight, hydrogen-bond counts, TPSA, cLogP, ring and rotatable-bond
counts, and positive RDKit fragment descriptors.

## Context-Centered Embeddings

Three independent electrolyte-design axes are used:

1. oxidative and interphase compatibility with NCM811;
2. reductive and interphase compatibility with lithium metal;
3. moderate, non-disruptive compatibility with LiFSI/DME/TTE solvation.

For each axis, ChemDFM encodes this fixed entity order:

```text
no-structure baseline
A01-A10 anchors
200 candidates ordered by Stage 1 rank
```

The backend takes the final hidden state at the last non-padding prompt token.
It does not generate text or score output labels. Raw vectors are stored as
float32. Within each axis, the no-structure baseline is subtracted from every
anchor and candidate vector, then each centered vector is L2-normalized.

## Semantic Ranking

For each axis, every candidate is compared with all ten anchors by cosine
similarity. The mean of its two highest anchor similarities is the axis score.
Descending average ranks are converted to percentiles:

```text
R_axis = (200 - average_rank_axis) / 199
```

The pure semantic score is the equal mean of the three axis percentiles:

```text
S_semantic = (R_NCM811 + R_Li + R_LHCE) / 3
```

`semantic_rank` depends only on this value. Stage 1 scores do not enter the
semantic formula.

## Deterministic Ordering

Candidates are ordered by descending pure semantic score. Anonymous
`candidate_id` is used only when semantic scores are exactly equal. No
additional chemical or structural score enters the ranking. The former
structure-similarity idea is retained separately as an exploratory diagnostic
and is not part of this method.

## Blind Execution

The public workflow is:

```text
import-source -> freeze -> screen -> reveal-target
```

`freeze` binds source inputs, configuration, prompts, entity order, environment,
model revision and weights, tokenizer, smoke evidence, batch size, and code
commit. `screen` has no private-data argument and creates resumable embedding
checkpoints. It publishes a result lock over embeddings, completion bitmap,
cosines, scores, diagnostics, and Top 50. `reveal-target` verifies the complete
public and private hash chain before reading identity.

## Formal Result

Run `20260724-organic-additive-v4-01` produced the following locked result for
3-cyanotetrahydrofuran:

- Stage 1 rank: 141;
- NCM811 axis rank: 1;
- lithium-metal axis rank: 1;
- LHCE-solvation axis rank: 1;
- pure semantic rank: 1;
- deterministic final rank: 1.

The fixed success criterion, pure semantic rank no greater than 15, was met.
