# Organic Additive Semantic Fingerprint

Context-centered ChemDFM screening for neutral organic electrolyte additives in
a LiFSI/DME/TTE localized high-concentration electrolyte for high-voltage
NCM811||Li-metal full cells.

## Method

The workflow ranks a frozen Stage 1 shortlist of 200 candidates against ten
reference electrolyte additives on three independent semantic axes:

1. NCM811 oxidative and interphase compatibility;
2. lithium-metal reductive and interphase compatibility;
3. non-disruptive LiFSI/DME/TTE solvation compatibility.

For each axis, the model encodes a no-structure baseline, ten anchors, and 200
candidate molecular cards. After subtracting the axis-specific baseline and L2
normalizing, each candidate is scored by the mean cosine similarity to its two
closest anchors. The three axis-rank percentiles are averaged equally to obtain
the pure semantic score.

Morgan similarity is used only to resolve exact semantic-score ties. It does not
enter the semantic score or the scientific success criterion.

## Repository Layout

```text
project_config/                 Frozen scientific and model configuration
semantic_fingerprint/v4/       Molecular cards, model backend, ranking, locking
scripts/run_organic_screening_v4.py
tests/v4/                       Unit, security, recovery, and integration tests
docs/superpowers/specs/         Complete method specification
```

## Environment

```bash
conda env create -f environment.yml
conda activate organic-semantic-fingerprint
python -m pytest -q
```

ChemDFM is loaded from the local model directory fixed in the configuration.
Model weights are not committed to Git.

## Workflow

```text
import-source -> freeze -> screen -> reveal-target
```

Use the CLI help for the complete arguments:

```bash
python scripts/run_organic_screening_v4.py --help
```

The blind `screen` command accepts no private target input. Public results are
hash-locked before `reveal-target` can read the private target registry.

## Formal Result

For run `20260724-organic-additive-v4-01`, 3-cyanotetrahydrofuran achieved rank
1 on all three semantic axes, pure semantic rank 1, and final rank 1. The fixed
success criterion was pure semantic rank no greater than 15.

Run products, private reveal files, model weights, and caches are excluded from
Git by `.gitignore`.
