from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from numbers import Real

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata

from semantic_fingerprint.v4.backend import center_and_normalize


AXES = ("ncm811", "lithium_metal", "lhce_solvation")
_CANDIDATE_COUNT = 200
_ANCHOR_COUNT = 10
_IDENTITY_FIELDS = {
    "name",
    "cid",
    "cas",
    "inchikey",
    "inchi_key",
    "identity",
    "is_target",
    "pubchem_cid",
}


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite.")
    return result


def _validate_candidates(
    candidates: Sequence[Mapping[str, object]],
) -> list[tuple[str, str]]:
    if len(candidates) != _CANDIDATE_COUNT:
        raise ValueError("v4 requires exactly 200 selected candidates.")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    stage1_ranks: list[int] = []
    has_stage1_rank = all("stage1_rank" in row for row in candidates)
    for row in candidates:
        if not isinstance(row, Mapping):
            raise TypeError("Each candidate must be a mapping.")
        forbidden = {
            str(field)
            for field in row
            if str(field).lower() in _IDENTITY_FIELDS
            or str(field).lower().startswith("target")
        }
        if forbidden:
            raise ValueError("Candidate identity fields are not allowed.")
        candidate_id = row.get("candidate_id")
        smiles = row.get("connectivity_smiles")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("connectivity_smiles must be a non-empty string.")
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}.")
        seen.add(candidate_id)
        result.append((candidate_id, smiles))
        if has_stage1_rank:
            rank = row["stage1_rank"]
            if isinstance(rank, bool) or not isinstance(rank, Real):
                raise TypeError("stage1_rank must be numeric.")
            if not float(rank).is_integer():
                raise ValueError("stage1_rank must be an integer position.")
            stage1_ranks.append(int(rank))
    if has_stage1_rank and stage1_ranks != list(range(1, _CANDIDATE_COUNT + 1)):
        raise ValueError("Candidates must be ordered by ascending Stage 1 rank.")
    return result


def _validate_anchors(
    anchors: Sequence[Mapping[str, object]],
) -> list[tuple[str, str]]:
    if len(anchors) != _ANCHOR_COUNT:
        raise ValueError("v4 requires exactly 10 anchors.")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for expected_index, row in enumerate(anchors, start=1):
        if not isinstance(row, Mapping):
            raise TypeError("Each anchor must be a mapping.")
        anchor_id = row.get("anchor_id")
        smiles = row.get("connectivity_smiles")
        expected_id = f"A{expected_index:02d}"
        if anchor_id != expected_id:
            raise ValueError("Anchors must be ordered A01 through A10.")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("Anchor connectivity_smiles must be non-empty.")
        if anchor_id in seen:
            raise ValueError(f"Duplicate anchor_id: {anchor_id}.")
        seen.add(anchor_id)
        result.append((anchor_id, smiles))
    return result


def morgan_fingerprint(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
):
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer.")
    if isinstance(n_bits, bool) or not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer.")
    if not isinstance(use_chirality, bool):
        raise TypeError("use_chirality must be a boolean.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse a structural-score SMILES.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=use_chirality,
    )
    return generator.GetFingerprint(molecule)


def compute_structural_scores(
    candidates: Sequence[Mapping[str, object]],
    anchors: Sequence[Mapping[str, object]],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
) -> dict[str, float]:
    anchor_values = _validate_anchors(anchors)
    anchor_fingerprints = [
        morgan_fingerprint(
            smiles,
            radius=radius,
            n_bits=n_bits,
            use_chirality=use_chirality,
        )
        for _, smiles in anchor_values
    ]
    result: dict[str, float] = {}
    for row in candidates:
        if not isinstance(row, Mapping):
            raise TypeError("Each candidate must be a mapping.")
        candidate_id = row.get("candidate_id")
        smiles = row.get("connectivity_smiles")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string.")
        if candidate_id in result:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}.")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("connectivity_smiles must be a non-empty string.")
        fingerprint = morgan_fingerprint(
            smiles,
            radius=radius,
            n_bits=n_bits,
            use_chirality=use_chirality,
        )
        similarities = np.array(
            [
                DataStructs.TanimotoSimilarity(fingerprint, anchor_fingerprint)
                for anchor_fingerprint in anchor_fingerprints
            ],
            dtype=np.float64,
        )
        top_two = np.partition(similarities, -2)[-2:]
        score = float(top_two.mean())
        if not math.isfinite(score):
            raise ValueError("Structural score must be finite.")
        result[candidate_id] = score
    return result


def aggregate_semantic_fingerprints(
    candidates,
    anchors,
    raw_embeddings,
    structural_scores,
) -> list[dict[str, object]]:
    candidate_values = list(candidates)
    anchor_values = list(anchors)
    candidate_identity = _validate_candidates(candidate_values)
    anchor_identity = _validate_anchors(anchor_values)
    candidate_ids = [item[0] for item in candidate_identity]
    anchor_ids = [item[0] for item in anchor_identity]

    raw = np.asarray(raw_embeddings)
    expected_shape = (
        len(AXES),
        1 + _ANCHOR_COUNT + _CANDIDATE_COUNT,
    )
    if raw.ndim != 3 or raw.shape[:2] != expected_shape or raw.shape[2] < 1:
        raise ValueError(
            "raw_embeddings must have shape (3, 211, hidden_size)."
        )
    if not np.isfinite(raw).all():
        raise ValueError("raw_embeddings must contain only finite values.")

    if not isinstance(structural_scores, Mapping):
        raise TypeError("structural_scores must be a mapping.")
    if set(structural_scores) != set(candidate_ids) or len(structural_scores) != len(
        candidate_ids
    ):
        raise ValueError("Structural scores must exactly cover all candidates.")
    structural = np.array(
        [
            _finite_float(
                structural_scores[candidate_id],
                f"structural score for {candidate_id}",
            )
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    if np.any((structural < 0.0) | (structural > 1.0)):
        raise ValueError("Structural Tanimoto scores must lie in [0, 1].")

    rows: list[dict[str, object]] = [
        {
            "candidate_id": candidate_id,
            "connectivity_smiles": smiles,
        }
        for candidate_id, smiles in candidate_identity
    ]
    axis_scores = np.empty((len(AXES), _CANDIDATE_COUNT), dtype=np.float64)
    for axis_index, axis in enumerate(AXES):
        normalized = center_and_normalize(
            raw[axis_index, 1:],
            raw[axis_index, 0],
        )
        anchor_vectors = normalized[:_ANCHOR_COUNT]
        candidate_vectors = normalized[_ANCHOR_COUNT:]
        cosines = candidate_vectors @ anchor_vectors.T
        if cosines.shape != (_CANDIDATE_COUNT, _ANCHOR_COUNT):
            raise RuntimeError("Unexpected candidate-anchor cosine shape.")
        if not np.isfinite(cosines).all():
            raise ValueError("Candidate-anchor cosine values must be finite.")
        top_order = np.argsort(-cosines, axis=1, kind="stable")[:, :2]
        top_values = np.take_along_axis(cosines, top_order, axis=1)
        scores = top_values.mean(axis=1, dtype=np.float64)
        axis_scores[axis_index] = scores
        for candidate_index, row in enumerate(rows):
            for anchor_index, anchor_id in enumerate(anchor_ids):
                row[f"{axis}_anchor_{anchor_id}_cosine"] = float(
                    cosines[candidate_index, anchor_index]
                )
            row[f"{axis}_top_anchor_1_id"] = anchor_ids[
                int(top_order[candidate_index, 0])
            ]
            row[f"{axis}_top_anchor_2_id"] = anchor_ids[
                int(top_order[candidate_index, 1])
            ]
            row[f"{axis}_score"] = float(scores[candidate_index])

    axis_ranks = np.empty_like(axis_scores)
    for axis_index, axis in enumerate(AXES):
        ranks = rankdata(-axis_scores[axis_index], method="average")
        percentiles = (_CANDIDATE_COUNT - ranks) / (_CANDIDATE_COUNT - 1)
        axis_ranks[axis_index] = ranks
        for index, row in enumerate(rows):
            row[f"{axis}_rank"] = float(ranks[index])
            row[f"{axis}_percentile"] = float(percentiles[index])

    rank_sums = axis_ranks.sum(axis=0)
    semantic_scores = (
        len(AXES) * _CANDIDATE_COUNT - rank_sums
    ) / (len(AXES) * (_CANDIDATE_COUNT - 1))
    semantic_ranks = rankdata(-semantic_scores, method="average")
    structural_ranks = rankdata(-structural, method="average")
    for index, row in enumerate(rows):
        row["semantic_score"] = float(semantic_scores[index])
        row["semantic_rank"] = float(semantic_ranks[index])
        row["structural_score"] = float(structural[index])
        row["structural_rank"] = float(structural_ranks[index])

    order = sorted(
        range(_CANDIDATE_COUNT),
        key=lambda index: (
            -float(semantic_scores[index]),
            -float(structural[index]),
            candidate_ids[index],
        ),
    )
    for final_rank, index in enumerate(order, start=1):
        rows[index]["final_rank"] = final_rank
    return [rows[index] for index in order]


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("Diagnostic summaries require finite one-dimensional data.")
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "median": float(np.median(array)),
    }


def _pair_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    value = float(left.corr(right, method=method))
    return value if math.isfinite(value) else 0.0


def _correlation_matrix(
    frame: pd.DataFrame,
    columns: Mapping[str, str],
    method: str,
) -> dict[str, dict[str, float]]:
    labels = list(columns)
    matrix = {left: {right: 0.0 for right in labels} for left in labels}
    for left_index, left in enumerate(labels):
        matrix[left][left] = 1.0
        for right in labels[left_index + 1 :]:
            value = _pair_correlation(
                frame[columns[left]], frame[columns[right]], method
            )
            matrix[left][right] = value
            matrix[right][left] = value
    return matrix


def build_v4_diagnostics(
    *,
    semantic_rows: Sequence[Mapping[str, object]],
    raw_embeddings: np.ndarray,
) -> dict[str, object]:
    if len(semantic_rows) != _CANDIDATE_COUNT:
        raise ValueError("v4 diagnostics require exactly 200 semantic rows.")
    frame = pd.DataFrame(semantic_rows)
    required_columns = {
        "candidate_id",
        "semantic_score",
        "semantic_rank",
        "structural_score",
        *{f"{axis}_score" for axis in AXES},
        *{f"{axis}_percentile" for axis in AXES},
        *{f"{axis}_top_anchor_1_id" for axis in AXES},
        *{f"{axis}_top_anchor_2_id" for axis in AXES},
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError("Semantic rows are incomplete for diagnostics.")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Diagnostic candidate IDs must be unique.")

    raw = np.asarray(raw_embeddings)
    if raw.ndim != 3 or raw.shape[:2] != (3, 211):
        raise ValueError("Diagnostic raw embeddings must have shape (3, 211, H).")
    if not np.isfinite(raw).all():
        raise ValueError("Diagnostic embeddings must be finite.")

    centered_norms: dict[str, object] = {}
    anchor_frequency: dict[str, dict[str, int]] = {}
    axis_summaries: dict[str, object] = {}
    for axis_index, axis in enumerate(AXES):
        differences = raw[axis_index, 1:] - raw[axis_index, 0]
        norms = np.linalg.norm(differences.astype(np.float32), axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("Centered diagnostic norms must be positive and finite.")
        centered_norms[axis] = {
            "anchors": _summary(norms[:_ANCHOR_COUNT]),
            "candidates": _summary(norms[_ANCHOR_COUNT:]),
        }
        counts = Counter(
            str(value)
            for column in (
                f"{axis}_top_anchor_1_id",
                f"{axis}_top_anchor_2_id",
            )
            for value in frame[column]
        )
        anchor_frequency[axis] = {
            f"A{index:02d}": int(counts[f"A{index:02d}"])
            for index in range(1, _ANCHOR_COUNT + 1)
        }
        axis_summaries[axis] = {
            "score": _summary(frame[f"{axis}_score"].to_numpy(dtype=float)),
            "percentile": _summary(
                frame[f"{axis}_percentile"].to_numpy(dtype=float)
            ),
        }

    axis_columns = {axis: f"{axis}_score" for axis in AXES}
    return {
        "schema_version": 4,
        "method_version": "organic-semantic-fingerprint-v4",
        "axis_correlations": {
            method: _correlation_matrix(frame, axis_columns, method)
            for method in ("spearman", "pearson")
        },
        "axis_summaries": axis_summaries,
        "centered_norms": centered_norms,
        "anchor_selection_frequency": anchor_frequency,
        "structure_semantic_correlation": {
            method: _pair_correlation(
                frame["structural_score"], frame["semantic_score"], method
            )
            for method in ("spearman", "pearson")
        },
    }
