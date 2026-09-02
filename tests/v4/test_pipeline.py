from __future__ import annotations

import json
import math

import numpy as np
import pytest

from semantic_fingerprint.v4.backend import center_and_normalize
from semantic_fingerprint.v4.pipeline import (
    AXES,
    aggregate_semantic_fingerprints,
    build_v4_diagnostics,
)


def _candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"C{index:04d}",
            "connectivity_smiles": "C" + "C" * (index % 7),
            "stage1_rank": index,
        }
        for index in range(1, 201)
    ]


def _anchors() -> list[dict[str, str]]:
    smiles = [
        "CCO",
        "CCN",
        "CC#N",
        "O=C1OCCO1",
        "CS(=O)(=O)C",
        "B(OC)(OC)OC",
        "P(OC)(OC)OC",
        "C1COCO1",
        "CC(F)(F)F",
        "N#CCCC#N",
    ]
    return [
        {
            "anchor_id": f"A{index:02d}",
            "connectivity_smiles": value,
        }
        for index, value in enumerate(smiles, start=1)
    ]


def _score_embeddings(axis_scores: np.ndarray) -> np.ndarray:
    assert axis_scores.shape == (3, 200)
    raw = np.zeros((3, 211, 2), dtype=np.float32)
    raw[:, 1:11, 0] = 1.0
    for axis_index in range(3):
        values = axis_scores[axis_index]
        raw[axis_index, 11:, 0] = values
        raw[axis_index, 11:, 1] = np.sqrt(1.0 - values**2)
    return raw


def test_aggregate_computes_all_three_200_by_10_cosine_matrices_and_top2() -> None:
    rng = np.random.default_rng(20260724)
    baseline = rng.normal(size=(3, 1, 12)).astype(np.float32)
    offsets = rng.normal(size=(3, 210, 12)).astype(np.float32)
    raw = np.concatenate((baseline, baseline + offsets), axis=1)

    rows = aggregate_semantic_fingerprints(
        _candidates(),
        _anchors(),
        raw,
    )

    assert len(rows) == 200
    by_id = {str(row["candidate_id"]): row for row in rows}
    first = by_id["C0001"]
    for axis_index, axis in enumerate(AXES):
        normalized = center_and_normalize(
            raw[axis_index, 1:], raw[axis_index, 0]
        )
        expected = normalized[10] @ normalized[:10].T
        actual = np.array(
            [first[f"{axis}_anchor_A{index:02d}_cosine"] for index in range(1, 11)]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        assert first[f"{axis}_score"] == pytest.approx(
            float(np.sort(expected)[-2:].mean())
        )
    assert sum("_anchor_A" in key for key in first) == 30


def test_average_tied_axis_and_semantic_ranks_use_candidate_id_fallback() -> None:
    scores = np.empty((3, 200), dtype=np.float32)
    scores[:, 0:2] = 0.99
    descending = np.linspace(0.8, 0.1, 198, dtype=np.float32)
    scores[:, 2:] = descending
    raw = _score_embeddings(scores)
    rows = aggregate_semantic_fingerprints(_candidates(), _anchors(), raw)
    by_id = {str(row["candidate_id"]): row for row in rows}

    expected_percentile = (200.0 - 1.5) / 199.0
    for candidate_id in ("C0001", "C0002"):
        row = by_id[candidate_id]
        assert [row[f"{axis}_rank"] for axis in AXES] == pytest.approx(
            [1.5, 1.5, 1.5]
        )
        assert [row[f"{axis}_percentile"] for axis in AXES] == pytest.approx(
            [expected_percentile] * 3
        )
        assert row["semantic_score"] == pytest.approx(expected_percentile)
        assert row["semantic_rank"] == pytest.approx(1.5)

    assert by_id["C0001"]["final_rank"] == 1
    assert by_id["C0002"]["final_rank"] == 2
    assert all("structural" not in key for row in rows for key in row)


def test_equal_axis_percentile_mean_uses_only_axis_ranks() -> None:
    values = np.linspace(0.1, 0.99, 200, dtype=np.float32)
    scores = np.stack((values, values[::-1], np.roll(values, 37)))
    rows = aggregate_semantic_fingerprints(
        _candidates(),
        _anchors(),
        _score_embeddings(scores),
    )

    for row in rows:
        expected = sum(float(row[f"{axis}_percentile"]) for axis in AXES) / 3.0
        assert float(row["semantic_score"]) == pytest.approx(expected)
    semantic_scores = np.array([float(row["semantic_score"]) for row in rows])
    assert np.isfinite(semantic_scores).all()


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate_count",
        "anchor_count",
        "raw_shape",
        "nonfinite_raw",
        "duplicate_candidate",
        "duplicate_anchor",
    ],
)
def test_aggregate_rejects_drift_or_incomplete_inputs(mutation: str) -> None:
    candidates = _candidates()
    anchors = _anchors()
    raw = _score_embeddings(np.full((3, 200), 0.5, dtype=np.float32))

    if mutation == "candidate_count":
        candidates.pop()
    elif mutation == "anchor_count":
        anchors.pop()
    elif mutation == "raw_shape":
        raw = raw[:, :-1]
    elif mutation == "nonfinite_raw":
        raw[1, 12, 0] = np.nan
    elif mutation == "duplicate_candidate":
        candidates[1]["candidate_id"] = "C0001"
    elif mutation == "duplicate_anchor":
        anchors[1]["anchor_id"] = "A01"

    with pytest.raises((TypeError, ValueError)):
        aggregate_semantic_fingerprints(candidates, anchors, raw)


def test_v4_diagnostics_are_anonymous_symmetric_finite_and_complete() -> None:
    values = np.linspace(0.1, 0.99, 200, dtype=np.float32)
    raw = _score_embeddings(
        np.stack((values, np.roll(values, 17), np.roll(values, 43)))
    )
    rows = aggregate_semantic_fingerprints(_candidates(), _anchors(), raw)
    diagnostics = build_v4_diagnostics(
        semantic_rows=rows,
        raw_embeddings=raw,
    )

    assert diagnostics["schema_version"] == 4
    for method in ("spearman", "pearson"):
        matrix = diagnostics["axis_correlations"][method]
        assert set(matrix) == set(AXES)
        for left in AXES:
            assert set(matrix[left]) == set(AXES)
            assert matrix[left][left] == 1.0
            for right in AXES:
                assert matrix[left][right] == pytest.approx(matrix[right][left])
                assert math.isfinite(matrix[left][right])

    for axis in AXES:
        norm_summary = diagnostics["centered_norms"][axis]
        assert norm_summary["anchors"]["count"] == 10
        assert norm_summary["candidates"]["count"] == 200
        frequencies = diagnostics["anchor_selection_frequency"][axis]
        assert set(frequencies) == {f"A{index:02d}" for index in range(1, 11)}
        assert sum(frequencies.values()) == 400
    assert "structure_semantic_correlation" not in diagnostics
    serialized = json.dumps(diagnostics).lower()
    assert "target" not in serialized
    assert "10219385" not in serialized
