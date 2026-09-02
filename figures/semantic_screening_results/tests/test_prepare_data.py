from __future__ import annotations

import json
from pathlib import Path

import pytest

from figures.semantic_screening_results.prepare_data import (
    annotate_selected_candidate,
    average_descending_ranks,
    derive_anonymous_tables,
    load_config,
    prepare,
    validate_public_run,
    verify_generated,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_average_descending_ranks_uses_average_ties():
    assert average_descending_ranks([0.9, 0.8, 0.8, 0.1]) == [
        1.0,
        2.5,
        2.5,
        4.0,
    ]


def test_formal_public_data_and_anonymous_derivations():
    public = validate_public_run(REPO_ROOT, load_config())
    anonymous = derive_anonymous_tables(public)

    assert len(anonymous["ranking_all"]) == 200
    assert len(anonymous["axis_scores_long"]) == 600
    assert len(anonymous["anchor_cosines_long"]) == 6000
    assert len(anonymous["top20_axis_matrix"]) == 20
    assert len(anonymous["top50_ranking"]) == 50
    assert len(anonymous["stage1_all"]) == 800
    assert len(anonymous["semantic_tie_groups"]) == 52
    assert anonymous["processing_summary"]["semantic_tie_group_count"] == 24
    assert anonymous["processing_summary"]["maximum_semantic_tie_size"] == 3

    selected = anonymous["ranking_all"][0]
    assert selected["candidate_id"] == "C0686"
    assert selected["semantic_rank"] == 1.0
    assert selected["final_rank"] == 1
    assert selected["semantic_tie_size"] == 1
    assert selected["candidate_id_tiebreak_applied"] is False
    assert "structural_score" not in selected
    assert "structural_rank" not in selected
    assert "display_name" not in selected
    assert "target" not in selected


def test_annotation_occurs_only_for_public_rank_one():
    public = validate_public_run(REPO_ROOT, load_config())
    anonymous = derive_anonymous_tables(public)
    annotated = annotate_selected_candidate(public, anonymous)

    summary = annotated["selected_candidate_summary"]
    assert summary["candidate_id"] == "C0686"
    assert summary["display_name"] == "3-cyanotetrahydrofuran"
    assert summary["stage1_rank"] == 141
    assert summary["ncm811_rank"] == 1.0
    assert summary["lithium_metal_rank"] == 1.0
    assert summary["lhce_solvation_rank"] == 1.0
    assert summary["semantic_rank"] == 1.0
    assert summary["final_rank"] == 1
    assert summary["candidate_id_tiebreak_applied"] is False
    assert "structural_score" not in summary
    assert "structural_rank" not in summary
    assert "role" not in summary

    selected_rows = [
        row
        for row in annotated["ranking_all_annotated"]
        if row["is_selected_candidate"]
    ]
    assert len(selected_rows) == 1
    assert selected_rows[0]["candidate_id"] == "C0686"
    profile = annotated["selected_anchor_profile"]
    assert len(profile) == 30
    for axis in ("ncm811", "lithium_metal", "lhce_solvation"):
        assert [
            row["anchor_id"]
            for row in profile
            if row["axis"] == axis and row["is_top2"]
        ] == ["A05", "A06"]


def test_annotation_rejects_a_different_public_rank_one():
    public = validate_public_run(REPO_ROOT, load_config())
    anonymous = derive_anonymous_tables(public)
    changed = dict(anonymous)
    changed["ranking_all"] = [dict(row) for row in anonymous["ranking_all"]]
    changed["ranking_all"][0]["candidate_id"] = "C0001"
    with pytest.raises(ValueError, match="rank-1 candidate"):
        annotate_selected_candidate(public, changed)


def test_prepare_verify_and_idempotency(tmp_path):
    output = tmp_path / "data"
    first = prepare(REPO_ROOT, output)
    assert first["status"] == "created"
    manifest = verify_generated(REPO_ROOT, output)
    assert manifest["selection"]["anonymous_rank1_candidate_id"] == "C0686"

    second = prepare(REPO_ROOT, output)
    assert second["status"] == "unchanged"

    ranking_path = output / "anonymous/ranking_all.csv"
    ranking_path.write_text(
        ranking_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatch"):
        verify_generated(REPO_ROOT, output)
    with pytest.raises(FileExistsError, match="does not verify"):
        prepare(REPO_ROOT, output)


def test_generated_manifest_does_not_publish_private_registry(tmp_path):
    output = tmp_path / "data"
    prepare(REPO_ROOT, output)
    manifest_text = (output / "data_manifest.json").read_text(encoding="utf-8")
    assert "target_registry" not in manifest_text
    assert "target_reveal" not in manifest_text
    assert "private_runs_v4" not in manifest_text
    selected = json.loads(
        (output / "annotated/selected_candidate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected["identity_used_for_ranking"] is False
