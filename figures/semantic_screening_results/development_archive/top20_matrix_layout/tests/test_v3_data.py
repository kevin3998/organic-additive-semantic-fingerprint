from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.semantic_screening_results.development_archive.top20_matrix_layout.plot_results import (
    create_individual_figures,
    load_data,
)
from figures.semantic_screening_results.development_archive.top20_matrix_layout.prepare_v3_data import (
    verify_bundle,
)


V3_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = V3_ROOT.parent.parent / "data"
DATA_DIR = V3_ROOT / "data"


def test_v3_bundle_is_bound_and_complete():
    manifest = verify_bundle(SOURCE_DIR, DATA_DIR)
    assert manifest["figure_version"] == "v3"

    expected_rows = {
        "rank_migration.csv": 200,
        "axis_score_distribution.csv": 600,
        "axis_rank1_rank2_summary.csv": 3,
        "top20_semantic_profiles.csv": 20,
        "selected_anchor_profile.csv": 30,
        "anchor_selection_frequency.csv": 30,
    }
    for name, count in expected_rows.items():
        frame = pd.read_csv(DATA_DIR / name)
        assert len(frame) == count
        assert not any("structural" in column for column in frame.columns)


def test_v3_plot_data_preserve_the_selected_semantic_result():
    data = load_data(DATA_DIR)
    selected = data["selected"]
    assert selected["candidate_id"] == "C0686"
    assert selected["display_name"] == "3-cyanotetrahydrofuran"
    assert int(selected["stage1_rank"]) == 141
    assert int(selected["final_rank"]) == 1

    distributions = data["distributions"]
    selected_scores = distributions.loc[
        distributions["is_selected_candidate"], ["axis", "axis_rank"]
    ]
    assert set(selected_scores["axis"]) == {
        "ncm811",
        "lithium_metal",
        "lhce_solvation",
    }
    assert selected_scores["axis_rank"].astype(int).eq(1).all()


def test_v3_rank_gaps_and_global_anchor_context_are_complete():
    data = load_data(DATA_DIR)
    gaps = data["gaps"].set_index("axis")
    expected = {
        "ncm811": 0.007522284984588845,
        "lithium_metal": 0.008336186408996582,
        "lhce_solvation": 0.00476706027984608,
    }
    for axis, gap in expected.items():
        assert abs(float(gaps.loc[axis, "rank1_rank2_gap"]) - gap) <= 1e-12
        assert gaps.loc[axis, "rank1_candidate_id"] == "C0686"

    frequency = data["anchor_frequency"]
    assert len(frequency) == 30
    assert frequency.groupby("axis")["selection_count"].sum().eq(400).all()
    assert int(
        frequency.loc[
            (frequency["axis"] == "lithium_metal")
            & (frequency["anchor_id"] == "A02"),
            "selection_count",
        ].iloc[0]
    ) == 160


def test_v3_identity_is_post_ranking_and_not_a_scoring_field():
    data = load_data(DATA_DIR)
    molecule = data["molecule"]
    assert molecule["candidate_id"] == "C0686"
    assert molecule["molecular_formula"] == "C5H7NO"
    assert molecule["identity_used_for_ranking"] is False
    for frame_name in (
        "migration",
        "distributions",
        "gaps",
        "top20",
        "anchors",
        "anchor_frequency",
    ):
        frame = data[frame_name]
        assert not any("structural" in column for column in frame.columns)
        assert "target" not in frame.columns


def test_v3_builds_four_independent_panel_figures():
    panels = create_individual_figures(DATA_DIR)
    try:
        assert set(panels) == {"a", "b", "c", "d"}
        assert all(panel.axes for panel in panels.values())
        assert all((panel.get_size_inches() > 0).all() for panel in panels.values())
    finally:
        for panel in panels.values():
            plt.close(panel)
