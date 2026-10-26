from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.semantic_screening_results.manuscript_figure.plot_results import (
    create_figure,
    create_individual_figures,
    load_data,
)
from figures.semantic_screening_results.manuscript_figure.prepare_figure_data import (
    verify_bundle,
)


FIGURE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = FIGURE_ROOT.parent / "data"
DATA_DIR = FIGURE_ROOT / "data"


def test_figure_bundle_is_bound_and_complete():
    manifest = verify_bundle(SOURCE_DIR, DATA_DIR)
    assert manifest["figure_release"] == "manuscript"

    expected_rows = {
        "rank_migration.csv": 200,
        "axis_score_distribution.csv": 600,
        "axis_rank1_rank2_summary.csv": 3,
        "semantic_aggregation_trajectories.csv": 200,
        "selected_anchor_profile.csv": 30,
        "anchor_selection_frequency.csv": 30,
    }
    for name, count in expected_rows.items():
        frame = pd.read_csv(DATA_DIR / name)
        assert len(frame) == count
        assert not any("structural" in column for column in frame.columns)


def test_plot_data_preserve_the_selected_semantic_result():
    data = load_data(DATA_DIR)
    selected = data["selected"]
    assert selected["candidate_id"] == "C0686"
    assert selected["display_name"] == "3-cyanotetrahydrofuran"
    assert int(selected["stage1_rank"]) == 141
    assert int(selected["final_rank"]) == 1
    assert np.isclose(
        float(selected["stage1_ranking_percentile"]),
        (200.0 - 141.0) / 199.0 * 100.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert float(selected["final_semantic_ranking_percentile"]) == 100.0

    migration = data["migration"]
    assert np.allclose(
        migration["stage1_ranking_percentile"].to_numpy(dtype=float),
        (200.0 - migration["stage1_rank"].to_numpy(dtype=float)) / 199.0 * 100.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        migration["final_semantic_ranking_percentile"].to_numpy(dtype=float),
        (200.0 - migration["final_rank"].to_numpy(dtype=float)) / 199.0 * 100.0,
        rtol=0.0,
        atol=1e-12,
    )

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


def test_aggregation_uses_all_candidates_and_equal_weights():
    trajectories = load_data(DATA_DIR)["trajectories"]
    assert len(trajectories) == 200
    percentiles = trajectories[
        ["ncm811_percentile", "lithium_metal_percentile", "lhce_solvation_percentile"]
    ].to_numpy(dtype=float)
    assert np.allclose(
        percentiles.mean(axis=1),
        trajectories["semantic_score"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    selected = trajectories.loc[trajectories["is_selected_candidate"]].iloc[0]
    assert np.allclose(
        selected[
            [
                "ncm811_percentile",
                "lithium_metal_percentile",
                "lhce_solvation_percentile",
                "semantic_score",
            ]
        ].to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_rank_gaps_and_anchor_context_are_complete():
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

    anchors = data["anchors"]
    distributions = data["distributions"]
    for axis in expected:
        top_two = anchors.loc[
            (anchors["axis"] == axis) & anchors["is_top2"], "cosine_similarity"
        ].to_numpy(dtype=float)
        rank1_score = float(
            distributions.loc[
                (distributions["axis"] == axis)
                & distributions["is_selected_candidate"],
                "axis_score",
            ].iloc[0]
        )
        assert len(top_two) == 2
        assert abs(float(top_two.mean()) - rank1_score) <= 1e-12

    frequency = data["anchor_frequency"]
    assert frequency.groupby("axis")["selection_count"].sum().eq(400).all()


def test_identity_is_post_ranking_and_not_a_scoring_field():
    data = load_data(DATA_DIR)
    molecule = data["molecule"]
    assert molecule["candidate_id"] == "C0686"
    assert molecule["identity_used_for_ranking"] is False
    for frame_name in (
        "migration",
        "distributions",
        "gaps",
        "trajectories",
        "anchors",
        "anchor_frequency",
    ):
        frame = data[frame_name]
        assert not any("structural" in column for column in frame.columns)
        assert "target" not in frame.columns


def test_builds_four_independent_panel_figures():
    panels = create_individual_figures(DATA_DIR)
    try:
        assert set(panels) == {"a", "b", "c", "d"}
        assert all(panel.axes for panel in panels.values())
        assert all((panel.get_size_inches() > 0).all() for panel in panels.values())
        assert len(panels["c"].axes[0].lines) >= 204
    finally:
        for panel in panels.values():
            plt.close(panel)


def test_composite_headers_align_and_axis_facets_share_scale():
    figure = create_figure(DATA_DIR)
    try:
        title_positions = {
            text.get_text(): text.get_position()[1]
            for text in figure.texts
            if text.get_text()
            in {
                "Semantic reranking",
                "Candidate score distributions",
                "Equal-weight semantic aggregation",
                "Anchor similarity and selection frequency",
            }
        }
        assert title_positions["Semantic reranking"] == title_positions[
            "Candidate score distributions"
        ]
        assert title_positions["Equal-weight semantic aggregation"] == title_positions[
            "Anchor similarity and selection frequency"
        ]
        assert max(title_positions.values()) < 1.0

        rank_axes = [
            axis
            for axis in figure.axes
            if axis.get_xlabel() == "Stage 1 ranking percentile (%)"
        ]
        assert len(rank_axes) == 1
        rank_axis = rank_axes[0]
        assert np.allclose(rank_axis.get_xlim(), (-3, 103), rtol=0.0, atol=1e-12)
        assert np.allclose(rank_axis.get_ylim(), (-3, 103), rtol=0.0, atol=1e-12)
        assert np.array_equal(rank_axis.get_xticks(), rank_axis.get_yticks())

        all_axes = [
            child
            for axis in figure.axes
            for child in (axis, *axis.child_axes)
        ]
        shared_scale_axes = [
            axis
            for axis in all_axes
            if np.allclose(axis.get_xlim(), (0.865, 0.985), rtol=0.0, atol=1e-12)
        ]
        assert len(shared_scale_axes) == 3
        assert all(
            np.allclose(
                axis.get_xticks(),
                [0.87, 0.90, 0.93, 0.96, 0.98],
                rtol=0.0,
                atol=1e-12,
            )
            for axis in shared_scale_axes
        )

        figure.canvas.draw()
        heatmap_axes = [
            axis
            for axis in all_axes
            if axis.images
            and tuple(axis.images[0].get_array().shape) == (3, 10)
        ]
        assert len(heatmap_axes) == 2
        lower, upper = sorted(heatmap_axes, key=lambda axis: axis.get_position().y0)
        lower_box = lower.get_position()
        upper_box = upper.get_position()
        assert np.isclose(lower_box.height, upper_box.height, rtol=0.0, atol=1e-12)
        heatmap_gap = upper_box.y0 - lower_box.y1
        assert 0.0 < heatmap_gap < 0.5 * lower_box.height

        count_colorbars = [axis for axis in all_axes if axis.get_title() == "count"]
        assert len(count_colorbars) == 1
        count_bar = count_colorbars[0]
        renderer = figure.canvas.get_renderer()
        title_box = count_bar.title.get_window_extent(renderer=renderer)
        tick_boxes = [
            tick.get_window_extent(renderer=renderer)
            for tick in count_bar.get_yticklabels()
            if tick.get_visible() and tick.get_text()
        ]
        assert all(not title_box.overlaps(tick_box) for tick_box in tick_boxes)

        semantic_axes = [
            axis
            for axis in figure.axes
            if len(axis.lines) >= 204
            and np.allclose(axis.get_xlim(), (-0.035, 1.055), rtol=0.0, atol=1e-12)
        ]
        anchor_containers = [
            axis for axis in figure.axes if len(axis.child_axes) == 4
        ]
        assert len(semantic_axes) == 1
        assert len(anchor_containers) == 1
        semantic_box = semantic_axes[0].get_position()
        anchor_box = anchor_containers[0].get_position()
        assert np.isclose(anchor_box.y0, semantic_box.y0, rtol=0.0, atol=1e-12)
        assert np.isclose(anchor_box.y1, semantic_box.y1, rtol=0.0, atol=1e-12)
        assert anchor_box.width > 0.565
    finally:
        plt.close(figure)
