#!/usr/bin/env python3
"""Plot the Nature-style v3 pure-semantic screening Results figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 150
TIFF_DPI = 3200

AXIS_ORDER = ("ncm811", "lithium_metal", "lhce_solvation")
AXIS_LABELS = {
    "ncm811": "NCM811",
    "lithium_metal": "Li metal",
    "lhce_solvation": "LHCE",
}
ANCHOR_SHORT = {
    "FEC": "A01 FEC",
    "VEC": "A02 VEC",
    "PS": "A03 PS",
    "DTD": "A04 DTD",
    "SN": "A05 SN",
    "ADN": "A06 ADN",
    "TMSPi": "A07 TMSPi",
    "DETSP": "A08 DETSP",
    "trimethyl borate": "A09 TMB",
    "trimethoxyboroxine": "A10 TMBO",
}

COLORS = {
    "ink": "#20212A",
    "mid": "#4D4E5A",
    "light": "#D9D8EA",
    "very_light": "#F6F4FA",
    "point": "#C9CBE3",
    "selected": "#F09668",
    "selected_dark": "#C9653E",
    "selected_light": "#FADCCB",
    "top20": "#F5F1FA",
    "ncm811": "#E66F8D",
    "ncm811_light": "#F5C9D4",
    "lithium_metal": "#5969D8",
    "lithium_metal_light": "#CDD2F4",
    "lhce_solvation": "#A05BC6",
    "lhce_solvation_light": "#E5CDEF",
}

HERE = Path(__file__).resolve().parent


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(
    ax: mpl.axes.Axes,
    label: str,
    *,
    x: float = -0.10,
    y: float = 1.04,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def contrast_text_color(
    image: mpl.image.AxesImage,
    value: float,
) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.57 else COLORS["ink"]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false"}).all():
        raise ValueError("Invalid Boolean values in v3 data.")
    return values.eq("true")


def load_data(data_dir: Path) -> dict[str, object]:
    migration = pd.read_csv(data_dir / "rank_migration.csv")
    distributions = pd.read_csv(data_dir / "axis_score_distribution.csv")
    gaps = pd.read_csv(data_dir / "axis_rank1_rank2_summary.csv")
    top20 = pd.read_csv(data_dir / "top20_semantic_profiles.csv")
    anchors = pd.read_csv(data_dir / "selected_anchor_profile.csv")
    anchor_frequency = pd.read_csv(data_dir / "anchor_selection_frequency.csv")
    with (data_dir / "selected_molecule.json").open(encoding="utf-8") as handle:
        molecule = json.load(handle)
    with (data_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    require_columns(
        migration,
        [
            "candidate_id",
            "stage1_rank",
            "stage1_score",
            "semantic_score",
            "semantic_rank",
            "final_rank",
            "is_selected_candidate",
            "display_name",
        ],
        "rank migration",
    )
    require_columns(
        distributions,
        [
            "candidate_id",
            "axis",
            "axis_score",
            "axis_rank",
            "final_rank",
            "is_selected_candidate",
        ],
        "axis-score distributions",
    )
    require_columns(
        gaps,
        [
            "axis",
            "rank1_candidate_id",
            "rank1_score",
            "rank2_candidate_id",
            "rank2_score",
            "rank1_rank2_gap",
            "p25",
            "median",
            "p75",
        ],
        "rank-gap summary",
    )
    require_columns(
        top20,
        [
            "candidate_id",
            "final_rank",
            "is_selected_candidate",
            *[
                f"{axis}_{suffix}"
                for axis in AXIS_ORDER
                for suffix in ("score", "rank", "percentile")
            ],
        ],
        "Top 20 profiles",
    )
    require_columns(
        anchors,
        [
            "axis",
            "anchor_id",
            "anchor_name",
            "cosine_similarity",
            "is_top2",
        ],
        "selected-anchor profile",
    )
    require_columns(
        anchor_frequency,
        ["axis", "anchor_id", "anchor_name", "selection_count"],
        "anchor-selection frequency",
    )

    if (
        len(migration) != 200
        or migration["candidate_id"].nunique() != 200
        or sorted(migration["final_rank"].astype(int)) != list(range(1, 201))
    ):
        raise ValueError("Rank migration must contain final ranks 1–200.")
    if (
        len(distributions) != 600
        or distributions["candidate_id"].nunique() != 200
        or set(distributions["axis"]) != set(AXIS_ORDER)
    ):
        raise ValueError("Axis distributions must contain 200 × 3 records.")
    if len(gaps) != 3 or set(gaps["axis"]) != set(AXIS_ORDER):
        raise ValueError("Rank-gap summary must contain the three semantic axes.")
    if len(top20) != 20 or top20["final_rank"].astype(int).tolist() != list(
        range(1, 21)
    ):
        raise ValueError("Top-20 profiles must contain ranks 1–20.")
    if (
        len(anchors) != 30
        or anchors["anchor_id"].nunique() != 10
        or set(anchors["axis"]) != set(AXIS_ORDER)
    ):
        raise ValueError("Anchor profile must contain 3 × 10 records.")
    if (
        len(anchor_frequency) != 30
        or anchor_frequency["anchor_id"].nunique() != 10
        or set(anchor_frequency["axis"]) != set(AXIS_ORDER)
    ):
        raise ValueError("Anchor frequency must contain 3 × 10 records.")

    migration["is_selected_candidate"] = _as_bool(
        migration["is_selected_candidate"]
    )
    distributions["is_selected_candidate"] = _as_bool(
        distributions["is_selected_candidate"]
    )
    top20["is_selected_candidate"] = _as_bool(top20["is_selected_candidate"])
    anchors["is_top2"] = _as_bool(anchors["is_top2"])
    selected = migration.loc[migration["is_selected_candidate"]]
    if (
        len(selected) != 1
        or int(selected.iloc[0]["stage1_rank"]) != 141
        or int(selected.iloc[0]["final_rank"]) != 1
        or molecule.get("candidate_id") != str(selected.iloc[0]["candidate_id"])
        or molecule.get("identity_used_for_ranking") is not False
    ):
        raise ValueError("Expected one post-ranking candidate moving from rank 141 to 1.")

    numeric_blocks = (
        migration[["stage1_rank", "final_rank", "stage1_score", "semantic_score"]],
        distributions[["axis_score", "axis_rank"]],
        gaps[["rank1_score", "rank2_score", "rank1_rank2_gap"]],
        anchors[["cosine_similarity"]],
        anchor_frequency[["selection_count"]],
    )
    if not all(np.isfinite(block.to_numpy(dtype=float)).all() for block in numeric_blocks):
        raise ValueError("v3 plot data contain non-finite values.")

    for axis in AXIS_ORDER:
        subset = distributions.loc[distributions["axis"] == axis].sort_values(
            ["axis_rank", "candidate_id"], kind="stable"
        )
        gap = gaps.loc[gaps["axis"] == axis].iloc[0]
        if (
            str(subset.iloc[0]["candidate_id"]) != str(gap["rank1_candidate_id"])
            or str(subset.iloc[1]["candidate_id"]) != str(gap["rank2_candidate_id"])
            or not np.isclose(
                float(subset.iloc[0]["axis_score"] - subset.iloc[1]["axis_score"]),
                float(gap["rank1_rank2_gap"]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"Rank-gap summary is inconsistent for {axis}.")

    return {
        "migration": migration.sort_values("final_rank", kind="stable"),
        "distributions": distributions,
        "gaps": gaps,
        "top20": top20,
        "anchors": anchors,
        "anchor_frequency": anchor_frequency,
        "selected": selected.iloc[0],
        "molecule": molecule,
        "summary": summary,
    }


def plot_rank_migration(
    ax: mpl.axes.Axes,
    migration: pd.DataFrame,
    selected: pd.Series,
) -> None:
    stage1_rank = migration["stage1_rank"].to_numpy(dtype=float)
    final_rank = migration["final_rank"].to_numpy(dtype=float)

    ax.axhspan(0.5, 20.5, color=COLORS["top20"], lw=0, zorder=0)
    ax.plot(
        [1, 200],
        [1, 200],
        color="#AAB3B5",
        lw=0.75,
        ls=(0, (3, 2)),
        zorder=1,
    )
    ax.scatter(
        stage1_rank,
        final_rank,
        s=11,
        color=COLORS["point"],
        alpha=0.70,
        edgecolor="white",
        linewidth=0.20,
        zorder=2,
    )

    sx = float(selected["stage1_rank"])
    sy = float(selected["final_rank"])
    ax.scatter(
        [sx],
        [sy],
        s=105,
        color=COLORS["selected_light"],
        edgecolor="none",
        zorder=5,
    )
    ax.scatter(
        [sx],
        [sy],
        s=72,
        color="white",
        edgecolor="none",
        zorder=6,
    )
    ax.scatter(
        [sx],
        [sy],
        s=48,
        color=COLORS["selected"],
        edgecolor="white",
        linewidth=0.65,
        zorder=7,
    )
    ax.annotate(
        "C0686  ·  141 → 1",
        xy=(sx, sy),
        xytext=(87, 27),
        ha="left",
        va="center",
        fontsize=6.3,
        fontweight="bold",
        color=COLORS["selected_dark"],
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["selected_dark"],
            "lw": 0.8,
            "shrinkA": 2,
            "shrinkB": 4,
        },
    )
    ax.set(
        xlim=(-8, 205),
        ylim=(205, -8),
        xlabel="Stage 1 anchor-prescreen rank",
        ylabel="Final semantic rank",
        xticks=[1, 50, 100, 150, 200],
        yticks=[1, 20, 50, 100, 150, 200],
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E9E7F1", lw=0.45, zorder=0)
    ax.set_title("Semantic reranking", loc="left", pad=6)
    add_panel_label(ax, "a", x=-0.14, y=1.04)


def _deterministic_jitter(count: int, amplitude: float = 0.125) -> np.ndarray:
    index = np.arange(count, dtype=float)
    return amplitude * np.sin(index * 2.399963229728653)


def plot_axis_score_facets(
    figure: mpl.figure.Figure,
    rect: tuple[float, float, float, float],
    distributions: pd.DataFrame,
    gaps: pd.DataFrame,
) -> mpl.axes.Axes:
    container = figure.add_axes(rect)
    container.set_axis_off()
    container.set_xlim(0, 1)
    container.set_ylim(0, 1)
    container.text(
        0.00,
        1.025,
        "b",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="bottom",
        clip_on=False,
    )
    container.text(
        0.045,
        1.025,
        "Candidate score distributions",
        fontsize=7.5,
        color=COLORS["ink"],
        ha="left",
        va="bottom",
        clip_on=False,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["selected"],
            markeredgecolor="white",
            markersize=5.0,
            label="Rank 1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.8,
            markersize=4.6,
            label="Rank 2",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["ink"],
            lw=2.4,
            label="IQR; tick = median",
        ),
    ]
    container.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.055),
        ncol=3,
        handlelength=1.4,
        columnspacing=0.9,
        fontsize=5.5,
        borderaxespad=0,
    )

    facet_bottoms = (0.685, 0.365, 0.045)
    for axis, bottom in zip(AXIS_ORDER, facet_bottoms):
        ax = container.inset_axes([0.06, bottom, 0.925, 0.225])
        subset = distributions.loc[distributions["axis"] == axis].sort_values(
            ["axis_rank", "candidate_id"], kind="stable"
        )
        gap = gaps.loc[gaps["axis"] == axis].iloc[0]
        values = subset["axis_score"].to_numpy(dtype=float)
        jitter = _deterministic_jitter(len(subset))
        span = float(values.max() - values.min())
        xpad = max(span * 0.055, 0.003)

        ax.scatter(
            values,
            jitter,
            s=7.5,
            color=COLORS["point"],
            alpha=0.62,
            edgecolors="none",
            zorder=2,
        )
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.plot([q25, q75], [-0.185, -0.185], color=COLORS[axis], lw=2.5, zorder=3)
        ax.plot(
            [median, median],
            [-0.225, -0.145],
            color=COLORS["ink"],
            lw=0.8,
            zorder=4,
        )

        rank1_index = int(np.flatnonzero(subset["axis_rank"].to_numpy() == 1)[0])
        rank2_index = int(np.flatnonzero(subset["axis_rank"].to_numpy() == 2)[0])
        ax.scatter(
            [float(gap["rank2_score"])],
            [jitter[rank2_index]],
            s=35,
            facecolor="white",
            edgecolor=COLORS["ink"],
            linewidth=0.75,
            zorder=5,
        )
        ax.scatter(
            [float(gap["rank1_score"])],
            [jitter[rank1_index]],
            s=43,
            color=COLORS["selected"],
            edgecolor="white",
            linewidth=0.75,
            zorder=6,
        )
        ax.text(
            0.002,
            0.93,
            AXIS_LABELS[axis],
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=COLORS[axis],
            fontweight="bold",
            fontsize=6.3,
        )
        ax.text(
            0.998,
            0.93,
            rf"$\Delta_{{1-2}}$ = {float(gap['rank1_rank2_gap']):.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=COLORS[axis],
            fontsize=5.7,
            fontweight="bold",
        )
        ax.set_xlim(float(values.min() - xpad), float(values.max() + xpad))
        ax.set_ylim(-0.25, 0.25)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="x", labelsize=5.6, pad=1.5)
        ax.grid(axis="x", color="#E9E7F1", lw=0.45, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#868694")
        if axis != AXIS_ORDER[-1]:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Raw Top-2 anchor cosine score", labelpad=2)

    return container


def plot_top20_profiles(ax: mpl.axes.Axes, top20: pd.DataFrame) -> None:
    columns = [f"{axis}_percentile" for axis in AXIS_ORDER]
    matrix = top20[columns].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "v3_percentile",
        ["#F5F3FB", "#C4B9E5", "#6B5EC7"],
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.83, vmax=1.0)

    for row in range(matrix.shape[0] + 1):
        ax.axhline(row - 0.5, color="white", lw=0.5)
    for column in range(matrix.shape[1] + 1):
        ax.axvline(column - 0.5, color="white", lw=0.65)

    for row in range(matrix.shape[0]):
        for column, axis in enumerate(AXIS_ORDER):
            rank = int(round(float(top20.iloc[row][f"{axis}_rank"])))
            ax.text(
                column,
                row,
                str(rank),
                ha="center",
                va="center",
                fontsize=5.2,
                color=contrast_text_color(image, matrix[row, column]),
                fontweight="bold" if row == 0 else "normal",
            )

    ax.set_xticks(range(3), [AXIS_LABELS[axis] for axis in AXIS_ORDER])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)
    for tick, axis in zip(ax.get_xticklabels(), AXIS_ORDER):
        tick.set_color(COLORS[axis])
        tick.set_fontweight("bold")
    labels = [
        f"{int(row.final_rank)} · {row.candidate_id}"
        for row in top20.itertuples(index=False)
    ]
    ax.set_yticks(range(20), labels)
    ax.tick_params(axis="y", length=0, pad=3, labelsize=5.1)
    ax.get_yticklabels()[0].set_color(COLORS["selected"])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            3,
            1,
            fill=False,
            edgecolor=COLORS["selected"],
            linewidth=1.2,
            clip_on=False,
        )
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Top-20 three-axis semantic profiles", loc="left", pad=22)
    add_panel_label(ax, "c", x=-0.14, y=1.10)
    cax = ax.inset_axes([0.10, -0.115, 0.80, 0.025])
    colorbar = plt.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0.83, 0.90, 1.00])
    colorbar.ax.tick_params(labelsize=5.2, length=2, pad=1)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label(
        "Axis percentile",
        fontsize=5.6,
        labelpad=1,
    )


def _anchor_table(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    table = (
        frame[["anchor_id", "anchor_name"]]
        .drop_duplicates()
        .sort_values("anchor_id", kind="stable")
    )
    names = table["anchor_name"].astype(str).tolist()
    labels = [ANCHOR_SHORT[name] for name in names]
    return names, labels


def _style_heatmap_axis(ax: mpl.axes.Axes) -> None:
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_anchor_context(
    figure: mpl.figure.Figure,
    rect: tuple[float, float, float, float],
    anchors: pd.DataFrame,
    anchor_frequency: pd.DataFrame,
) -> mpl.axes.Axes:
    container = figure.add_axes(rect)
    container.set_axis_off()
    container.set_xlim(0, 1)
    container.set_ylim(0, 1)
    container.text(
        0.0,
        1.025,
        "d",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="bottom",
        clip_on=False,
    )
    container.text(
        0.045,
        1.025,
        "Anchor similarity and selection frequency",
        fontsize=7.5,
        color=COLORS["ink"],
        ha="left",
        va="bottom",
        clip_on=False,
    )

    names, anchor_labels = _anchor_table(anchors)
    selected_pivot = anchors.pivot(
        index="axis",
        columns="anchor_name",
        values="cosine_similarity",
    ).loc[list(AXIS_ORDER), names]
    selected_cmap = LinearSegmentedColormap.from_list(
        "v3_anchor_similarity",
        ["#FBF2F7", "#E6A0C1", "#A84FAF"],
    )
    selected_ax = container.inset_axes([0.135, 0.625, 0.785, 0.225])
    selected_image = selected_ax.imshow(
        selected_pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap=selected_cmap,
        vmin=0.89,
        vmax=0.985,
    )
    for row, axis in enumerate(AXIS_ORDER):
        axis_rows = anchors.loc[anchors["axis"] == axis]
        for column, name in enumerate(names):
            entry = axis_rows.loc[axis_rows["anchor_name"] == name].iloc[0]
            if bool(entry["is_top2"]):
                selected_ax.add_patch(
                    Rectangle(
                        (column - 0.46, row - 0.46),
                        0.92,
                        0.92,
                        fill=False,
                        edgecolor=COLORS["selected"],
                        linewidth=1.15,
                    )
                )
                selected_ax.text(
                    column,
                    row,
                    f"{float(entry['cosine_similarity']):.3f}",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color=contrast_text_color(
                        selected_image,
                        float(entry["cosine_similarity"]),
                    ),
                    fontweight="bold",
                )
    selected_ax.set_xticks(
        range(10),
        [label.split()[0] for label in anchor_labels],
        fontsize=5.0,
    )
    selected_ax.tick_params(axis="x", pad=1.2)
    selected_ax.set_yticks(
        range(3), [AXIS_LABELS[axis] for axis in AXIS_ORDER], fontsize=5.4
    )
    for tick, axis in zip(selected_ax.get_yticklabels(), AXIS_ORDER):
        tick.set_color(COLORS[axis])
        tick.set_fontweight("bold")
    _style_heatmap_axis(selected_ax)
    selected_ax.set_title(
        "Anchor cosine similarity",
        loc="left",
        fontsize=6.0,
        pad=4,
    )
    selected_cax = container.inset_axes([0.945, 0.625, 0.012, 0.225])
    selected_bar = plt.colorbar(selected_image, cax=selected_cax)
    selected_bar.set_ticks([0.90, 0.94, 0.98])
    selected_bar.ax.tick_params(labelsize=5.0, length=1.8, pad=1)
    selected_bar.outline.set_linewidth(0.4)
    selected_bar.ax.set_title("cosine", fontsize=5.0, pad=2)

    frequency_pivot = anchor_frequency.pivot(
        index="axis",
        columns="anchor_name",
        values="selection_count",
    ).loc[list(AXIS_ORDER), names]
    frequency_cmap = LinearSegmentedColormap.from_list(
        "v3_anchor_frequency",
        ["#F3F2FB", "#B9B8E4", "#5A61C8"],
    )
    frequency_ax = container.inset_axes([0.135, 0.115, 0.785, 0.285])
    frequency_image = frequency_ax.imshow(
        frequency_pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap=frequency_cmap,
        vmin=0,
        vmax=160,
    )
    for row in range(3):
        for column in range(10):
            value = int(frequency_pivot.iloc[row, column])
            frequency_ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=5.0,
                color=contrast_text_color(frequency_image, value),
            )
    frequency_ax.set_xticks(
        range(10), anchor_labels, rotation=48, ha="right", fontsize=5.0
    )
    frequency_ax.set_yticks(
        range(3), [AXIS_LABELS[axis] for axis in AXIS_ORDER], fontsize=5.4
    )
    for tick, axis in zip(frequency_ax.get_yticklabels(), AXIS_ORDER):
        tick.set_color(COLORS[axis])
        tick.set_fontweight("bold")
    _style_heatmap_axis(frequency_ax)
    frequency_ax.set_title(
        "Global Top-2 anchor frequency",
        loc="left",
        fontsize=6.0,
        pad=4,
    )
    frequency_cax = container.inset_axes([0.945, 0.115, 0.012, 0.285])
    frequency_bar = plt.colorbar(frequency_image, cax=frequency_cax)
    frequency_bar.set_ticks([0, 80, 160])
    frequency_bar.ax.tick_params(labelsize=5.0, length=1.8, pad=1)
    frequency_bar.outline.set_linewidth(0.4)
    frequency_bar.ax.set_title("count", fontsize=5.0, pad=2)

    return container


def create_figure(data_dir: Path) -> mpl.figure.Figure:
    data = load_data(data_dir)
    apply_publication_style()
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
    )

    ax_a = figure.add_axes([0.075, 0.615, 0.295, 0.360])
    plot_rank_migration(ax_a, data["migration"], data["selected"])
    plot_axis_score_facets(
        figure,
        (0.455, 0.615, 0.515, 0.360),
        data["distributions"],
        data["gaps"],
    )

    ax_c = figure.add_axes([0.075, 0.085, 0.295, 0.380])
    plot_top20_profiles(ax_c, data["top20"])
    plot_anchor_context(
        figure,
        (0.405, 0.070, 0.565, 0.430),
        data["anchors"],
        data["anchor_frequency"],
    )
    return figure


def _new_figure(width_mm: float, height_mm: float) -> mpl.figure.Figure:
    return plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))


def create_individual_figures(
    data_dir: Path,
) -> dict[str, mpl.figure.Figure]:
    data = load_data(data_dir)
    apply_publication_style()

    panel_a = _new_figure(89, 86)
    ax_a = panel_a.add_axes([0.18, 0.17, 0.72, 0.745])
    plot_rank_migration(ax_a, data["migration"], data["selected"])

    panel_b = _new_figure(183, 80)
    plot_axis_score_facets(
        panel_b,
        (0.055, 0.12, 0.92, 0.80),
        data["distributions"],
        data["gaps"],
    )

    panel_c = _new_figure(89, 100)
    ax_c = panel_c.add_axes([0.18, 0.16, 0.68, 0.67])
    plot_top20_profiles(ax_c, data["top20"])

    panel_d = _new_figure(183, 85)
    plot_anchor_context(
        panel_d,
        (0.055, 0.08, 0.92, 0.84),
        data["anchors"],
        data["anchor_frequency"],
    )
    return {"a": panel_a, "b": panel_b, "c": panel_c, "d": panel_d}


def save_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    *,
    tiff_dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.tiff",
        output_dir / f"{stem}.png",
    ]
    figure.savefig(outputs[0], format="svg")
    figure.savefig(outputs[1], format="pdf")
    figure.savefig(
        outputs[2],
        format="tiff",
        dpi=tiff_dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(outputs[3], format="png", dpi=300)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "images")
    parser.add_argument("--stem", default="semantic_screening_results_v3")
    parser.add_argument("--tiff-dpi", type=int, default=TIFF_DPI)
    parser.add_argument(
        "--panel-output-dir",
        type=Path,
        default=None,
        help="Directory for separate a-d panel exports (default: OUTPUT/panels).",
    )
    parser.add_argument(
        "--skip-individual-panels",
        action="store_true",
        help="Export only the composite figure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tiff_dpi < 600:
        raise ValueError("--tiff-dpi must be at least 600.")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure = create_figure(data_dir)
    outputs = save_figure(
        figure,
        output_dir,
        args.stem,
        tiff_dpi=args.tiff_dpi,
    )
    plt.close(figure)
    if not args.skip_individual_panels:
        panel_output_dir = (
            args.panel_output_dir.resolve()
            if args.panel_output_dir is not None
            else output_dir / "panels"
        )
        for label, panel in create_individual_figures(data_dir).items():
            outputs.extend(
                save_figure(
                    panel,
                    panel_output_dir,
                    f"{args.stem}_panel_{label}",
                    tiff_dpi=args.tiff_dpi,
                )
            )
            plt.close(panel)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
