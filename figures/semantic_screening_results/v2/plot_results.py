#!/usr/bin/env python3
"""Create the v2 publication figure for semantic-screening results.

The script reads only publication-facing derived data produced by
``prepare_data.py``. It does not access the formal screening run or private
identity files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Mandatory publication settings: keep SVG text editable.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'svg.fonttype': 'none'})

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 190
TIFF_DPI = 3200

COLORS = {
    "ink": "#202426",
    "mid": "#697377",
    "light": "#D8DEDF",
    "very_light": "#F3F6F6",
    "selected": "#1E776D",
    "selected_light": "#D8ECE8",
    "ncm811": "#B26776",
    "lithium": "#2F8C87",
    "lhce": "#7768A6",
    "top20": "#E2F0ED",
    "top50": "#F1F7F5",
}

AXIS_KEYS = ["ncm811", "lithium_metal", "lhce_solvation"]
AXIS_LABELS = ["NCM811", "Li metal", "LHCE"]
AXIS_COLORS = [COLORS["ncm811"], COLORS["lithium"], COLORS["lhce"]]


def apply_publication_style() -> None:
    """Apply a compact Nature-family manuscript style."""

    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(
    ax: mpl.axes.Axes,
    label: str,
    x: float = -0.09,
    y: float = 1.045,
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
    )


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_data(data_dir: Path) -> dict[str, object]:
    """Load and validate the publication-facing tables used in the figure."""

    ranking_columns = [
        "candidate_id",
        "connectivity_smiles",
        *[
            field
            for axis in AXIS_KEYS
            for field in (
                f"{axis}_score",
                f"{axis}_rank",
                f"{axis}_percentile",
            )
        ],
        "semantic_score",
        "semantic_rank",
        "final_rank",
        "is_selected_candidate",
        "display_name",
    ]
    ranking = pd.read_csv(
        data_dir / "annotated" / "ranking_all_annotated.csv",
        usecols=ranking_columns,
    )
    stage1 = pd.read_csv(data_dir / "anonymous" / "stage1_all.csv")
    source_top20 = pd.read_csv(data_dir / "anonymous" / "top20_axis_matrix.csv")
    anchors = pd.read_csv(data_dir / "annotated" / "selected_anchor_profile.csv")

    require_columns(
        ranking,
        [
            "candidate_id",
            "semantic_score",
            "semantic_rank",
            "final_rank",
            "ncm811_rank",
            "ncm811_percentile",
            "lithium_metal_rank",
            "lithium_metal_percentile",
            "lhce_solvation_rank",
            "lhce_solvation_percentile",
            "is_selected_candidate",
            "display_name",
        ],
        "ranking_all_annotated.csv",
    )
    require_columns(
        stage1,
        [
            "candidate_id",
            "stage1_score",
            "stage1_rank",
            "selected",
        ],
        "stage1_all.csv",
    )
    require_columns(
        source_top20,
        [
            "candidate_id",
            "final_rank",
            "ncm811_percentile",
            "lithium_metal_percentile",
            "lhce_solvation_percentile",
        ],
        "top20_axis_matrix.csv",
    )
    require_columns(
        anchors,
        [
            "axis",
            "axis_label",
            "anchor_id",
            "anchor_name",
            "cosine_similarity",
            "is_top2",
        ],
        "selected_anchor_profile.csv",
    )

    selected_stage1 = stage1.loc[
        stage1["selected"].astype(bool), ["candidate_id", "stage1_rank", "stage1_score"]
    ].copy()
    if len(stage1) != 800 or stage1["candidate_id"].nunique() != 800:
        raise ValueError("Stage 1 source must contain 800 unique candidates.")
    if len(selected_stage1) != 200:
        raise ValueError("Stage 1 source must contain exactly 200 selected candidates.")

    ranking = ranking.merge(
        selected_stage1,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    source_order = ranking.sort_values("final_rank", kind="stable").reset_index(
        drop=True
    )
    source_top20 = source_top20.sort_values("final_rank", kind="stable").reset_index(
        drop=True
    )
    if len(ranking) != 200 or ranking["candidate_id"].nunique() != 200:
        raise ValueError("The figure requires exactly 200 unique ranked candidates.")
    if source_order["final_rank"].tolist() != list(range(1, 201)):
        raise ValueError("Source final ranks must be the complete sequence 1–200.")
    if sorted(ranking["stage1_rank"].astype(int).tolist()) != list(range(1, 201)):
        raise ValueError("Selected Stage 1 ranks must be the complete sequence 1–200.")
    if (
        len(source_top20) != 20
        or source_top20["final_rank"].tolist() != list(range(1, 21))
    ):
        raise ValueError("Top20 source table must contain final ranks 1–20.")
    if source_top20["candidate_id"].tolist() != source_order.head(20)[
        "candidate_id"
    ].tolist():
        raise ValueError("Top20 source table disagrees with the complete ranking.")

    # Publication mainline: pure semantic score, with anonymous candidate ID
    # used only as a deterministic fallback for exact score ties.
    ranking = ranking.sort_values(
        ["semantic_score", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking["final_rank"] = np.arange(1, len(ranking) + 1, dtype=int)
    top20 = ranking.head(20).copy()
    if top20["candidate_id"].tolist() != source_top20["candidate_id"].tolist():
        raise ValueError(
            "Pure-semantic Top 20 differs from the locked source Top 20."
        )
    if len(anchors) != 30:
        raise ValueError("Selected anchor profile must contain 3 × 10 rows.")
    if not np.isfinite(
        ranking[
            [
                "semantic_score",
                "ncm811_percentile",
                "lithium_metal_percentile",
                "lhce_solvation_percentile",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Ranking table contains non-finite plotting values.")

    selected = ranking.loc[ranking["is_selected_candidate"].astype(bool)]
    if (
        len(selected) != 1
        or int(selected.iloc[0]["final_rank"]) != 1
        or float(selected.iloc[0]["semantic_rank"]) != 1.0
    ):
        raise ValueError("Exactly one pure-semantic rank-1 annotation is required.")
    if int(selected.iloc[0]["stage1_rank"]) != 141:
        raise ValueError("The rank-1 candidate does not have the expected Stage 1 rank 141.")

    stage1_final_spearman = float(
        ranking["stage1_rank"].corr(ranking["final_rank"], method="spearman")
    )
    if not np.isfinite(stage1_final_spearman):
        raise ValueError("Stage 1-to-final rank correlation is non-finite.")

    return {
        "ranking": ranking,
        "top20": top20,
        "anchors": anchors,
        "selected": selected.iloc[0],
        "stage1_final_spearman": stage1_final_spearman,
    }


def plot_rank_migration(
    ax: mpl.axes.Axes,
    ranking: pd.DataFrame,
    selected: pd.Series,
) -> None:
    """Panel a upper: rank migration from the prescreen to semantic ranking."""

    stage1_rank = ranking["stage1_rank"].to_numpy(dtype=float)
    final_rank = ranking["final_rank"].to_numpy(dtype=float)
    ax.axhspan(0.5, 20.5, color=COLORS["top20"], lw=0, zorder=0)
    ax.plot(
        [1, 200],
        [1, 200],
        color="#AEB6B8",
        lw=0.8,
        ls=(0, (3, 2)),
        zorder=1,
    )
    ax.scatter(
        stage1_rank,
        final_rank,
        s=13,
        color="#AEB8BA",
        alpha=0.70,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )

    sx = float(selected["stage1_rank"])
    sy = float(selected["final_rank"])
    ax.scatter(
        [sx],
        [sy],
        s=48,
        color=COLORS["selected"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.annotate(
        "3-cyanotetrahydrofuran\nStage 1 rank 141 \u2192 semantic rank 1",
        xy=(sx, sy),
        xytext=(sx, -10),
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLORS["selected"],
        fontweight="bold",
        annotation_clip=False,
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["selected"],
            "lw": 0.8,
            "shrinkA": 2,
            "shrinkB": 4,
        },
    )
    ax.text(
        6,
        12,
        "Final Top 20",
        ha="left",
        va="center",
        fontsize=6.2,
        color=COLORS["selected"],
        fontweight="bold",
    )
    ax.set(
        xlim=(0.5, 200.5),
        ylim=(205.0, -30.0),
        xlabel="Stage 1 anchor-prescreen rank",
        ylabel="Final semantic rank",
        xticks=[1, 50, 100, 150, 200],
        yticks=[1, 20, 50, 100, 150, 200],
    )
    ax.grid(color="#E6EAEA", lw=0.5, zorder=0)
    ax.set_title("Semantic reranking of the prescreened candidate set", loc="left", pad=7)
    add_panel_label(ax, "a", x=-0.085)


def plot_semantic_landscape(
    ax: mpl.axes.Axes,
    cbar_ax: mpl.axes.Axes,
    ranking: pd.DataFrame,
) -> None:
    """Panel a lower: three-axis fingerprints and their equal-weight mean."""

    matrix = ranking[
        [
            "ncm811_percentile",
            "lithium_metal_percentile",
            "lhce_solvation_percentile",
            "semantic_score",
        ]
    ].to_numpy(dtype=float).T
    cmap = LinearSegmentedColormap.from_list(
        "landscape_map", ["#F2F3F4", "#BFD7D4", "#287D76"]
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axvline(19.5, color=COLORS["selected"], lw=0.8, ls=(0, (2, 2)))
    ax.axvline(49.5, color=COLORS["mid"], lw=0.7, ls=(0, (2, 2)))
    ax.add_patch(
        Rectangle(
            (-0.49, -0.49),
            0.98,
            3.98,
            fill=False,
            edgecolor=COLORS["selected"],
            linewidth=1.2,
            clip_on=False,
        )
    )
    for row in range(5):
        ax.axhline(row - 0.5, color="white", lw=0.55)
    ax.set_yticks(
        range(4),
        ["NCM811", "Li metal", "LHCE", "Semantic mean"],
    )
    ax.set_xticks(
        [0, 19, 49, 99, 149, 199],
        ["1", "20", "50", "100", "150", "200"],
    )
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_xlabel("Deterministic final rank", labelpad=2)
    ax.set_title(
        "Final-rank-ordered semantic fingerprint landscape",
        loc="left",
        pad=4,
        fontsize=7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(image, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label("Percentile / mean", fontsize=6, labelpad=2)


def plot_top20_heatmap(
    ax: mpl.axes.Axes,
    top20: pd.DataFrame,
    selected: pd.Series,
) -> None:
    """Panel b: Top-20 three-axis percentile matrix."""

    matrix = top20[
        [
            "ncm811_percentile",
            "lithium_metal_percentile",
            "lhce_solvation_percentile",
        ]
    ].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "percentile_map", ["#F2F3F4", "#BCD7D5", "#287D76"]
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.84, vmax=1.0)

    for row in range(matrix.shape[0] + 1):
        ax.axhline(row - 0.5, color="white", lw=0.55)
    for col in range(matrix.shape[1] + 1):
        ax.axvline(col - 0.5, color="white", lw=0.7)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rank_column = [
                "ncm811_rank",
                "lithium_metal_rank",
                "lhce_solvation_rank",
            ][j]
            rank_value = int(round(float(top20.iloc[i][rank_column])))
            text_color = "white" if matrix[i, j] >= 0.95 else COLORS["ink"]
            ax.text(
                j,
                i,
                str(rank_value),
                ha="center",
                va="center",
                fontsize=5.4,
                color=text_color,
                fontweight="bold" if i == 0 else "normal",
            )

    ax.set_xticks(range(3), AXIS_LABELS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)
    final_ranks = top20["final_rank"].astype(int).tolist()
    ax.set_yticks(range(len(final_ranks)), [str(rank) for rank in final_ranks])
    ax.tick_params(axis="y", length=0, pad=3, labelsize=5.2)
    ax.set_ylabel("Final rank", fontsize=6, labelpad=5)
    for tick_label, rank in zip(ax.get_yticklabels(), final_ranks):
        if rank == int(selected["final_rank"]):
            tick_label.set_color(COLORS["selected"])
            tick_label.set_fontweight("bold")
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            3,
            1,
            fill=False,
            edgecolor=COLORS["selected"],
            linewidth=1.25,
            clip_on=False,
        )
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "Top 20 three-axis semantic profiles",
        loc="left",
        pad=27,
        fontsize=7.5,
    )
    add_panel_label(ax, "b", x=-0.08, y=1.12)

    cax = ax.inset_axes([0.20, -0.090, 0.62, 0.025])
    cbar = plt.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.85, 0.90, 0.95, 1.00])
    cbar.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label("Axis percentile (cell text: axis rank)", fontsize=6, labelpad=1)


def plot_anchor_profile(ax: mpl.axes.Axes, anchors: pd.DataFrame) -> None:
    """Panel c: selected candidate's similarity to the ten reference anchors."""

    anchor_order = (
        anchors[["anchor_id", "anchor_name"]]
        .drop_duplicates()
        .sort_values("anchor_id")["anchor_name"]
        .tolist()
    )
    pivot = anchors.pivot(
        index="axis", columns="anchor_name", values="cosine_similarity"
    ).loc[AXIS_KEYS, anchor_order]
    cmap = LinearSegmentedColormap.from_list(
        "cosine_map", ["#F3F3F4", "#CBC5DE", "#71639B"]
    )
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=0.89, vmax=0.985)

    for i, axis_key in enumerate(AXIS_KEYS):
        axis_rows = anchors.loc[anchors["axis"] == axis_key]
        for j, anchor_name in enumerate(anchor_order):
            value = float(pivot.loc[axis_key, anchor_name])
            if bool(
                axis_rows.loc[
                    axis_rows["anchor_name"] == anchor_name, "is_top2"
                ].iloc[0]
            ):
                ax.add_patch(
                    Rectangle(
                        (j - 0.47, i - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        edgecolor=COLORS["ink"],
                        linewidth=1.1,
                    )
                )
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white",
                    fontweight="bold",
                )

    anchor_short_labels = {
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
    ax.set_xticks(
        range(10),
        [anchor_short_labels[name] for name in anchor_order],
        rotation=55,
        ha="right",
    )
    ax.set_yticks(range(3), AXIS_LABELS)
    ax.tick_params(axis="x", length=0, pad=2, labelsize=5.0)
    ax.tick_params(axis="y", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Rank-1 candidate: anchor similarity profile",
        loc="left",
        pad=6,
    )
    add_panel_label(ax, "c", x=-0.12)

    cax = ax.inset_axes([0.19, -0.73, 0.64, 0.06])
    cbar = plt.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.90, 0.94, 0.98])
    cbar.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label("Cosine similarity (outlined: axis Top-2)", fontsize=6, labelpad=1)


def plot_axis_concordance(
    ax: mpl.axes.Axes,
    ranking: pd.DataFrame,
) -> None:
    """Panel d: pairwise rank concordance among the three semantic axes."""

    score_columns = {
        "NCM811–Li metal": ("ncm811_percentile", "lithium_metal_percentile"),
        "Li metal–LHCE": ("lithium_metal_percentile", "lhce_solvation_percentile"),
        "NCM811–LHCE": ("ncm811_percentile", "lhce_solvation_percentile"),
    }
    labels = list(score_columns)
    correlations = np.array(
        [
            ranking[left].corr(ranking[right], method="spearman")
            for left, right in score_columns.values()
        ],
        dtype=float,
    )
    if not np.isfinite(correlations).all():
        raise ValueError("Semantic-axis Spearman correlation is non-finite.")

    y_positions = np.arange(len(labels))
    ax.hlines(
        y_positions,
        0.88,
        correlations,
        color="#C8D0D1",
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        correlations,
        y_positions,
        s=28,
        color=COLORS["selected"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for y_position, value in zip(y_positions, correlations):
        ax.text(
            value + 0.004,
            y_position,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=6,
            color=COLORS["ink"],
        )

    ax.set_xlim(0.88, 1.0)
    ax.set_ylim(-0.55, len(labels) - 0.45)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xticks([0.88, 0.92, 0.96, 1.00])
    ax.set_xlabel(r"Spearman $\rho$")
    ax.tick_params(axis="y", length=0, pad=3, labelsize=5.6)
    ax.grid(axis="x", color="#E8EBEC", lw=0.5, zorder=0)
    ax.text(
        0.99,
        0.04,
        "n = 200 candidates",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=COLORS["mid"],
    )
    ax.set_title("Concordance across semantic axes", loc="left", pad=7)
    add_panel_label(ax, "d", x=-0.16, y=1.06)


def create_figure(data_dir: Path) -> mpl.figure.Figure:
    """Assemble the aligned four-panel manuscript figure."""

    data = load_data(data_dir)
    apply_publication_style()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
    )
    # One master grid: panels a and b share the same outer left edge.
    ax_a = fig.add_axes([0.090, 0.800, 0.895, 0.160])
    ax_a_heat = fig.add_axes([0.090, 0.675, 0.850, 0.055])
    ax_a_cbar = fig.add_axes([0.957, 0.675, 0.008, 0.055])

    ax_b = fig.add_axes([0.090, 0.090, 0.490, 0.450])
    ax_c = fig.add_axes([0.685, 0.490, 0.300, 0.090])
    ax_d = fig.add_axes([0.685, 0.080, 0.300, 0.220])

    plot_rank_migration(
        ax_a,
        data["ranking"],
        data["selected"],
    )
    plot_semantic_landscape(ax_a_heat, ax_a_cbar, data["ranking"])
    plot_top20_heatmap(
        ax_b,
        data["top20"],
        data["selected"],
    )
    plot_anchor_profile(ax_c, data["anchors"])
    plot_axis_concordance(ax_d, data["ranking"])
    return fig


def save_figure(
    fig: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    *,
    tiff_dpi: int,
) -> list[Path]:
    """Export editable and high-resolution manuscript formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.tiff",
        output_dir / f"{stem}.png",
    ]
    fig.savefig(outputs[0], format="svg")
    fig.savefig(outputs[1], format="pdf")
    fig.savefig(
        outputs[2],
        format="tiff",
        dpi=tiff_dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(outputs[3], format="png", dpi=300)
    return outputs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot the Nature-style semantic screening result figure."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir.parent / "data",
        help="Publication-facing derived data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "images",
        help="Figure output directory.",
    )
    parser.add_argument(
        "--stem",
        default="semantic_screening_results_v2",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--tiff-dpi",
        type=int,
        default=TIFF_DPI,
        help="TIFF export resolution; defaults to 3200 dpi.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tiff_dpi < 600:
        raise ValueError("--tiff-dpi must be at least 600 for manuscript export.")
    figure = create_figure(args.data_dir.resolve())
    outputs = save_figure(
        figure,
        args.output_dir.resolve(),
        args.stem,
        tiff_dpi=args.tiff_dpi,
    )
    plt.close(figure)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
