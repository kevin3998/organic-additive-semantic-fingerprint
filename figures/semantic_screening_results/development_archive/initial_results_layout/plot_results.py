#!/usr/bin/env python3
"""Create the publication figure for semantic-screening results.

The script reads only publication-facing derived data produced by
``prepare_data.py``. It does not access the formal screening run or private
identity files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# Mandatory publication settings: keep SVG text editable.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'svg.fonttype': 'none'})

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 160

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
    "tie": "#8B78A8",
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


def add_panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.09) -> None:
    ax.text(
        x,
        1.045,
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

    ranking = pd.read_csv(data_dir / "annotated" / "ranking_all_annotated.csv")
    top20 = pd.read_csv(data_dir / "anonymous" / "top20_axis_matrix.csv")
    anchors = pd.read_csv(data_dir / "annotated" / "selected_anchor_profile.csv")
    ties = pd.read_csv(data_dir / "anonymous" / "semantic_tie_groups.csv")
    with (data_dir / "anonymous" / "structure_semantic_correlation.json").open(
        encoding="utf-8"
    ) as handle:
        structure_correlation = json.load(handle)

    require_columns(
        ranking,
        [
            "candidate_id",
            "semantic_score",
            "semantic_rank",
            "structural_score",
            "final_rank",
            "ncm811_percentile",
            "lithium_metal_percentile",
            "lhce_solvation_percentile",
            "is_selected_candidate",
            "display_name",
            "semantic_tie_size",
            "structural_tiebreak_applied",
        ],
        "ranking_all_annotated.csv",
    )
    require_columns(
        top20,
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

    ranking = ranking.sort_values("final_rank", kind="stable").reset_index(drop=True)
    top20 = top20.sort_values("final_rank", kind="stable").reset_index(drop=True)
    if len(ranking) != 200 or ranking["candidate_id"].nunique() != 200:
        raise ValueError("The figure requires exactly 200 unique ranked candidates.")
    if ranking["final_rank"].tolist() != list(range(1, 201)):
        raise ValueError("Final ranks must be the complete sequence 1–200.")
    if len(top20) != 20 or top20["final_rank"].tolist() != list(range(1, 21)):
        raise ValueError("Top20 source table must contain final ranks 1–20.")
    if top20["candidate_id"].tolist() != ranking.head(20)["candidate_id"].tolist():
        raise ValueError("Top20 source table disagrees with the complete ranking.")
    if len(anchors) != 30:
        raise ValueError("Selected anchor profile must contain 3 × 10 rows.")
    if not np.isfinite(
        ranking[
            [
                "semantic_score",
                "structural_score",
                "ncm811_percentile",
                "lithium_metal_percentile",
                "lhce_solvation_percentile",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Ranking table contains non-finite plotting values.")

    selected = ranking.loc[ranking["is_selected_candidate"].astype(bool)]
    if len(selected) != 1 or int(selected.iloc[0]["final_rank"]) != 1:
        raise ValueError("Exactly one post-ranking annotation at final rank 1 is required.")
    if int(selected.iloc[0]["semantic_tie_size"]) != 1:
        raise ValueError("The selected candidate is unexpectedly in a semantic tie.")
    if bool(selected.iloc[0]["structural_tiebreak_applied"]):
        raise ValueError("The selected candidate unexpectedly used structural tie-breaking.")

    tied_ids = set(ties["candidate_id"].astype(str))
    if tied_ids != set(
        ranking.loc[ranking["semantic_tie_size"].astype(int) > 1, "candidate_id"]
    ):
        raise ValueError("Tie-group membership disagrees with the complete ranking.")

    return {
        "ranking": ranking,
        "top20": top20,
        "anchors": anchors,
        "selected": selected.iloc[0],
        "tied_ids": tied_ids,
        "structure_correlation": structure_correlation,
    }


def plot_ranking(
    ax: mpl.axes.Axes, ranking: pd.DataFrame, selected: pd.Series
) -> None:
    """Panel a: complete semantic ranking with three contributing axes."""

    x = ranking["final_rank"].to_numpy()
    ax.axvspan(0.5, 20.5, color=COLORS["top20"], lw=0, zorder=0)
    ax.axvspan(20.5, 50.5, color=COLORS["top50"], lw=0, zorder=0)

    for column, label, color in zip(
        [
            "ncm811_percentile",
            "lithium_metal_percentile",
            "lhce_solvation_percentile",
        ],
        AXIS_LABELS,
        AXIS_COLORS,
        strict=True,
    ):
        ax.scatter(
            x,
            ranking[column],
            s=5.5,
            color=color,
            alpha=0.46,
            edgecolors="none",
            rasterized=False,
            label=label,
            zorder=2,
        )

    ax.plot(
        x,
        ranking["semantic_score"],
        color=COLORS["ink"],
        lw=1.15,
        label="Equal-weight semantic score",
        zorder=3,
    )

    sx = float(selected["final_rank"])
    sy = float(selected["semantic_score"])
    ax.scatter(
        [sx],
        [sy],
        s=34,
        color=COLORS["selected"],
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )
    ax.annotate(
        "3-cyanotetrahydrofuran\nsemantic rank 1",
        xy=(sx, sy),
        xytext=(19, 0.86),
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color=COLORS["selected"],
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["selected"],
            "lw": 0.8,
            "shrinkA": 1,
            "shrinkB": 3,
        },
    )

    ax.text(
        9.5,
        0.055,
        "Top 20",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["selected"],
        fontweight="bold",
    )
    ax.text(
        35.5,
        0.055,
        "Top 50",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["mid"],
    )
    ax.set(
        xlim=(0.5, 200.5),
        ylim=(-0.01, 1.035),
        xlabel="Deterministic final rank",
        ylabel="Percentile / semantic score",
        xticks=[1, 20, 50, 100, 150, 200],
        yticks=[0, 0.25, 0.5, 0.75, 1.0],
    )
    ax.grid(axis="y", color="#E6EAEA", lw=0.55, zorder=0)
    ax.legend(
        loc="upper right",
        ncol=4,
        fontsize=6,
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    ax.set_title("Complete three-axis semantic ranking", loc="left", pad=7)
    add_panel_label(ax, "a", x=-0.055)


def plot_top20_heatmap(
    ax: mpl.axes.Axes, top20: pd.DataFrame, selected: pd.Series
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

    labels = [
        f"{int(row.final_rank):>2}  {row.candidate_id}"
        for row in top20.itertuples(index=False)
    ]
    labels[0] = f"1  {selected['candidate_id']}"
    ax.set_xticks(range(3), AXIS_LABELS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(range(20), labels)
    ax.tick_params(axis="y", length=0, pad=2, labelsize=5.8)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color(COLORS["selected"])
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
    ax.set_title("Top 20 candidates across the three semantic axes", loc="left", pad=22)
    add_panel_label(ax, "b", x=-0.28)

    cax = ax.inset_axes([0.20, -0.115, 0.62, 0.025])
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


def plot_structure_semantic(
    ax: mpl.axes.Axes,
    ranking: pd.DataFrame,
    selected: pd.Series,
    tied_ids: set[str],
    correlation: dict[str, float],
) -> None:
    """Panel d: structural score is diagnostic and only breaks semantic ties."""

    tied = ranking["candidate_id"].isin(tied_ids)
    ax.scatter(
        ranking.loc[~tied, "structural_score"],
        ranking.loc[~tied, "semantic_score"],
        s=11,
        color="#BFC7C8",
        alpha=0.72,
        edgecolors="none",
        label=f"No semantic tie (n={(~tied).sum()})",
        zorder=2,
    )
    ax.scatter(
        ranking.loc[tied, "structural_score"],
        ranking.loc[tied, "semantic_score"],
        s=15,
        facecolors="white",
        edgecolors=COLORS["tie"],
        linewidths=0.65,
        label=f"Exact semantic tie (n={tied.sum()})",
        zorder=3,
    )
    ax.scatter(
        [float(selected["structural_score"])],
        [float(selected["semantic_score"])],
        s=34,
        color=COLORS["selected"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.annotate(
        "Rank 1\n(no tie-break)",
        xy=(float(selected["structural_score"]), float(selected["semantic_score"])),
        xytext=(0.153, 0.91),
        fontsize=6.2,
        color=COLORS["selected"],
        fontweight="bold",
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COLORS["selected"]},
    )
    ax.text(
        0.97,
        0.04,
        rf"Spearman $\rho$ = {float(correlation['spearman']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=COLORS["mid"],
    )
    ax.set(
        xlim=(0.02, 0.25),
        ylim=(-0.02, 1.04),
        xlabel="Structural score (tie-break only)",
        ylabel="Semantic score",
    )
    ax.grid(color="#E8EBEC", lw=0.5, zorder=0)
    ax.legend(
        loc="lower left",
        fontsize=5.7,
        handletextpad=0.35,
        borderaxespad=0.35,
    )
    ax.set_title("Structure does not enter the semantic score", loc="left", pad=6)
    add_panel_label(ax, "d", x=-0.12)


def create_figure(data_dir: Path) -> mpl.figure.Figure:
    """Assemble the four-panel manuscript figure."""

    data = load_data(data_dir)
    apply_publication_style()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
    )
    # Explicit axes positions preserve the 183-mm final width and provide
    # sufficient gutters for the dense Top-20 labels and anchor identifiers.
    ax_a = fig.add_axes([0.075, 0.685, 0.91, 0.265])
    ax_b = fig.add_axes([0.185, 0.105, 0.405, 0.465])
    ax_c = fig.add_axes([0.650, 0.475, 0.335, 0.125])
    ax_d = fig.add_axes([0.650, 0.090, 0.335, 0.215])

    plot_ranking(ax_a, data["ranking"], data["selected"])
    plot_top20_heatmap(ax_b, data["top20"], data["selected"])
    plot_anchor_profile(ax_c, data["anchors"])
    plot_structure_semantic(
        ax_d,
        data["ranking"],
        data["selected"],
        data["tied_ids"],
        data["structure_correlation"],
    )
    return fig


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> list[Path]:
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
        dpi=600,
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
        default="semantic_screening_results",
        help="Output filename stem.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    figure = create_figure(args.data_dir.resolve())
    outputs = save_figure(figure, args.output_dir.resolve(), args.stem)
    plt.close(figure)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
