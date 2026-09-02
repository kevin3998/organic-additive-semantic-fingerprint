#!/usr/bin/env python3
"""Archive the exploratory ECFP structure–semantic diagnostic.

This script is deliberately isolated from the publication mainline. It reads
only the archived structural-diagnostic table in this directory and exports a
standalone figure without changing semantic scores or ranks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 89
FIGURE_HEIGHT_MM = 72
TIFF_DPI = 600

COLORS = {
    "ink": "#202426",
    "mid": "#697377",
    "neutral": "#BFC7C8",
    "selected": "#1E776D",
    "tie": "#8B78A8",
    "grid": "#E8EBEC",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_diagnostic(data_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    ranking = pd.read_csv(data_dir / "structural_similarity_diagnostic.csv")
    with (data_dir / "diagnostic_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    required = {
        "candidate_id",
        "connectivity_smiles",
        "semantic_score",
        "semantic_rank",
        "pure_semantic_final_rank",
        "structural_score",
        "structural_rank",
        "final_rank",
        "semantic_tie_size",
        "structural_tiebreak_applied",
        "is_selected_candidate",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"Ranking table is missing columns: {missing}")
    if len(ranking) != 200 or ranking["candidate_id"].nunique() != 200:
        raise ValueError("Expected 200 unique Stage-2 candidates.")
    if not np.isfinite(
        ranking[["semantic_score", "structural_score"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Diagnostic scores contain non-finite values.")

    expected_pure = ranking.sort_values(
        ["semantic_score", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    ).copy()
    expected_pure["expected_rank"] = np.arange(1, len(expected_pure) + 1, dtype=int)
    if not np.array_equal(
        expected_pure["pure_semantic_final_rank"].to_numpy(dtype=int),
        expected_pure["expected_rank"].to_numpy(dtype=int),
    ):
        raise ValueError("Archived pure-semantic ranks are inconsistent.")

    tied_ids = set(
        ranking.loc[ranking["semantic_tie_size"].astype(int) > 1, "candidate_id"]
    )
    old_top20 = set(ranking.loc[ranking["final_rank"] <= 20, "candidate_id"])
    pure_top20 = set(
        ranking.loc[ranking["pure_semantic_final_rank"] <= 20, "candidate_id"]
    )
    old_top50 = set(ranking.loc[ranking["final_rank"] <= 50, "candidate_id"])
    pure_top50 = set(
        ranking.loc[ranking["pure_semantic_final_rank"] <= 50, "candidate_id"]
    )

    observed_spearman = float(
        ranking["structural_score"].corr(
            ranking["semantic_score"],
            method="spearman",
        )
    )
    if not np.isclose(
        observed_spearman,
        float(summary["structure_semantic_spearman"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("Stored and recomputed Spearman correlations disagree.")
    expected_summary = {
        "used_in_main_ranking": False,
        "candidate_count": len(ranking),
        "candidates_in_ties": len(tied_ids),
        "historical_structural_tiebreak_changed_internal_positions": int(
            (ranking["final_rank"] != ranking["pure_semantic_final_rank"]).sum()
        ),
        "top20_membership_unchanged_without_structure": old_top20 == pure_top20,
        "top50_membership_unchanged_without_structure": old_top50 == pure_top50,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"Archived diagnostic summary mismatch: {field}")
    return ranking, summary


def create_figure(ranking: pd.DataFrame, summary: dict[str, object]) -> mpl.figure.Figure:
    apply_style()
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
    )
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.78)

    tied = ranking["semantic_tie_size"].astype(int) > 1
    ax.scatter(
        ranking.loc[~tied, "structural_score"],
        ranking.loc[~tied, "semantic_score"],
        s=12,
        color=COLORS["neutral"],
        alpha=0.72,
        edgecolors="none",
        label=f"No semantic tie (n={(~tied).sum()})",
        zorder=2,
    )
    ax.scatter(
        ranking.loc[tied, "structural_score"],
        ranking.loc[tied, "semantic_score"],
        s=16,
        facecolors="white",
        edgecolors=COLORS["tie"],
        linewidths=0.7,
        label=f"Exact semantic tie (n={tied.sum()})",
        zorder=3,
    )

    selected = ranking.loc[ranking["is_selected_candidate"].astype(bool)]
    if len(selected) != 1:
        raise ValueError("Expected exactly one selected candidate annotation.")
    selected_row = selected.iloc[0]
    ax.scatter(
        [float(selected_row["structural_score"])],
        [float(selected_row["semantic_score"])],
        s=38,
        color=COLORS["selected"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.annotate(
        "Semantic rank 1",
        xy=(
            float(selected_row["structural_score"]),
            float(selected_row["semantic_score"]),
        ),
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
        rf"Spearman $\rho$ = {float(summary['structure_semantic_spearman']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=COLORS["mid"],
    )
    ax.set(
        xlim=(0.02, 0.25),
        ylim=(-0.02, 1.04),
        xlabel="ECFP structural score",
        ylabel="Semantic score",
    )
    ax.grid(color=COLORS["grid"], lw=0.5, zorder=0)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        fontsize=5.5,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax.set_title(
        "Exploratory structure–semantic diagnostic",
        loc="left",
        pad=28,
    )
    ax.text(
        0.0,
        1.17,
        "Archived separately; not used in the publication ranking",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=COLORS["mid"],
    )
    return fig


def save_figure(fig: mpl.figure.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "structural_similarity_diagnostic"
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf")
    fig.savefig(
        stem.with_suffix(".tiff"),
        format="tiff",
        dpi=TIFF_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=300)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Archive the exploratory structure–semantic diagnostic."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ranking, summary = load_diagnostic(args.data_dir.resolve())
    figure = create_figure(ranking, summary)
    save_figure(figure, output_dir / "images")
    plt.close(figure)
    print(output_dir / "images" / "structural_similarity_diagnostic.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
