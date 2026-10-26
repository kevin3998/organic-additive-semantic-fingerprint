#!/usr/bin/env python3
"""Prepare the publication data bundle for the semantic-screening figure."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = HERE.parent / "data"
DEFAULT_OUTPUT_DIR = HERE / "data"

INPUT_FILES = {
    "ranking": "annotated/ranking_all_annotated.csv",
    "stage1": "anonymous/stage1_all.csv",
    "axis_scores": "anonymous/axis_scores_long.csv",
    "selected_anchors": "annotated/selected_anchor_profile.csv",
    "anchor_frequency": "anonymous/anchor_selection_frequency.csv",
    "selected_summary": "annotated/selected_candidate_summary.json",
}

OUTPUT_FILES = (
    "rank_migration.csv",
    "axis_score_distribution.csv",
    "axis_rank1_rank2_summary.csv",
    "semantic_aggregation_trajectories.csv",
    "selected_anchor_profile.csv",
    "anchor_selection_frequency.csv",
    "selected_molecule.json",
    "summary.json",
)

AXIS_ORDER = ("ncm811", "lithium_metal", "lhce_solvation")
AXIS_LABELS = {
    "ncm811": "NCM811",
    "lithium_metal": "Li metal",
    "lhce_solvation": "LHCE",
}
FORBIDDEN_PUBLIC_FIELDS = {
    "structural_score",
    "structural_rank",
    "structural_tiebreak_applied",
    "target",
    "role",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty table: {path.name}")
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite values.")


def verify_source_manifest(source_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "data_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("run_id") != "20260724-organic-additive-v4-01":
        raise ValueError("Unexpected source run ID.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Source data manifest has no output records.")
    for relative in INPUT_FILES.values():
        record = outputs.get(relative)
        if not isinstance(record, dict):
            raise ValueError(f"Source manifest does not record {relative}.")
        path = source_dir / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required regular source file is missing: {relative}.")
        if path.stat().st_size != int(record["size"]):
            raise ValueError(f"Source size mismatch: {relative}.")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"Source SHA256 mismatch: {relative}.")
    return manifest


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("Boolean source column contains invalid values.")
    return normalized.eq("true")


def derive_tables(source_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    verify_source_manifest(source_dir)
    ranking = pd.read_csv(source_dir / INPUT_FILES["ranking"])
    stage1 = pd.read_csv(source_dir / INPUT_FILES["stage1"])
    axis_scores = pd.read_csv(source_dir / INPUT_FILES["axis_scores"])
    anchors = pd.read_csv(source_dir / INPUT_FILES["selected_anchors"])
    anchor_frequency = pd.read_csv(source_dir / INPUT_FILES["anchor_frequency"])
    selected_summary = load_json(source_dir / INPUT_FILES["selected_summary"])

    require_columns(
        ranking,
        [
            "candidate_id",
            "connectivity_smiles",
            "semantic_score",
            "semantic_rank",
            "final_rank",
            "is_selected_candidate",
            "display_name",
            *[
                f"{axis}_{suffix}"
                for axis in AXIS_ORDER
                for suffix in ("score", "rank", "percentile")
            ],
        ],
        "ranking",
    )
    require_columns(
        stage1,
        ["candidate_id", "stage1_score", "stage1_rank", "selected"],
        "Stage 1",
    )
    require_columns(
        axis_scores,
        [
            "candidate_id",
            "connectivity_smiles",
            "axis",
            "axis_label",
            "axis_score",
            "axis_rank",
            "axis_percentile",
            "semantic_score",
            "semantic_rank",
            "final_rank",
        ],
        "axis scores",
    )
    require_columns(
        anchors,
        [
            "candidate_id",
            "connectivity_smiles",
            "final_rank",
            "axis",
            "axis_label",
            "anchor_id",
            "anchor_name",
            "cosine_similarity",
            "top2_order",
            "is_top2",
            "display_name",
        ],
        "selected-anchor profile",
    )
    require_columns(
        anchor_frequency,
        [
            "axis",
            "axis_label",
            "anchor_id",
            "anchor_name",
            "selection_count",
            "selection_fraction_of_slots",
            "candidate_fraction",
        ],
        "anchor-selection frequency",
    )

    if len(ranking) != 200 or ranking["candidate_id"].nunique() != 200:
        raise ValueError("Ranking must contain 200 unique candidates.")
    if ranking["connectivity_smiles"].nunique() != 200:
        raise ValueError("Ranking SMILES must be unique.")
    if sorted(ranking["final_rank"].astype(int)) != list(range(1, 201)):
        raise ValueError("Final ranks must be the complete sequence 1–200.")
    ranking = ranking.sort_values("final_rank", kind="stable").reset_index(drop=True)
    expected_order = ranking.sort_values(
        ["semantic_score", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )["candidate_id"].tolist()
    if ranking["candidate_id"].tolist() != expected_order:
        raise ValueError("Ranking is not pure-semantic deterministic order.")

    stage1_selected = stage1.loc[_bool_series(stage1["selected"])].copy()
    if len(stage1) != 800 or stage1["candidate_id"].nunique() != 800:
        raise ValueError("Stage 1 must contain 800 unique candidates.")
    if len(stage1_selected) != 200:
        raise ValueError("Stage 1 must retain exactly 200 candidates.")
    if set(stage1_selected["candidate_id"]) != set(ranking["candidate_id"]):
        raise ValueError("Stage 1 and semantic candidate sets differ.")

    selected_mask = _bool_series(ranking["is_selected_candidate"])
    selected = ranking.loc[selected_mask]
    if len(selected) != 1:
        raise ValueError("Exactly one post-ranking selected candidate is required.")
    selected_row = selected.iloc[0]
    if (
        int(selected_row["final_rank"]) != 1
        or float(selected_row["semantic_rank"]) != 1.0
        or str(selected_row["display_name"]) != "3-cyanotetrahydrofuran"
    ):
        raise ValueError("The selected annotation is not the pure-semantic rank 1.")

    numeric_ranking_columns = [
        "semantic_score",
        "semantic_rank",
        "final_rank",
        *[
            f"{axis}_{suffix}"
            for axis in AXIS_ORDER
            for suffix in ("score", "rank", "percentile")
        ],
    ]
    require_finite(ranking, numeric_ranking_columns, "ranking")

    migration = ranking[
        [
            "candidate_id",
            "connectivity_smiles",
            "semantic_score",
            "semantic_rank",
            "final_rank",
            "is_selected_candidate",
            "display_name",
        ]
    ].merge(
        stage1_selected[["candidate_id", "stage1_score", "stage1_rank"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    migration = migration.sort_values("final_rank", kind="stable").reset_index(drop=True)
    migration["stage1_ranking_percentile"] = (
        (200.0 - migration["stage1_rank"].to_numpy(dtype=float)) / 199.0 * 100.0
    )
    migration["final_semantic_ranking_percentile"] = (
        (200.0 - migration["final_rank"].to_numpy(dtype=float)) / 199.0 * 100.0
    )
    if not np.allclose(
        migration.loc[migration["is_selected_candidate"], "stage1_ranking_percentile"],
        [(200.0 - 141.0) / 199.0 * 100.0],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Selected Stage 1 ranking percentile is inconsistent.")
    if int(migration.loc[migration["is_selected_candidate"], "stage1_rank"].iloc[0]) != 141:
        raise ValueError("Selected candidate must have Stage 1 rank 141.")

    if (
        len(axis_scores) != 600
        or axis_scores["candidate_id"].nunique() != 200
        or set(axis_scores["axis"]) != set(AXIS_ORDER)
    ):
        raise ValueError("Axis-score table must contain 200 × 3 records.")
    if not axis_scores.groupby("candidate_id").size().eq(3).all():
        raise ValueError("Each candidate must have exactly three axis scores.")
    require_finite(
        axis_scores,
        ["axis_score", "axis_rank", "axis_percentile", "semantic_score"],
        "axis scores",
    )
    selection_lookup = ranking.set_index("candidate_id")[
        ["is_selected_candidate", "display_name"]
    ]
    axis_distribution = axis_scores.copy()
    axis_distribution["is_selected_candidate"] = axis_distribution[
        "candidate_id"
    ].map(selection_lookup["is_selected_candidate"])
    axis_distribution["display_name"] = axis_distribution["candidate_id"].map(
        selection_lookup["display_name"]
    )
    axis_distribution["axis_order"] = axis_distribution["axis"].map(
        {axis: index + 1 for index, axis in enumerate(AXIS_ORDER)}
    )
    if axis_distribution["axis_order"].isna().any():
        raise ValueError("Axis order mapping failed.")
    axis_distribution = axis_distribution.sort_values(
        ["axis_order", "axis_rank", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)

    trajectory_fields = [
        "candidate_id",
        "connectivity_smiles",
        "final_rank",
        *[f"{axis}_percentile" for axis in AXIS_ORDER],
        "semantic_score",
        "semantic_rank",
        "is_selected_candidate",
        "display_name",
    ]
    trajectories = ranking[trajectory_fields].copy()
    axis_percentiles = trajectories[
        [f"{axis}_percentile" for axis in AXIS_ORDER]
    ].to_numpy(dtype=float)
    expected_semantic = axis_percentiles.mean(axis=1)
    if not np.allclose(
        trajectories["semantic_score"].to_numpy(dtype=float),
        expected_semantic,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Semantic score is not the equal-weight axis-percentile mean.")
    trajectories = trajectories.sort_values("final_rank", kind="stable").reset_index(
        drop=True
    )

    if (
        len(anchors) != 30
        or anchors["candidate_id"].nunique() != 1
        or set(anchors["axis"]) != set(AXIS_ORDER)
        or anchors["anchor_id"].nunique() != 10
    ):
        raise ValueError("Selected-anchor profile must contain 3 × 10 values.")
    if str(anchors["candidate_id"].iloc[0]) != str(selected_row["candidate_id"]):
        raise ValueError("Anchor profile does not describe the selected candidate.")
    anchors["is_top2"] = _bool_series(anchors["is_top2"])
    if not anchors.groupby("axis")["is_top2"].sum().eq(2).all():
        raise ValueError("Each axis must have exactly two selected anchors.")
    require_finite(anchors, ["cosine_similarity"], "selected-anchor profile")
    anchors["axis_order"] = anchors["axis"].map(
        {axis: index + 1 for index, axis in enumerate(AXIS_ORDER)}
    )
    anchors = anchors.sort_values(
        ["axis_order", "anchor_id"], kind="stable"
    ).reset_index(drop=True)

    if (
        len(anchor_frequency) != 30
        or set(anchor_frequency["axis"]) != set(AXIS_ORDER)
        or anchor_frequency["anchor_id"].nunique() != 10
        or not anchor_frequency.groupby("axis").size().eq(10).all()
    ):
        raise ValueError("Anchor frequency must contain 3 × 10 values.")
    require_finite(
        anchor_frequency,
        [
            "selection_count",
            "selection_fraction_of_slots",
            "candidate_fraction",
        ],
        "anchor-selection frequency",
    )
    if not anchor_frequency.groupby("axis")["selection_count"].sum().eq(400).all():
        raise ValueError("Each axis must contain 400 Top-2 anchor selections.")
    expected_slot_fraction = (
        anchor_frequency["selection_count"].to_numpy(dtype=float) / 400.0
    )
    expected_candidate_fraction = (
        anchor_frequency["selection_count"].to_numpy(dtype=float) / 200.0
    )
    if not np.allclose(
        anchor_frequency["selection_fraction_of_slots"].to_numpy(dtype=float),
        expected_slot_fraction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Anchor slot fractions are inconsistent with counts.")
    if not np.allclose(
        anchor_frequency["candidate_fraction"].to_numpy(dtype=float),
        expected_candidate_fraction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Anchor candidate fractions are inconsistent with counts.")
    anchor_frequency["axis_order"] = anchor_frequency["axis"].map(
        {axis: index + 1 for index, axis in enumerate(AXIS_ORDER)}
    )
    anchor_frequency = anchor_frequency.sort_values(
        ["axis_order", "anchor_id"], kind="stable"
    ).reset_index(drop=True)

    if (
        selected_summary.get("candidate_id") != str(selected_row["candidate_id"])
        or int(selected_summary.get("final_rank", -1)) != 1
        or selected_summary.get("display_name") != "3-cyanotetrahydrofuran"
        or selected_summary.get("identity_used_for_ranking") is not False
    ):
        raise ValueError("Selected-molecule identity does not match public rank 1.")
    if any(
        field in selected_summary
        for field in ("target", "role", "structural_score", "structural_rank")
    ):
        raise ValueError("Forbidden identity or structural field in molecule summary.")
    selected_molecule = {
        "candidate_id": str(selected_summary["candidate_id"]),
        "display_name": str(selected_summary["display_name"]),
        "systematic_name": str(selected_summary["systematic_name"]),
        "connectivity_smiles": str(selected_summary["connectivity_smiles"]),
        "molecular_formula": str(selected_summary["molecular_formula"]),
        "molecular_weight": float(selected_summary["molecular_weight"]),
        "final_rank": int(selected_summary["final_rank"]),
        "identity_annotation_stage": "after_anonymous_final_ranking",
        "identity_used_for_ranking": False,
    }

    gap_rows: list[dict[str, object]] = []
    for axis_index, axis in enumerate(AXIS_ORDER, start=1):
        subset = axis_distribution.loc[axis_distribution["axis"] == axis].sort_values(
            ["axis_rank", "candidate_id"], kind="stable"
        )
        first = subset.iloc[0]
        second = subset.iloc[1]
        if int(first["axis_rank"]) != 1 or int(second["axis_rank"]) != 2:
            raise ValueError(f"Axis {axis} does not have unique ranks 1 and 2.")
        values = subset["axis_score"].to_numpy(dtype=float)
        gap_rows.append(
            {
                "axis": axis,
                "axis_label": AXIS_LABELS[axis],
                "axis_order": axis_index,
                "rank1_candidate_id": str(first["candidate_id"]),
                "rank1_score": float(first["axis_score"]),
                "rank2_candidate_id": str(second["candidate_id"]),
                "rank2_score": float(second["axis_score"]),
                "rank1_rank2_gap": float(first["axis_score"] - second["axis_score"]),
                "minimum": float(values.min()),
                "p25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "p75": float(np.quantile(values, 0.75)),
                "maximum": float(values.max()),
            }
        )
    axis_gap_summary = pd.DataFrame(gap_rows)

    for frame in (
        migration,
        axis_distribution,
        axis_gap_summary,
        trajectories,
        anchors,
        anchor_frequency,
    ):
        if FORBIDDEN_PUBLIC_FIELDS.intersection(frame.columns):
            raise ValueError("A forbidden structure or identity field entered figure data.")

    distribution_summary: dict[str, dict[str, float]] = {}
    for axis in AXIS_ORDER:
        values = axis_distribution.loc[
            axis_distribution["axis"] == axis, "axis_score"
        ].to_numpy(dtype=float)
        chosen = float(
            axis_distribution.loc[
                (axis_distribution["axis"] == axis)
                & _bool_series(axis_distribution["is_selected_candidate"]),
                "axis_score",
            ].iloc[0]
        )
        distribution_summary[axis] = {
            "minimum": float(values.min()),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
            "selected_score": chosen,
        }

    stage1_final_spearman = float(
        migration["stage1_rank"].corr(migration["final_rank"], method="spearman")
    )
    if not math.isfinite(stage1_final_spearman):
        raise ValueError("Stage 1-to-final Spearman correlation is non-finite.")
    summary = {
        "schema_version": 1,
        "run_id": "20260724-organic-additive-v4-01",
        "candidate_count_stage1": 800,
        "candidate_count_semantic": 200,
        "aggregation_trajectory_count": 200,
        "axis_score_record_count": 600,
        "selected_anchor_record_count": 30,
        "anchor_frequency_record_count": 30,
        "axis_order": list(AXIS_ORDER),
        "selected_candidate": {
            "candidate_id": str(selected_row["candidate_id"]),
            "display_name": str(selected_row["display_name"]),
            "connectivity_smiles": str(selected_row["connectivity_smiles"]),
            "stage1_rank": 141,
            "stage1_ranking_percentile": (200.0 - 141.0) / 199.0 * 100.0,
            "semantic_rank": 1,
            "final_rank": 1,
            "final_semantic_ranking_percentile": 100.0,
            "axis_ranks": {
                axis: int(selected_row[f"{axis}_rank"]) for axis in AXIS_ORDER
            },
            "axis_scores": {
                axis: float(selected_row[f"{axis}_score"]) for axis in AXIS_ORDER
            },
            "molecular_formula": selected_molecule["molecular_formula"],
        },
        "rank1_rank2_gaps": {
            str(row["axis"]): float(row["rank1_rank2_gap"])
            for row in gap_rows
        },
        "stage1_final_spearman": stage1_final_spearman,
        "axis_score_distributions": distribution_summary,
        "ranking_rule": "semantic_score descending; exact ties by candidate_id ascending",
        "structural_similarity_used": False,
        "identity_used_for_ranking": False,
    }
    return (
        {
            "rank_migration.csv": migration,
            "axis_score_distribution.csv": axis_distribution,
            "axis_rank1_rank2_summary.csv": axis_gap_summary,
            "semantic_aggregation_trajectories.csv": trajectories,
            "selected_anchor_profile.csv": anchors,
            "anchor_selection_frequency.csv": anchor_frequency,
        },
        {**summary, "_selected_molecule": selected_molecule},
    )


def build_bundle(source_dir: Path, output_dir: Path) -> None:
    tables, summary = derive_tables(source_dir)
    selected_molecule = summary.pop("_selected_molecule")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        write_csv(output_dir / name, frame)
    write_json(output_dir / "selected_molecule.json", selected_molecule)
    write_json(output_dir / "summary.json", summary)

    source_manifest = source_dir / "data_manifest.json"
    manifest = {
        "schema_version": 2,
        "run_id": summary["run_id"],
        "figure_release": "manuscript",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_manifest_sha256": sha256_file(source_manifest),
        "inputs": {
            name: {
                "path": relative,
                "sha256": sha256_file(source_dir / relative),
                "size": (source_dir / relative).stat().st_size,
            }
            for name, relative in INPUT_FILES.items()
        },
        "processing_script": {
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "size": (output_dir / name).stat().st_size,
            }
            for name in OUTPUT_FILES
        },
    }
    write_json(output_dir / "data_manifest.json", manifest)


def verify_bundle(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    verify_source_manifest(source_dir)
    manifest = load_json(output_dir / "data_manifest.json")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("figure_release") != "manuscript"
        or manifest.get("run_id") != "20260724-organic-additive-v4-01"
    ):
        raise ValueError("Invalid manuscript figure data-manifest boundary.")
    if manifest.get("source_data_manifest_sha256") != sha256_file(
        source_dir / "data_manifest.json"
    ):
        raise ValueError("Figure bundle is bound to a different source manifest.")
    if manifest.get("processing_script", {}).get("sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("Figure bundle was produced by a different processing script.")
    for name, relative in INPUT_FILES.items():
        record = manifest.get("inputs", {}).get(name)
        path = source_dir / relative
        if (
            not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
            or int(record.get("size", -1)) != path.stat().st_size
        ):
            raise ValueError(f"Figure input binding mismatch: {name}.")
    for name in OUTPUT_FILES:
        record = manifest.get("outputs", {}).get(name)
        path = output_dir / name
        if (
            not path.is_file()
            or path.is_symlink()
            or not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
            or int(record.get("size", -1)) != path.stat().st_size
        ):
            raise ValueError(f"Figure output mismatch: {name}.")

    expected_rows = {
        "rank_migration.csv": 200,
        "axis_score_distribution.csv": 600,
        "axis_rank1_rank2_summary.csv": 3,
        "semantic_aggregation_trajectories.csv": 200,
        "selected_anchor_profile.csv": 30,
        "anchor_selection_frequency.csv": 30,
    }
    for name, count in expected_rows.items():
        frame = pd.read_csv(output_dir / name)
        if len(frame) != count:
            raise ValueError(f"Unexpected row count for {name}.")
        if FORBIDDEN_PUBLIC_FIELDS.intersection(frame.columns):
            raise ValueError(f"Forbidden field found in {name}.")
    summary = load_json(output_dir / "summary.json")
    selected_molecule = load_json(output_dir / "selected_molecule.json")
    if (
        summary.get("selected_candidate", {}).get("candidate_id") != "C0686"
        or summary.get("selected_candidate", {}).get("final_rank") != 1
        or summary.get("structural_similarity_used") is not False
        or selected_molecule.get("candidate_id") != "C0686"
        or selected_molecule.get("identity_used_for_ranking") is not False
    ):
        raise ValueError("Figure summary acceptance boundary failed.")
    return manifest


def prepare(source_dir: Path, output_dir: Path, replace: bool) -> str:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and not replace:
        verify_bundle(source_dir, output_dir)
        return "unchanged"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    backup: Path | None = None
    try:
        build_bundle(source_dir, staging)
        verify_bundle(source_dir, staging)
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                raise ValueError("Refusing to overwrite an existing figure backup.")
            output_dir.replace(backup)
        staging.replace(output_dir)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise
    verify_bundle(source_dir, output_dir)
    return "created"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--source-dir",
            type=Path,
            default=DEFAULT_SOURCE_DIR,
        )
        subparser.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
        )
        if command == "prepare":
            subparser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        status = prepare(args.source_dir, args.output_dir, args.replace)
    else:
        verify_bundle(args.source_dir.resolve(), args.output_dir.resolve())
        status = "verified"
    print(
        json.dumps(
            {
                "status": status,
                "figure_release": "manuscript",
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
