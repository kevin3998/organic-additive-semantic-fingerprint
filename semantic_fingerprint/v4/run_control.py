from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Mapping, Sequence

import numpy as np

from semantic_fingerprint.v4.backend import ChemDFMContextEmbeddingBackend
from semantic_fingerprint.v4.molecular_card import build_molecular_card_rows
from semantic_fingerprint.v4.pipeline import (
    aggregate_semantic_fingerprints,
    build_v4_diagnostics,
)


_METHOD_VERSION = "organic-semantic-fingerprint-v4"
_AXES = ("ncm811", "lithium_metal", "lhce_solvation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, object]:
    _require_regular_file(path, f"JSON input {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON source: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".publish",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite immutable JSON: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> bytes:
    if not rows and fieldnames is None:
        raise ValueError("Cannot infer CSV fields from an empty row set.")
    fields = list(fieldnames or rows[0].keys())
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_csv_bytes(rows))
    temporary.replace(path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require_regular_file(path, f"CSV input {path.name}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"Invalid CSV source: {path}") from error


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    _require_regular_file(path, f"hash input {path.name}", reject_hardlink=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_regular_file(
    path: Path,
    label: str,
    *,
    reject_hardlink: bool = True,
) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"Required source is missing for {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"Refusing symbolic link for {label}: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Expected a regular file for {label}: {path}")
    if reject_hardlink and metadata.st_nlink != 1:
        raise ValueError(f"Refusing hard-linked source for {label}: {path}")
    return metadata


def _assert_no_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    for part in path.absolute().parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise ValueError(f"Refusing symbolic link in {label}: {current}")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_value = left.resolve(strict=False)
    right_value = right.resolve(strict=False)
    return (
        left_value == right_value
        or left_value in right_value.parents
        or right_value in left_value.parents
    )


def _assert_disjoint(left: Path, right: Path, label: str) -> None:
    if _paths_overlap(left.absolute(), right.absolute()):
        raise ValueError(f"{label} must be disjoint.")


def _input_record(path: Path) -> dict[str, object]:
    metadata = _require_regular_file(path, path.name, reject_hardlink=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size": metadata.st_size,
    }


def _verify_record(
    record: Mapping[str, object],
    *,
    expected_path: Path,
    label: str,
) -> None:
    try:
        recorded_path = Path(str(record["path"]))
        recorded_hash = str(record["sha256"])
        recorded_size = int(record["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed frozen record for {label}.") from error
    _require_regular_file(expected_path, label)
    if recorded_path.resolve() != expected_path.resolve():
        raise ValueError(f"Frozen source path mismatch for {label}.")
    if expected_path.stat().st_size != recorded_size:
        raise ValueError(f"Frozen source size mismatch for {label}.")
    actual_hash = sha256_file(expected_path)
    if actual_hash != recorded_hash:
        raise ValueError(f"Frozen source hash mismatch for {label}.")


def _verify_fixed_hash(path: Path, expected_hash: object, label: str) -> None:
    actual = sha256_file(path)
    if actual != str(expected_hash):
        raise ValueError(f"Fixed source hash mismatch for {label}: {actual}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _load_v4_config(config_path: Path) -> dict[str, object]:
    config = _load_json(config_path)
    if (
        config.get("schema_version") != 4
        or config.get("method_version") != _METHOD_VERSION
    ):
        raise ValueError("Expected the fixed v4 configuration.")
    return config


def _expected_counts(config: Mapping[str, object]) -> dict[str, int]:
    source = _mapping(config.get("source_bundle"), "source_bundle")
    counts = _mapping(
        source.get("expected_counts"), "source_bundle.expected_counts"
    )
    names = (
        "candidate_count",
        "anchor_count",
        "stage1_embedding_rows",
        "selected_count",
    )
    result: dict[str, int] = {}
    for name in names:
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid fixed count: {name}.")
        result[name] = value
    return result


def _csv_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Invalid selected Boolean: {value}")
    return normalized == "true"


def _verify_source_bundle(
    *,
    config: Mapping[str, object],
    source_public_dir: Path,
    source_private_dir: Path,
) -> dict[str, object]:
    _assert_disjoint(
        source_public_dir,
        source_private_dir,
        "source bundle public and private roots",
    )
    _assert_no_symlink_components(source_public_dir, "public source bundle")
    _assert_no_symlink_components(source_private_dir, "private source bundle")
    source_config = _mapping(config.get("source_bundle"), "source_bundle")
    hashes = _mapping(source_config.get("hashes"), "source_bundle.hashes")
    paths = {
        "candidates": source_public_dir / "blind/candidates.csv",
        "reference_anchors": source_public_dir / "blind/reference_anchors.csv",
        "stage1_scores": source_public_dir / "stage1_scores.csv",
        "stage1_embeddings": source_public_dir / "stage1_embeddings.npy",
        "molecular_cards": source_public_dir / "blind/molecular_cards.csv",
        "target_registry": source_private_dir / "target_registry.json",
    }
    if set(hashes) != set(paths):
        raise ValueError("Fixed source-bundle hash set is incomplete.")
    for label, path in paths.items():
        _verify_fixed_hash(path, hashes[label], f"source bundle {label}")

    counts = _expected_counts(config)
    candidates = _read_csv(paths["candidates"])
    anchors = _read_csv(paths["reference_anchors"])
    stage1 = _read_csv(paths["stage1_scores"])
    if len(candidates) != counts["candidate_count"]:
        raise ValueError("Candidate count differs from the fixed source bundle.")
    candidate_ids = [row.get("candidate_id", "") for row in candidates]
    candidate_smiles = [row.get("connectivity_smiles", "") for row in candidates]
    if (
        any(not value for value in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(not value for value in candidate_smiles)
        or len(set(candidate_smiles)) != len(candidate_smiles)
    ):
        raise ValueError("Candidates are missing or duplicate IDs/structures.")
    if len(anchors) != counts["anchor_count"]:
        raise ValueError("Anchor count differs from the fixed source bundle.")
    anchor_ids = [row.get("anchor_id", "") for row in anchors]
    if anchor_ids != [f"A{index:02d}" for index in range(1, len(anchors) + 1)]:
        raise ValueError("Anchors are not ordered by fixed anonymous ID.")
    if len(stage1) != counts["candidate_count"] or {
        row.get("candidate_id") for row in stage1
    } != set(candidate_ids):
        raise ValueError("Stage 1 rows do not match candidates.")
    try:
        selected = sorted(
            (row for row in stage1 if _csv_bool(row.get("selected"))),
            key=lambda row: int(row["stage1_rank"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Stage 1 ranks are invalid.") from error
    if len(selected) != counts["selected_count"]:
        raise ValueError("Selected count differs from the fixed source bundle.")
    selected_ids = [str(row["candidate_id"]) for row in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Selected candidate IDs are duplicated.")
    try:
        embeddings = np.load(paths["stage1_embeddings"], mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ValueError("Stage 1 embeddings are invalid.") from error
    if embeddings.ndim != 2 or embeddings.shape[0] != counts["stage1_embedding_rows"]:
        raise ValueError("Stage 1 embedding shape differs from the fixed source bundle.")

    regenerated_cards = [
        {key: str(value) for key, value in row.items()}
        for row in build_molecular_card_rows(candidates)
    ]
    if _csv_bytes(regenerated_cards) != paths["molecular_cards"].read_bytes():
        raise ValueError("Molecular cards do not regenerate byte-for-byte.")
    anchor_inputs = [
        {
            "candidate_id": str(row["anchor_id"]),
            "connectivity_smiles": str(row["connectivity_smiles"]),
        }
        for row in anchors
    ]
    generated_anchor_cards = []
    by_anchor = {
        str(row["candidate_id"]): row
        for row in build_molecular_card_rows(anchor_inputs)
    }
    for anchor in anchors:
        anchor_id = str(anchor["anchor_id"])
        card = by_anchor[anchor_id]
        generated_anchor_cards.append(
            {
                "anchor_id": anchor_id,
                "name": str(anchor.get("name", "")),
                **{
                    key: value
                    for key, value in card.items()
                    if key != "candidate_id"
                },
            }
        )
    return {
        "paths": paths,
        "candidates": candidates,
        "anchors": anchors,
        "selected_ids": selected_ids,
        "anchor_cards": generated_anchor_cards,
        "counts": counts,
    }


def _copy_verified(source: Path, destination: Path, label: str) -> None:
    _require_regular_file(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _require_regular_file(destination, f"copied {label}", reject_hardlink=False)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise RuntimeError(f"Refusing hard-linked source copy for {label}.")
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"Copied {label} hash differs from its source.")


def _materialize_import(
    *,
    config_path: Path,
    verified: Mapping[str, object],
    public_temp: Path,
    private_temp: Path,
) -> dict[str, object]:
    paths = _mapping(verified.get("paths"), "verified source-bundle paths")
    public_sources = {
        "candidates": (paths["candidates"], public_temp / "blind/candidates.csv"),
        "reference_anchors": (
            paths["reference_anchors"],
            public_temp / "blind/reference_anchors.csv",
        ),
        "stage1_scores": (paths["stage1_scores"], public_temp / "stage1_scores.csv"),
        "stage1_embeddings": (
            paths["stage1_embeddings"],
            public_temp / "stage1_embeddings.npy",
        ),
        "molecular_cards": (
            paths["molecular_cards"],
            public_temp / "blind/molecular_cards.csv",
        ),
        "screening_config": (
            config_path,
            public_temp / "blind/screening_config.json",
        ),
    }
    public_hashes: dict[str, str] = {}
    for label, pair in public_sources.items():
        source, destination = pair
        _copy_verified(Path(source), destination, label)
        public_hashes[label] = sha256_file(destination)
    _write_csv(
        public_temp / "blind/anchor_molecular_cards.csv",
        list(verified["anchor_cards"]),
    )
    public_hashes["anchor_molecular_cards"] = sha256_file(
        public_temp / "blind/anchor_molecular_cards.csv"
    )

    private_sources = {
        "target_registry": (
            paths["target_registry"],
            private_temp / "target_registry.json",
        ),
    }
    private_hashes: dict[str, str] = {}
    for label, pair in private_sources.items():
        source, destination = pair
        _copy_verified(Path(source), destination, label)
        private_hashes[label] = sha256_file(destination)

    source_record = {
        "schema_version": 4,
        "method_version": _METHOD_VERSION,
        "copied_public_hashes": public_hashes,
    }
    source_path = public_temp / "blind/source_bundle.json"
    _write_json(source_path, source_record)
    source_hash = sha256_file(source_path)
    private_source = {
        **source_record,
        "source_bundle_sha256": source_hash,
        "copied_private_hashes": private_hashes,
    }
    private_source_path = private_temp / "source_bundle_private.json"
    _write_json(private_source_path, private_source)
    private_source_hash = sha256_file(private_source_path)
    counts = _mapping(verified.get("counts"), "verified counts")
    import_metadata = {
        "schema_version": 4,
        "method_version": _METHOD_VERSION,
        "source_bundle_sha256": source_hash,
        "source_bundle_private_sha256": private_source_hash,
        **{key: int(value) for key, value in counts.items()},
    }
    _write_json(private_temp / "import_metadata.json", import_metadata)
    return {
        "candidate_count": int(counts["candidate_count"]),
        "anchor_count": int(counts["anchor_count"]),
        "selected_count": int(counts["selected_count"]),
        "source_bundle_sha256": source_hash,
        "source_bundle_private_sha256": private_source_hash,
    }


def import_source_bundle(
    *,
    config_path: Path,
    source_public_dir: Path,
    source_private_dir: Path,
    run_dir: Path,
    private_dir: Path,
) -> dict[str, object]:
    config_path = Path(config_path)
    source_public_dir = Path(source_public_dir)
    source_private_dir = Path(source_private_dir)
    run_dir = Path(run_dir)
    private_dir = Path(private_dir)
    _assert_disjoint(run_dir, private_dir, "v4 public and private destinations")
    _assert_no_symlink_components(run_dir.parent, "v4 public destination")
    _assert_no_symlink_components(private_dir.parent, "v4 private destination")
    if run_dir.exists() or private_dir.exists():
        raise FileExistsError("v4 import destination already exists.")
    config = _load_v4_config(config_path)
    verified = _verify_source_bundle(
        config=config,
        source_public_dir=source_public_dir,
        source_private_dir=source_private_dir,
    )
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    private_dir.parent.mkdir(parents=True, exist_ok=True)
    public_temp = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}.import-", dir=run_dir.parent)
    )
    private_temp = Path(
        tempfile.mkdtemp(prefix=f".{private_dir.name}.import-", dir=private_dir.parent)
    )
    public_published = False
    try:
        result = _materialize_import(
            config_path=config_path,
            verified=verified,
            public_temp=public_temp,
            private_temp=private_temp,
        )
        if run_dir.exists() or private_dir.exists():
            raise FileExistsError("v4 import destination appeared during import.")
        public_temp.replace(run_dir)
        public_published = True
        private_temp.replace(private_dir)
        return result
    except BaseException:
        if public_published and run_dir.exists():
            shutil.rmtree(run_dir)
        raise
    finally:
        if public_temp.exists():
            shutil.rmtree(public_temp)
        if private_temp.exists():
            shutil.rmtree(private_temp)


def _validate_import_provenance(
    *,
    config_path: Path,
    run_dir: Path,
    private_dir: Path,
) -> dict[str, object]:
    _assert_disjoint(run_dir, private_dir, "v4 public and private run roots")
    source_path = run_dir / "blind/source_bundle.json"
    private_source_path = private_dir / "source_bundle_private.json"
    source = _load_json(source_path)
    private_source = _load_json(private_source_path)
    if source.get("schema_version") != 4 or source.get("method_version") != _METHOD_VERSION:
        raise ValueError("Invalid imported public source-bundle record.")
    if private_source.get("source_bundle_sha256") != sha256_file(source_path):
        raise ValueError("Imported private source does not bind the public source.")
    public_hashes = _mapping(source.get("copied_public_hashes"), "copied public hashes")
    public_paths = {
        "candidates": run_dir / "blind/candidates.csv",
        "reference_anchors": run_dir / "blind/reference_anchors.csv",
        "stage1_scores": run_dir / "stage1_scores.csv",
        "stage1_embeddings": run_dir / "stage1_embeddings.npy",
        "molecular_cards": run_dir / "blind/molecular_cards.csv",
        "anchor_molecular_cards": run_dir / "blind/anchor_molecular_cards.csv",
        "screening_config": run_dir / "blind/screening_config.json",
    }
    if set(public_hashes) != set(public_paths):
        raise ValueError("Imported public source hash set is incomplete.")
    for label, path in public_paths.items():
        _verify_fixed_hash(path, public_hashes[label], f"imported {label}")
    if sha256_file(config_path) != public_hashes["screening_config"]:
        raise ValueError("Freeze configuration hash differs from imported config.")
    private_hashes = _mapping(
        private_source.get("copied_private_hashes"), "copied private hashes"
    )
    private_paths = {
        "target_registry": private_dir / "target_registry.json",
    }
    if set(private_hashes) != set(private_paths):
        raise ValueError("Imported private source hash set is incomplete.")
    for label, path in private_paths.items():
        _verify_fixed_hash(path, private_hashes[label], f"imported private {label}")
    return {
        "source_bundle_sha256": sha256_file(source_path),
        "source_bundle_private_sha256": sha256_file(private_source_path),
        "public_paths": public_paths,
        "private_hashes": dict(private_hashes),
    }


def _git_state(repo_root: Path, require_clean: bool) -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("Unable to inspect the v4 Git worktree.") from error
    if require_clean and status:
        raise ValueError("Freeze requires a worktree with no tracked changes.")
    return commit, status


def _validate_model(config: Mapping[str, object], model_dir: Path) -> tuple[int, dict[str, dict[str, object]]]:
    model = _mapping(config.get("model"), "model configuration")
    configured_dir = Path(str(model.get("local_dir", "")))
    if model_dir.resolve() != configured_dir.resolve():
        raise ValueError("Model directory differs from the fixed v4 config.")
    records: dict[str, dict[str, object]] = {}
    metadata = _mapping(model.get("metadata_sha256"), "model metadata hashes")
    weights = _mapping(model.get("weight_sha256"), "model weight hashes")
    for name, expected in {**metadata, **weights}.items():
        path = model_dir / str(name)
        _verify_fixed_hash(path, expected, f"model/{name}")
        records[f"model/{name}"] = _input_record(path)
    index = _load_json(model_dir / "model.safetensors.index.json")
    index_metadata = _mapping(index.get("metadata"), "model index metadata")
    total = index_metadata.get("total_size")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total != int(model.get("weight_total_bytes", -1))
    ):
        raise ValueError("Model tensor byte count differs from the fixed config.")
    model_config = _load_json(model_dir / "config.json")
    hidden_size = model_config.get("hidden_size")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("Model hidden_size is invalid.")
    return hidden_size, records


def _selected_entity_order(run_dir: Path) -> tuple[list[str], list[str]]:
    anchors = _read_csv(run_dir / "blind/reference_anchors.csv")
    stage1 = _read_csv(run_dir / "stage1_scores.csv")
    selected = sorted(
        (row for row in stage1 if _csv_bool(row.get("selected"))),
        key=lambda row: int(row["stage1_rank"]),
    )
    anchor_ids = [str(row["anchor_id"]) for row in anchors]
    selected_ids = [str(row["candidate_id"]) for row in selected]
    return ["axis_baseline", *anchor_ids, *selected_ids], selected_ids


def _validate_smoke_report(
    path: Path,
    *,
    batch_size: int,
    hidden_size: int,
    selected_ids: Sequence[str],
    epsilon: float,
) -> dict[str, object]:
    report = _load_json(path)
    if (
        report.get("schema_version") != 4
        or report.get("method_version") != _METHOD_VERSION
    ):
        raise ValueError("Invalid v4 smoke report schema.")
    if report.get("batch_size") != batch_size:
        raise ValueError("Smoke batch size differs from freeze batch size.")
    if report.get("hidden_size") != hidden_size:
        raise ValueError("Smoke hidden size differs from model hidden size.")
    candidate_ids = report.get("candidate_ids")
    expected_count = min(8, len(selected_ids))
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != expected_count
        or len(set(candidate_ids)) != expected_count
        or not set(candidate_ids).issubset(selected_ids)
    ):
        raise ValueError("Smoke candidate IDs are invalid.")
    if report.get("vector_shape") != [3, expected_count, hidden_size]:
        raise ValueError("Smoke vector shape is invalid.")
    if report.get("vector_count") != len(_AXES) * expected_count:
        raise ValueError("Smoke vector count is invalid.")
    if report.get("all_finite") is not True:
        raise ValueError("Smoke vectors are not all finite.")
    minimum_norm = report.get("minimum_centered_norm")
    if (
        isinstance(minimum_norm, bool)
        or not isinstance(minimum_norm, (int, float))
        or not math.isfinite(float(minimum_norm))
        or float(minimum_norm) <= epsilon
    ):
        raise ValueError("Smoke centered norm does not exceed epsilon.")
    token_lengths = report.get("token_lengths")
    if not isinstance(token_lengths, Mapping) or set(token_lengths) != set(candidate_ids):
        raise ValueError("Smoke token lengths are incomplete.")
    for candidate_id in candidate_ids:
        row = token_lengths[candidate_id]
        if not isinstance(row, Mapping) or set(row) != set(_AXES):
            raise ValueError("Smoke token lengths do not cover all axes.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in row.values()
        ):
            raise ValueError("Smoke token length must be a positive integer.")
    return report


def freeze_v4_run(
    *,
    config_path: Path,
    environment_path: Path,
    run_dir: Path,
    private_dir: Path,
    model_dir: Path,
    batch_size: int,
    smoke_report_path: Path,
    repo_root: Path,
    require_clean: bool = True,
) -> dict[str, object]:
    if isinstance(batch_size, bool) or batch_size not in (4, 8):
        raise ValueError("v4 batch_size must be 4 or 8.")
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    config_path = Path(config_path)
    environment_path = Path(environment_path)
    run_dir = Path(run_dir)
    private_dir = Path(private_dir)
    model_dir = Path(model_dir)
    smoke_report_path = Path(smoke_report_path)
    config = _load_v4_config(config_path)
    provenance = _validate_import_provenance(
        config_path=config_path,
        run_dir=run_dir,
        private_dir=private_dir,
    )
    hidden_size, model_records = _validate_model(config, model_dir)
    commit, status = _git_state(Path(repo_root), require_clean)
    entity_order, selected_ids = _selected_entity_order(run_dir)
    embedding = _mapping(config.get("embedding"), "embedding configuration")
    epsilon = float(embedding.get("epsilon", 1e-12))
    report = _validate_smoke_report(
        smoke_report_path,
        batch_size=batch_size,
        hidden_size=hidden_size,
        selected_ids=selected_ids,
        epsilon=epsilon,
    )
    prompts = _mapping(config.get("prompts"), "prompt configuration")
    criteria = _mapping(prompts.get("criteria"), "prompt criteria")
    screening = _mapping(config.get("screening"), "screening configuration")
    if list(screening.get("axis_order", [])) != list(_AXES) or set(criteria) != set(
        _AXES
    ):
        raise ValueError("Configured v4 axis order or criteria have drifted.")
    prompt_hashes = {
        "system": _sha256_text(str(prompts.get("system", ""))),
        "template": _sha256_text(str(prompts.get("user_template", ""))),
        **{axis: _sha256_text(str(criteria[axis])) for axis in _AXES},
    }
    public_paths = _mapping(provenance["public_paths"], "imported public paths")
    inputs = {
        label: _input_record(Path(path)) for label, path in public_paths.items()
    }
    inputs["source_bundle"] = _input_record(
        run_dir / "blind/source_bundle.json"
    )
    inputs["environment"] = _input_record(environment_path)
    inputs["smoke_report"] = _input_record(smoke_report_path)
    inputs.update(model_records)
    model_config = _mapping(config.get("model"), "model configuration")
    manifest = {
        "schema_version": 4,
        "method_version": _METHOD_VERSION,
        "created_at": _utc_now(),
        "git_commit": commit,
        "git_status_at_freeze": status,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "device": model_config.get("device"),
        "dtype": model_config.get("dtype"),
        "model_revision": model_config.get("revision"),
        "axis_order": list(_AXES),
        "entity_order": entity_order,
        "prompt_sha256": prompt_hashes,
        "semantic_success_rank": screening.get("semantic_success_rank"),
        "smoke_candidate_ids": list(report["candidate_ids"]),
        "source_bundle_sha256": provenance["source_bundle_sha256"],
        "source_bundle_private_sha256": provenance[
            "source_bundle_private_sha256"
        ],
        "private_input_hashes": provenance["private_hashes"],
        "inputs": inputs,
    }
    _write_json_exclusive(manifest_path, manifest)
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "entity_count": len(entity_order),
    }


def verify_v4_manifest(
    *,
    run_dir: Path,
    model_dir: Path,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    model_dir = Path(model_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != 4
        or manifest.get("method_version") != _METHOD_VERSION
    ):
        raise ValueError("Expected a frozen v4 manifest.")
    if manifest.get("axis_order") != list(_AXES):
        raise ValueError("Frozen v4 axis order has drifted.")
    if manifest.get("batch_size") not in (4, 8):
        raise ValueError("Frozen v4 batch size is invalid.")
    hidden_size = manifest.get("hidden_size")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("Frozen v4 hidden size is invalid.")
    entity_order = manifest.get("entity_order")
    if (
        not isinstance(entity_order, list)
        or len(entity_order) != 211
        or entity_order[0] != "axis_baseline"
        or len(set(entity_order)) != len(entity_order)
    ):
        raise ValueError("Frozen v4 entity order must contain 211 unique entries.")
    prompt_hashes = manifest.get("prompt_sha256")
    if not isinstance(prompt_hashes, Mapping) or set(prompt_hashes) != {
        "system",
        "template",
        *_AXES,
    }:
        raise ValueError("Frozen v4 prompt hashes are incomplete.")

    inputs = _mapping(manifest.get("inputs"), "v4 manifest inputs")
    for label, raw_record in inputs.items():
        record = _mapping(raw_record, f"v4 manifest input {label}")
        path = Path(str(record.get("path", "")))
        _verify_record(record, expected_path=path, label=f"v4/{label}")
        if str(label).startswith("model/") and path.parent.resolve() != model_dir.resolve():
            raise ValueError("Frozen model input path differs from screen model dir.")

    config_path = run_dir / "blind/screening_config.json"
    config = _load_v4_config(config_path)
    if inputs.get("screening_config", {}).get("sha256") != sha256_file(config_path):
        raise ValueError("Frozen v4 config record mismatch.")
    model = _mapping(config.get("model"), "model configuration")
    if Path(str(model.get("local_dir", ""))).resolve() != model_dir.resolve():
        raise ValueError("Screen model directory differs from the frozen config.")
    if manifest.get("model_revision") != model.get("revision"):
        raise ValueError("Frozen model revision has drifted.")
    if manifest.get("device") != model.get("device") or manifest.get("dtype") != model.get(
        "dtype"
    ):
        raise ValueError("Frozen device or dtype has drifted.")

    current_order, selected_ids = _selected_entity_order(run_dir)
    if current_order != entity_order:
        raise ValueError("Current entity order differs from the frozen manifest.")
    if len(selected_ids) != 200:
        raise ValueError("v4 screen requires exactly 200 selected candidates.")
    anchors = _read_csv(run_dir / "blind/reference_anchors.csv")
    if len(anchors) != 10:
        raise ValueError("v4 screen requires exactly 10 anchors.")

    prompts = _mapping(config.get("prompts"), "prompt configuration")
    criteria = _mapping(prompts.get("criteria"), "prompt criteria")
    current_prompt_hashes = {
        "system": _sha256_text(str(prompts.get("system", ""))),
        "template": _sha256_text(str(prompts.get("user_template", ""))),
        **{axis: _sha256_text(str(criteria.get(axis, ""))) for axis in _AXES},
    }
    if current_prompt_hashes != dict(prompt_hashes):
        raise ValueError("Current prompts differ from the frozen manifest.")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "config": config,
        "entity_order": entity_order,
        "selected_ids": selected_ids,
        "hidden_size": hidden_size,
        "batch_size": int(manifest["batch_size"]),
    }


@contextmanager
def _exclusive_screen_claim(run_dir: Path):
    path = run_dir / "checkpoints/screen.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("Safe screen claims require O_NOFOLLOW.")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | no_follow, 0o600)
    except OSError as error:
        raise ValueError(f"Unsafe screen claim path: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        path_metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise ValueError("Unsafe screen claim object.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError("v4 screen is already in progress.") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _expected_checkpoint_metadata(
    verified: Mapping[str, object],
) -> dict[str, object]:
    manifest = _mapping(verified.get("manifest"), "verified manifest")
    hidden_size = int(verified["hidden_size"])
    entity_order = list(verified["entity_order"])
    return {
        "schema_version": 4,
        "method_version": _METHOD_VERSION,
        "manifest_sha256": str(verified["manifest_sha256"]),
        "axis_order": list(_AXES),
        "entity_order": entity_order,
        "prompt_sha256": dict(
            _mapping(manifest.get("prompt_sha256"), "manifest prompt hashes")
        ),
        "model_revision": manifest.get("model_revision"),
        "shape": [len(_AXES), len(entity_order), hidden_size],
        "dtype": "float32",
        "batch_size": int(verified["batch_size"]),
    }


def _open_or_create_embedding_checkpoint(
    *,
    run_dir: Path,
    metadata: Mapping[str, object],
) -> tuple[np.memmap, np.ndarray]:
    embedding_path = run_dir / "semantic_embeddings.npy"
    completion_path = run_dir / "embedding_completion.npy"
    metadata_path = run_dir / "checkpoints/embedding.metadata.json"
    existence = [path.exists() for path in (embedding_path, completion_path, metadata_path)]
    shape = tuple(int(value) for value in metadata["shape"])
    if any(existence) and not all(existence):
        raise ValueError("Torn embedding checkpoint file set.")
    if not any(existence):
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = embedding_path.with_name(
            f".{embedding_path.name}.{os.getpid()}.create"
        )
        mmap = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        mmap[:] = 0.0
        mmap.flush()
        del mmap
        _fsync_file(temporary)
        temporary.replace(embedding_path)
        _atomic_save_npy(completion_path, np.zeros(shape[:2], dtype=np.bool_))
        _write_json(metadata_path, dict(metadata))
    else:
        actual_metadata = _load_json(metadata_path)
        if actual_metadata != dict(metadata):
            raise ValueError("Embedding checkpoint metadata differs from manifest.")

    _require_regular_file(embedding_path, "semantic embedding checkpoint")
    _require_regular_file(completion_path, "embedding completion checkpoint")
    try:
        embeddings = np.load(embedding_path, mmap_mode="r+")
        completion = np.load(completion_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("Invalid embedding checkpoint data.") from error
    if embeddings.shape != shape or embeddings.dtype != np.float32:
        raise ValueError("Embedding checkpoint shape or dtype is invalid.")
    if completion.shape != shape[:2] or completion.dtype != np.bool_:
        raise ValueError("Completion bitmap shape or dtype is invalid.")
    if not np.isfinite(embeddings[completion]).all():
        raise ValueError("Completed embeddings contain non-finite values.")
    return embeddings, completion


def _validate_batch_bitmap(completion: np.ndarray, batch_size: int) -> None:
    for axis_index in range(completion.shape[0]):
        for start in range(0, completion.shape[1], batch_size):
            batch = completion[axis_index, start : start + batch_size]
            if bool(batch.any()) != bool(batch.all()):
                raise ValueError("Torn completion bitmap inside a fixed batch.")


def _entity_cards(
    *,
    run_dir: Path,
    entity_order: Sequence[str],
    config: Mapping[str, object],
) -> list[str]:
    prompts = _mapping(config.get("prompts"), "prompt configuration")
    candidate_cards = {
        str(row["candidate_id"]): str(row["molecular_card"])
        for row in _read_csv(run_dir / "blind/molecular_cards.csv")
    }
    anchor_cards = {
        str(row["anchor_id"]): str(row["molecular_card"])
        for row in _read_csv(run_dir / "blind/anchor_molecular_cards.csv")
    }
    cards = [str(prompts.get("baseline_card", ""))]
    for entity_id in entity_order[1:]:
        source = anchor_cards if entity_id.startswith("A") else candidate_cards
        if entity_id not in source:
            raise ValueError(f"Missing molecular card for {entity_id}.")
        cards.append(source[entity_id])
    if len(cards) != len(entity_order):
        raise RuntimeError("Molecular-card entity order is incomplete.")
    return cards


def _build_axis_prompts(
    *,
    config: Mapping[str, object],
    cards: Sequence[str],
) -> tuple[str, dict[str, list[str]]]:
    prompts = _mapping(config.get("prompts"), "prompt configuration")
    system = str(prompts.get("system", ""))
    template = str(prompts.get("user_template", ""))
    criteria = _mapping(prompts.get("criteria"), "prompt criteria")
    return system, {
        axis: [
            template.format(criterion=str(criteria[axis]), molecular_card=card)
            for card in cards
        ]
        for axis in _AXES
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _publish_result_bytes(path: Path, content: bytes) -> dict[str, object]:
    expected_hash = hashlib.sha256(content).hexdigest()
    _atomic_write_bytes(path, content)
    if sha256_file(path) != expected_hash or path.stat().st_size != len(content):
        raise RuntimeError(f"Published result differs from intended bytes: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": expected_hash,
        "size": len(content),
    }


def _finalize_screen_outputs(
    *,
    run_dir: Path,
    verified: Mapping[str, object],
    embeddings: np.ndarray,
) -> dict[str, object]:
    config = _mapping(verified.get("config"), "verified config")
    candidates = _read_csv(run_dir / "blind/candidates.csv")
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    stage1 = _read_csv(run_dir / "stage1_scores.csv")
    selected = sorted(
        (row for row in stage1 if _csv_bool(row.get("selected"))),
        key=lambda row: int(row["stage1_rank"]),
    )
    selected_candidates = [
        {
            "candidate_id": candidate_id,
            "connectivity_smiles": candidate_by_id[candidate_id][
                "connectivity_smiles"
            ],
            "stage1_rank": int(row["stage1_rank"]),
        }
        for row in selected
        for candidate_id in [str(row["candidate_id"])]
    ]
    anchors = _read_csv(run_dir / "blind/reference_anchors.csv")
    aggregate_rows = aggregate_semantic_fingerprints(
        selected_candidates,
        anchors,
        embeddings,
    )
    diagnostics = build_v4_diagnostics(
        semantic_rows=aggregate_rows,
        raw_embeddings=embeddings,
    )

    cosine_fields = [
        f"{axis}_anchor_A{anchor_index:02d}_cosine"
        for axis in _AXES
        for anchor_index in range(1, 11)
    ]
    anchor_rows = [
        {
            "candidate_id": row["candidate_id"],
            "connectivity_smiles": row["connectivity_smiles"],
            **{field: row[field] for field in cosine_fields},
        }
        for row in aggregate_rows
    ]
    score_fields = [
        "candidate_id",
        "connectivity_smiles",
        *[
            field
            for axis in _AXES
            for field in (
                f"{axis}_score",
                f"{axis}_rank",
                f"{axis}_percentile",
            )
        ],
        "semantic_score",
        "semantic_rank",
        "final_rank",
    ]
    semantic_rows = [
        {field: row[field] for field in score_fields} for row in aggregate_rows
    ]
    screening = _mapping(config.get("screening"), "screening configuration")
    top_n = int(screening.get("top_n", 50))
    if top_n != 50:
        raise ValueError("v4 formal screen requires Top 50 output.")
    top_fields = [
        "candidate_id",
        "connectivity_smiles",
        *[f"{axis}_percentile" for axis in _AXES],
        "semantic_score",
        "semantic_rank",
        "final_rank",
    ]
    top_rows = [
        {field: row[field] for field in top_fields}
        for row in aggregate_rows[:top_n]
    ]
    if [int(row["final_rank"]) for row in top_rows] != list(range(1, top_n + 1)):
        raise RuntimeError("Top 50 does not match deterministic final ordering.")

    intended = {
        "anchor_cosines.csv": _csv_bytes(anchor_rows),
        "semantic_scores.csv": _csv_bytes(semantic_rows),
        "semantic_scores.jsonl": _jsonl_bytes(semantic_rows),
        "diagnostics.json": _json_bytes(diagnostics),
        "top50.csv": _csv_bytes(top_rows),
    }
    output_records = {
        name: _publish_result_bytes(run_dir / name, content)
        for name, content in intended.items()
    }
    output_records = {
        "semantic_embeddings.npy": _input_record(
            run_dir / "semantic_embeddings.npy"
        ),
        "embedding_completion.npy": _input_record(
            run_dir / "embedding_completion.npy"
        ),
        "checkpoint_metadata.json": _input_record(
            run_dir / "checkpoints/embedding.metadata.json"
        ),
        **output_records,
    }
    for name, record in output_records.items():
        _verify_record(
            record,
            expected_path=Path(str(record["path"])),
            label=f"final v4 output {name}",
        )
    return {
        "output_records": output_records,
        "candidate_count": len(semantic_rows),
        "top_n": len(top_rows),
    }


def screen_v4_run(*, run_dir: Path, model_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    model_dir = Path(model_dir)
    result_lock_path = run_dir / "result.lock.json"
    if result_lock_path.exists():
        raise FileExistsError("v4 result is already locked.")
    with _exclusive_screen_claim(run_dir):
        if result_lock_path.exists():
            raise FileExistsError("v4 result is already locked.")
        verified = verify_v4_manifest(run_dir=run_dir, model_dir=model_dir)
        metadata = _expected_checkpoint_metadata(verified)
        embeddings, completion = _open_or_create_embedding_checkpoint(
            run_dir=run_dir,
            metadata=metadata,
        )
        batch_size = int(verified["batch_size"])
        _validate_batch_bitmap(completion, batch_size)
        config = _mapping(verified.get("config"), "verified config")
        cards = _entity_cards(
            run_dir=run_dir,
            entity_order=list(verified["entity_order"]),
            config=config,
        )
        system_prompt, axis_prompts = _build_axis_prompts(
            config=config,
            cards=cards,
        )
        manifest = _mapping(verified.get("manifest"), "verified manifest")
        backend = ChemDFMContextEmbeddingBackend(
            model_dir,
            device=str(manifest["device"]),
            dtype=str(manifest["dtype"]),
        )
        if int(backend.hidden_size) != int(verified["hidden_size"]):
            raise ValueError("Backend hidden size differs from the frozen manifest.")
        completion_path = run_dir / "embedding_completion.npy"
        embedding_path = run_dir / "semantic_embeddings.npy"
        for axis_index, axis in enumerate(_AXES):
            prompts = axis_prompts[axis]
            for start in range(0, len(prompts), batch_size):
                end = min(start + batch_size, len(prompts))
                batch_bits = completion[axis_index, start:end]
                if bool(batch_bits.all()):
                    continue
                if bool(batch_bits.any()):
                    raise ValueError("Torn completion bitmap inside a fixed batch.")
                vectors = np.asarray(
                    backend.encode_prompts(
                        system_prompt=system_prompt,
                        user_prompts=prompts[start:end],
                        batch_size=batch_size,
                    ),
                    dtype=np.float32,
                )
                expected_shape = (end - start, int(verified["hidden_size"]))
                if vectors.shape != expected_shape:
                    raise ValueError("Backend returned an invalid embedding batch shape.")
                if not np.isfinite(vectors).all():
                    raise ValueError("Backend returned non-finite embeddings.")
                embeddings[axis_index, start:end] = vectors
                embeddings.flush()
                _fsync_file(embedding_path)
                completion[axis_index, start:end] = True
                _atomic_save_npy(completion_path, completion)

        if not bool(completion.all()):
            raise RuntimeError("Embedding checkpoint is incomplete after screening.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Completed semantic embeddings are non-finite.")
        finalized = _finalize_screen_outputs(
            run_dir=run_dir,
            verified=verified,
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
        outputs = _mapping(finalized.get("output_records"), "v4 output records")
        for name, record in outputs.items():
            record_mapping = _mapping(record, f"v4 output {name}")
            _verify_record(
                record_mapping,
                expected_path=Path(str(record_mapping["path"])),
                label=f"v4 output lock snapshot {name}",
            )
        lock = {
            "schema_version": 4,
            "method_version": _METHOD_VERSION,
            "created_at": _utc_now(),
            "manifest_sha256": verified["manifest_sha256"],
            "candidate_count": int(finalized["candidate_count"]),
            "top_n": int(finalized["top_n"]),
            "outputs": outputs,
        }
        _write_json_exclusive(result_lock_path, lock)
        return {
            "candidate_count": int(finalized["candidate_count"]),
            "top_n": int(finalized["top_n"]),
            "result_lock_sha256": sha256_file(result_lock_path),
        }


def _safe_file_snapshot(path: Path, *, label: str) -> dict[str, object]:
    path = Path(path)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("Safe file snapshots require O_NOFOLLOW.")
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
    except FileNotFoundError as error:
        raise ValueError(f"Required snapshot is missing: {label}.") from error
    except OSError as error:
        raise ValueError(f"Unsafe snapshot path for {label}: {path}") from error
    try:
        opened = os.fstat(descriptor)
        linked = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise ValueError(f"Unsafe snapshot object for {label}.")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        if len(content) != opened.st_size:
            raise ValueError(f"Snapshot size changed while reading {label}.")
        return {
            "path": path,
            "identity": (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
            ),
            "bytes": content,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    finally:
        os.close(descriptor)


def _parse_json_snapshot(snapshot: Mapping[str, object], *, label: str) -> dict[str, object]:
    try:
        value = json.loads(bytes(snapshot["bytes"]).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON snapshot: {label}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON snapshot must be an object: {label}.")
    return value


def _parse_csv_snapshot(snapshot: Mapping[str, object], *, label: str) -> list[dict[str, str]]:
    try:
        text = bytes(snapshot["bytes"]).decode("utf-8")
        return list(csv.DictReader(io.StringIO(text, newline="")))
    except (KeyError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"Invalid CSV snapshot: {label}.") from error


def _require_snapshot_unchanged(
    snapshot: Mapping[str, object],
    *,
    label: str,
) -> None:
    path = snapshot.get("path")
    if not isinstance(path, Path):
        raise ValueError(f"Malformed verified snapshot: {label}.")
    current = _safe_file_snapshot(path, label=label)
    if (
        current["identity"] != snapshot.get("identity")
        or current["sha256"] != snapshot.get("sha256")
        or current["bytes"] != snapshot.get("bytes")
    ):
        raise ValueError(f"verified snapshot changed: {label}.")


def _verify_snapshot_record(
    snapshot: Mapping[str, object],
    record: Mapping[str, object],
    *,
    expected_path: Path,
    label: str,
) -> None:
    try:
        recorded_path = Path(str(record["path"]))
        expected_hash = str(record["sha256"])
        expected_size = int(record["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed snapshot record for {label}.") from error
    lexical_recorded = Path(os.path.abspath(os.fspath(recorded_path)))
    lexical_expected = Path(os.path.abspath(os.fspath(expected_path)))
    if (
        lexical_recorded != lexical_expected
        or recorded_path.resolve() != expected_path.resolve()
    ):
        raise ValueError(f"Locked output path mismatch for {label}.")
    if snapshot.get("sha256") != expected_hash:
        raise ValueError(f"Locked output hash mismatch for {label}.")
    if snapshot.get("size") != expected_size:
        raise ValueError(f"Locked output size mismatch for {label}.")


def _load_semantic_score_snapshot(
    path: Path,
    record: Mapping[str, object],
) -> dict[str, object]:
    snapshot = _safe_file_snapshot(path, label="locked semantic scores")
    _verify_snapshot_record(
        snapshot,
        record,
        expected_path=path,
        label="semantic_scores.csv",
    )
    rows = _parse_csv_snapshot(snapshot, label="locked semantic scores")
    return {"snapshot": snapshot, "rows": rows}


def _numeric_value(
    row: Mapping[str, object],
    field: str,
    *,
    label: str,
) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has invalid {field}.") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} has non-finite {field}.")
    return value


def _one_candidate_row(
    rows: Sequence[Mapping[str, object]],
    candidate_id: str,
    *,
    label: str,
) -> Mapping[str, object]:
    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain the candidate exactly once.")
    return matches[0]


def _validate_reveal_topology(run_dir: Path, private_dir: Path) -> None:
    _assert_no_symlink_components(run_dir, "public reveal run")
    _assert_no_symlink_components(private_dir, "private reveal directory")
    lexical_run = Path(os.path.abspath(os.fspath(run_dir)))
    lexical_private = Path(os.path.abspath(os.fspath(private_dir)))
    if (
        lexical_run == lexical_private
        or lexical_run in lexical_private.parents
        or lexical_private in lexical_run.parents
    ):
        raise ValueError("Public and private reveal paths must be disjoint.")
    _assert_disjoint(run_dir, private_dir, "Public and private reveal paths")


def _verify_locked_public_snapshots(
    run_dir: Path,
) -> dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    manifest_snapshot = _safe_file_snapshot(manifest_path, label="v4 manifest")
    manifest = _parse_json_snapshot(manifest_snapshot, label="v4 manifest")
    if (
        manifest.get("schema_version") != 4
        or manifest.get("method_version") != _METHOD_VERSION
    ):
        raise ValueError("Expected a frozen v4 manifest snapshot.")

    lock_path = run_dir / "result.lock.json"
    lock_snapshot = _safe_file_snapshot(lock_path, label="v4 result lock")
    lock = _parse_json_snapshot(lock_snapshot, label="v4 result lock")
    if (
        lock.get("schema_version") != 4
        or lock.get("method_version") != _METHOD_VERSION
    ):
        raise ValueError("Expected a locked v4 result snapshot.")
    if lock.get("candidate_count") != 200 or lock.get("top_n") != 50:
        raise ValueError("Locked v4 candidate or Top 50 count is invalid.")
    if lock.get("manifest_sha256") != manifest_snapshot["sha256"]:
        raise ValueError("Result lock manifest hash mismatch.")
    expected_paths = {
        "semantic_embeddings.npy": run_dir / "semantic_embeddings.npy",
        "embedding_completion.npy": run_dir / "embedding_completion.npy",
        "checkpoint_metadata.json": run_dir / "checkpoints/embedding.metadata.json",
        "anchor_cosines.csv": run_dir / "anchor_cosines.csv",
        "semantic_scores.csv": run_dir / "semantic_scores.csv",
        "semantic_scores.jsonl": run_dir / "semantic_scores.jsonl",
        "diagnostics.json": run_dir / "diagnostics.json",
        "top50.csv": run_dir / "top50.csv",
    }
    outputs = _mapping(lock.get("outputs"), "v4 locked outputs")
    if set(outputs) != set(expected_paths):
        raise ValueError("v4 result lock output set is invalid.")

    snapshots: dict[str, Mapping[str, object]] = {}
    semantic_bundle = _load_semantic_score_snapshot(
        expected_paths["semantic_scores.csv"],
        _mapping(outputs["semantic_scores.csv"], "semantic score record"),
    )
    snapshots["semantic_scores.csv"] = _mapping(
        semantic_bundle["snapshot"], "semantic score snapshot"
    )
    for label, path in expected_paths.items():
        if label == "semantic_scores.csv":
            continue
        snapshot = _safe_file_snapshot(path, label=f"locked output {label}")
        _verify_snapshot_record(
            snapshot,
            _mapping(outputs[label], f"locked output record {label}"),
            expected_path=path,
            label=label,
        )
        snapshots[label] = snapshot

    semantic_rows = list(semantic_bundle["rows"])
    anchor_rows = _parse_csv_snapshot(
        snapshots["anchor_cosines.csv"], label="anchor cosine output"
    )
    top_rows = _parse_csv_snapshot(snapshots["top50.csv"], label="Top 50 output")
    if len(semantic_rows) != 200 or len(anchor_rows) != 200 or len(top_rows) != 50:
        raise ValueError("Locked v4 tabular output counts are invalid.")
    semantic_numeric_fields = [
        *[
            field
            for axis in _AXES
            for field in (
                f"{axis}_score",
                f"{axis}_rank",
                f"{axis}_percentile",
            )
        ],
        "semantic_score",
        "semantic_rank",
        "final_rank",
    ]
    semantic_ids: list[str] = []
    for row in semantic_rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("Locked semantic row has no candidate ID.")
        semantic_ids.append(candidate_id)
        for field in semantic_numeric_fields:
            _numeric_value(row, field, label="locked v4 semantic scores")
    if len(set(semantic_ids)) != 200:
        raise ValueError("Locked semantic candidate IDs are not unique.")
    if [int(float(row["final_rank"])) for row in semantic_rows] != list(
        range(1, 201)
    ):
        raise ValueError("Locked semantic final ranks are not ordered 1 through 200.")

    cosine_fields = [key for key in anchor_rows[0] if "_anchor_A" in key]
    if len(cosine_fields) != 30:
        raise ValueError("Locked anchor cosine output lacks 30 values per candidate.")
    for row in anchor_rows:
        for field in cosine_fields:
            _numeric_value(row, field, label="locked anchor cosine output")
    if [int(row["final_rank"]) for row in top_rows] != list(range(1, 51)):
        raise ValueError("Locked Top 50 ordering is invalid.")
    if [row.get("candidate_id") for row in top_rows] != semantic_ids[:50]:
        raise ValueError("Locked Top 50 candidate order differs from semantic scores.")
    try:
        completion = np.load(
            io.BytesIO(bytes(snapshots["embedding_completion.npy"]["bytes"])),
            allow_pickle=False,
        )
        embeddings = np.load(
            io.BytesIO(bytes(snapshots["semantic_embeddings.npy"]["bytes"])),
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError("Locked embedding output is invalid.") from error
    hidden_size = int(manifest.get("hidden_size", 0))
    if (
        completion.shape != (3, 211)
        or completion.dtype != np.bool_
        or not bool(completion.all())
        or embeddings.shape != (3, 211, hidden_size)
        or embeddings.dtype != np.float32
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Locked embedding shape, completion, or values are invalid.")
    diagnostics = _parse_json_snapshot(
        snapshots["diagnostics.json"], label="v4 diagnostics"
    )
    if diagnostics.get("schema_version") != 4:
        raise ValueError("Locked v4 diagnostics schema is invalid.")
    jsonl_lines = bytes(snapshots["semantic_scores.jsonl"]["bytes"]).splitlines()
    if len(jsonl_lines) != 200:
        raise ValueError("Locked semantic JSONL row count is invalid.")
    try:
        jsonl_rows = [json.loads(line.decode("utf-8")) for line in jsonl_lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Locked semantic JSONL is invalid.") from error
    if [row.get("candidate_id") for row in jsonl_rows] != semantic_ids:
        raise ValueError("Locked semantic JSONL candidate order differs from CSV.")
    for row in jsonl_rows:
        for field in semantic_numeric_fields:
            _numeric_value(row, field, label="locked semantic JSONL")

    manifest_inputs = _mapping(manifest.get("inputs"), "v4 manifest inputs")
    input_snapshots: dict[str, Mapping[str, object]] = {}
    for label, relative_path in {
        "candidates": "blind/candidates.csv",
        "stage1_scores": "stage1_scores.csv",
    }.items():
        path = run_dir / relative_path
        snapshot = _safe_file_snapshot(path, label=f"manifest input {label}")
        _verify_snapshot_record(
            snapshot,
            _mapping(manifest_inputs.get(label), f"manifest input record {label}"),
            expected_path=path,
            label=label,
        )
        input_snapshots[label] = snapshot
    return {
        "manifest": manifest,
        "manifest_snapshot": manifest_snapshot,
        "lock": lock,
        "lock_snapshot": lock_snapshot,
        "output_snapshots": snapshots,
        "input_snapshots": input_snapshots,
        "semantic_rows": semantic_rows,
    }


def reveal_v4_target(*, run_dir: Path, private_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    private_dir = Path(private_dir)
    _validate_reveal_topology(run_dir, private_dir)
    reveal_path = private_dir / "target_reveal.json"
    if reveal_path.exists():
        raise FileExistsError(f"Private v4 reveal already exists: {reveal_path}")

    verified = _verify_locked_public_snapshots(run_dir)
    manifest = _mapping(verified["manifest"], "verified v4 manifest")
    private_source_path = private_dir / "source_bundle_private.json"
    target_path = private_dir / "target_registry.json"
    private_source_snapshot = _safe_file_snapshot(
        private_source_path, label="private source-bundle record"
    )
    if private_source_snapshot["sha256"] != manifest.get(
        "source_bundle_private_sha256"
    ):
        raise ValueError("Private source record hash mismatch.")
    private_source = _parse_json_snapshot(
        private_source_snapshot, label="private source-bundle record"
    )
    manifest_private_hashes = _mapping(
        manifest.get("private_input_hashes"), "manifest private input hashes"
    )
    source_private_hashes = _mapping(
        private_source.get("copied_private_hashes"), "private copied hashes"
    )
    if dict(source_private_hashes) != dict(manifest_private_hashes):
        raise ValueError("Private source hashes differ from the frozen manifest.")

    target_snapshot = _safe_file_snapshot(target_path, label="private target registry")
    if target_snapshot["sha256"] != manifest_private_hashes.get("target_registry"):
        raise ValueError("Private target registry hash mismatch.")

    target = _parse_json_snapshot(target_snapshot, label="private target registry")
    candidate_id = target.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Target registry has no candidate_id.")

    input_snapshots = _mapping(verified["input_snapshots"], "verified input snapshots")
    candidates = _parse_csv_snapshot(
        _mapping(input_snapshots["candidates"], "candidate snapshot"),
        label="frozen candidates",
    )
    stage1_rows = _parse_csv_snapshot(
        _mapping(input_snapshots["stage1_scores"], "Stage 1 snapshot"),
        label="frozen Stage 1 scores",
    )
    semantic_rows = list(verified["semantic_rows"])
    candidate_row = _one_candidate_row(
        candidates, candidate_id, label="frozen candidates"
    )
    stage1_row = _one_candidate_row(
        stage1_rows, candidate_id, label="frozen Stage 1 scores"
    )
    semantic_row = _one_candidate_row(
        semantic_rows, candidate_id, label="locked v4 semantic scores"
    )
    if not _csv_bool(stage1_row.get("selected")):
        raise ValueError("Revealed target was not selected for v4 scoring.")
    connectivity = str(target.get("connectivity_smiles", ""))
    if (
        not connectivity
        or candidate_row.get("connectivity_smiles") != connectivity
        or semantic_row.get("connectivity_smiles") != connectivity
    ):
        raise ValueError("Target connectivity differs from frozen anonymous data.")

    axes = {
        axis: {
            "score": _numeric_value(
                semantic_row, f"{axis}_score", label="v4 semantic scores"
            ),
            "rank": _numeric_value(
                semantic_row, f"{axis}_rank", label="v4 semantic scores"
            ),
            "percentile": _numeric_value(
                semantic_row, f"{axis}_percentile", label="v4 semantic scores"
            ),
        }
        for axis in _AXES
    }
    semantic_rank = _numeric_value(
        semantic_row, "semantic_rank", label="v4 semantic scores"
    )
    final_rank = _numeric_value(semantic_row, "final_rank", label="v4 semantic scores")
    if not final_rank.is_integer():
        raise ValueError("v4 final rank must be an integer.")
    stage1_rank = _numeric_value(
        stage1_row, "stage1_rank", label="frozen Stage 1 scores"
    )
    if not stage1_rank.is_integer():
        raise ValueError("Stage 1 rank must be an integer.")
    success_rank = int(manifest.get("semantic_success_rank", 15))
    result = {
        "schema_version": 4,
        "method_version": _METHOD_VERSION,
        "revealed_at": _utc_now(),
        "manifest_sha256": _mapping(
            verified["manifest_snapshot"], "manifest snapshot"
        )["sha256"],
        "result_lock_sha256": _mapping(
            verified["lock_snapshot"], "result-lock snapshot"
        )["sha256"],
        "candidate_id": candidate_id,
        "identity": target,
        "stage1_rank": int(stage1_rank),
        "v4": {
            "axes": axes,
            "semantic_score": _numeric_value(
                semantic_row, "semantic_score", label="v4 semantic scores"
            ),
            "semantic_rank": semantic_rank,
            "final_rank": int(final_rank),
            "semantic_top15": semantic_rank <= success_rank,
        },
    }

    snapshots_to_recheck = [
        ("v4 manifest", _mapping(verified["manifest_snapshot"], "manifest snapshot")),
        ("v4 result lock", _mapping(verified["lock_snapshot"], "lock snapshot")),
        ("private source-bundle record", private_source_snapshot),
        ("private target registry", target_snapshot),
        *[
            (f"locked output {label}", _mapping(snapshot, label))
            for label, snapshot in _mapping(
                verified["output_snapshots"], "output snapshots"
            ).items()
        ],
        *[
            (f"manifest input {label}", _mapping(snapshot, label))
            for label, snapshot in input_snapshots.items()
        ],
    ]
    for label, snapshot in snapshots_to_recheck:
        _require_snapshot_unchanged(snapshot, label=label)
    _write_json_exclusive(reveal_path, result)
    return result
