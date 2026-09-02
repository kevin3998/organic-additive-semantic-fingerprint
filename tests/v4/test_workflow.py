from __future__ import annotations

import csv
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import semantic_fingerprint.v4.run_control as run_control
from scripts.run_organic_screening_v4 import main as v4_main
from semantic_fingerprint.v4.molecular_card import build_molecular_card_rows
from semantic_fingerprint.v4.run_control import (
    freeze_v4_run,
    import_source_bundle,
    reveal_v4_target,
    screen_v4_run,
)


AXES = ("ncm811", "lithium_metal", "lhce_solvation")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size": path.stat().st_size,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class WorkflowFixture:
    source_public: Path
    source_private: Path
    config: Path
    environment: Path
    model_dir: Path
    run_dir: Path
    private_dir: Path


def _build_fixture(
    tmp_path: Path,
    *,
    candidate_count: int = 4,
    anchor_count: int = 2,
    selected_count: int = 3,
) -> WorkflowFixture:
    source_public = tmp_path / "source-bundle" / "public"
    source_private = tmp_path / "source-bundle" / "private"
    blind = source_public / "blind"
    if candidate_count == 4:
        candidate_smiles = ["CCO", "CCN", "CCC", "COC"]
    else:
        suffixes = ("", "O", "N", "F", "Cl", "Br", "C#N", "C(=O)O")
        candidate_smiles = [
            "C" * chain_length + suffix
            for chain_length in range(1, 100)
            for suffix in suffixes
        ][:candidate_count]
    candidates = [
        {
            "candidate_id": f"C{index:04d}",
            "connectivity_smiles": smiles,
        }
        for index, smiles in enumerate(candidate_smiles, start=1)
    ]
    all_anchors = [
        {
            "anchor_id": "A01",
            "name": "anchor one",
            "connectivity_smiles": "O=C1OCCO1",
        },
        {
            "anchor_id": "A02",
            "name": "anchor two",
            "connectivity_smiles": "N#CCC#N",
        },
        {"anchor_id": "A03", "name": "anchor three", "connectivity_smiles": "CC#N"},
        {"anchor_id": "A04", "name": "anchor four", "connectivity_smiles": "CS(=O)(=O)C"},
        {"anchor_id": "A05", "name": "anchor five", "connectivity_smiles": "B(OC)(OC)OC"},
        {"anchor_id": "A06", "name": "anchor six", "connectivity_smiles": "P(OC)(OC)OC"},
        {"anchor_id": "A07", "name": "anchor seven", "connectivity_smiles": "C1COCO1"},
        {"anchor_id": "A08", "name": "anchor eight", "connectivity_smiles": "CC(F)(F)F"},
        {"anchor_id": "A09", "name": "anchor nine", "connectivity_smiles": "CCOC"},
        {"anchor_id": "A10", "name": "anchor ten", "connectivity_smiles": "S1COCO1"},
    ]
    anchors = all_anchors[:anchor_count]
    stage1 = [
        {
            **candidate,
            "stage1_score": 1.0 - index / 10.0,
            "stage1_rank": index,
            "selected": index <= selected_count,
        }
        for index, candidate in enumerate(candidates, start=1)
    ]
    cards = [
        {key: value for key, value in row.items()}
        for row in build_molecular_card_rows(candidates)
    ]

    paths = {
        "candidates": blind / "candidates.csv",
        "reference_anchors": blind / "reference_anchors.csv",
        "stage1_scores": source_public / "stage1_scores.csv",
        "stage1_embeddings": source_public / "stage1_embeddings.npy",
        "molecular_cards": blind / "molecular_cards.csv",
        "target_registry": source_private / "target_registry.json",
    }
    _write_csv(paths["candidates"], candidates)
    _write_csv(paths["reference_anchors"], anchors)
    _write_csv(paths["stage1_scores"], stage1)
    paths["stage1_embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(
        paths["stage1_embeddings"],
        np.arange(
            (candidate_count + anchor_count) * 4,
            dtype=np.float32,
        ).reshape(candidate_count + anchor_count, 4),
    )
    _write_csv(paths["molecular_cards"], cards)
    _write_json(
        paths["target_registry"],
        {
            "candidate_id": candidates[min(1, candidate_count - 1)]["candidate_id"],
            "name": "fixture identity",
            "connectivity_smiles": candidates[min(1, candidate_count - 1)][
                "connectivity_smiles"
            ],
        },
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_json(model_dir / "config.json", {"hidden_size": 8})
    weight_bytes = b"fixture weights"
    (model_dir / "model.safetensors").write_bytes(weight_bytes)
    _write_json(
        model_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": len(weight_bytes)},
            "weight_map": {"fixture.weight": "model.safetensors"},
        },
    )
    for name in (
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        _write_json(model_dir / name, {})
    (model_dir / "merges.txt").write_text("# fixture\n", encoding="utf-8")
    metadata_names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    config_path = tmp_path / "v4-config.json"
    _write_json(
        config_path,
        {
            "schema_version": 4,
            "method_version": "organic-semantic-fingerprint-v4",
            "source_bundle": {
                "expected_counts": {
                    "candidate_count": candidate_count,
                    "anchor_count": anchor_count,
                    "stage1_embedding_rows": candidate_count + anchor_count,
                    "selected_count": selected_count,
                },
                "hashes": {
                    "candidates": _sha(paths["candidates"]),
                    "reference_anchors": _sha(paths["reference_anchors"]),
                    "stage1_scores": _sha(paths["stage1_scores"]),
                    "stage1_embeddings": _sha(paths["stage1_embeddings"]),
                    "molecular_cards": _sha(paths["molecular_cards"]),
                    "target_registry": _sha(paths["target_registry"]),
                },
            },
            "screening": {
                "stage1_keep": selected_count,
                "top_n": min(50, max(1, selected_count - 1)),
                "semantic_success_rank": 15,
                "axis_order": list(AXES),
                "axis_weights": {axis: 1 / 3 for axis in AXES},
            },
            "molecular_card": {"numeric_precision": 3},
            "prompts": {
                "system": "system",
                "user_template": "{criterion}\n{molecular_card}",
                "baseline_card": "No candidate molecular structure is provided.",
                "criteria": {axis: f"criterion for {axis}" for axis in AXES},
            },
            "embedding": {
                "epsilon": 1e-12,
                "entity_order": [
                    "axis_baseline",
                    "anchors_A01_to_A10",
                    "selected_candidates_by_stage1_rank",
                ],
            },
            "semantic_scoring": {
                "mode": "context_centered_anchor_cosine",
                "anchor_top_k": 2,
            },
            "model": {
                "repo_id": "fixture/model",
                "revision": "fixture-revision",
                "local_dir": str(model_dir.resolve()),
                "device": "cpu",
                "dtype": "float32",
                "weight_total_bytes": (model_dir / "model.safetensors").stat().st_size,
                "metadata_sha256": {
                    name: _sha(model_dir / name) for name in metadata_names
                },
                "weight_sha256": {
                    "model.safetensors": _sha(model_dir / "model.safetensors")
                },
            },
        },
    )
    environment = tmp_path / "environment.yml"
    environment.write_text("name: fixture\n", encoding="utf-8")
    return WorkflowFixture(
        source_public=source_public,
        source_private=source_private,
        config=config_path,
        environment=environment,
        model_dir=model_dir,
        run_dir=tmp_path / "runs-v4" / "v4-run",
        private_dir=tmp_path / "private-v4" / "v4-run",
    )


def _import(fixture: WorkflowFixture) -> dict[str, object]:
    return import_source_bundle(
        config_path=fixture.config,
        source_public_dir=fixture.source_public,
        source_private_dir=fixture.source_private,
        run_dir=fixture.run_dir,
        private_dir=fixture.private_dir,
    )


def _write_smoke_report(fixture: WorkflowFixture, *, batch_size: int = 8) -> Path:
    path = fixture.run_dir / "smoke_report.json"
    selected = sorted(
        (
            row
            for row in _read_csv(fixture.run_dir / "stage1_scores.csv")
            if row["selected"].lower() == "true"
        ),
        key=lambda row: int(row["stage1_rank"]),
    )
    candidate_ids = [row["candidate_id"] for row in selected[:8]]
    _write_json(
        path,
        {
            "schema_version": 4,
            "method_version": "organic-semantic-fingerprint-v4",
            "batch_size": batch_size,
            "hidden_size": 8,
            "candidate_ids": candidate_ids,
            "token_lengths": {
                candidate_id: {axis: 20 + index for axis in AXES}
                for index, candidate_id in enumerate(candidate_ids)
            },
            "vector_shape": [3, len(candidate_ids), 8],
            "vector_count": 3 * len(candidate_ids),
            "all_finite": True,
            "minimum_centered_norm": 0.5,
        },
    )
    return path


def test_import_source_verifies_and_copies_complete_public_private_chain(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)

    result = _import(fixture)

    assert result["candidate_count"] == 4
    assert result["anchor_count"] == 2
    assert result["selected_count"] == 3
    public_files = {
        "blind/candidates.csv": fixture.source_public / "blind/candidates.csv",
        "blind/reference_anchors.csv": fixture.source_public / "blind/reference_anchors.csv",
        "stage1_scores.csv": fixture.source_public / "stage1_scores.csv",
        "stage1_embeddings.npy": fixture.source_public / "stage1_embeddings.npy",
        "blind/molecular_cards.csv": fixture.source_public / "blind/molecular_cards.csv",
    }
    for relative, source in public_files.items():
        copied = fixture.run_dir / relative
        assert _sha(copied) == _sha(source)
        assert (copied.stat().st_dev, copied.stat().st_ino) != (
            source.stat().st_dev,
            source.stat().st_ino,
        )
    assert len(_read_csv(fixture.run_dir / "blind/anchor_molecular_cards.csv")) == 2
    assert _sha(fixture.private_dir / "target_registry.json") == _sha(
        fixture.source_private / "target_registry.json"
    )
    public_source = json.loads(
        (fixture.run_dir / "blind/source_bundle.json").read_text(encoding="utf-8")
    )
    assert "copied_private_hashes" not in public_source
    assert "C0002" not in json.dumps(public_source)


@pytest.mark.parametrize(
    "tampered",
    [
        "candidates",
        "reference_anchors",
        "stage1_scores",
        "stage1_embeddings",
        "molecular_cards",
        "target_registry",
    ],
)
def test_import_source_rejects_every_tampered_source(
    tmp_path: Path,
    tampered: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    path = {
        "candidates": fixture.source_public / "blind/candidates.csv",
        "reference_anchors": fixture.source_public / "blind/reference_anchors.csv",
        "stage1_scores": fixture.source_public / "stage1_scores.csv",
        "stage1_embeddings": fixture.source_public / "stage1_embeddings.npy",
        "molecular_cards": fixture.source_public / "blind/molecular_cards.csv",
        "target_registry": fixture.source_private / "target_registry.json",
    }[tampered]
    path.write_bytes(path.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match="hash|Hash|JSON|source"):
        _import(fixture)
    assert not fixture.run_dir.exists()
    assert not fixture.private_dir.exists()


def test_import_source_rejects_overwrite_symlink_hardlink_and_crossed_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path / "overwrite")
    fixture.run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _import(fixture)

    fixture = _build_fixture(tmp_path / "symlink")
    candidate_path = fixture.source_public / "blind/candidates.csv"
    real_candidate_path = candidate_path.with_name("real-candidates.csv")
    candidate_path.rename(real_candidate_path)
    candidate_path.symlink_to(real_candidate_path)
    with pytest.raises(ValueError, match="symbolic link"):
        _import(fixture)

    fixture = _build_fixture(tmp_path / "hardlink")

    def hardlink_copy(source: str | Path, destination: str | Path) -> str:
        os.link(source, destination)
        return str(destination)

    monkeypatch.setattr(run_control.shutil, "copy2", hardlink_copy)
    with pytest.raises(RuntimeError, match="hard-linked"):
        _import(fixture)
    assert not fixture.run_dir.exists()
    assert not fixture.private_dir.exists()

    fixture = _build_fixture(tmp_path / "crossed")
    with pytest.raises(ValueError, match="disjoint"):
        import_source_bundle(
            config_path=fixture.config,
            source_public_dir=fixture.source_public,
            source_private_dir=fixture.source_private,
            run_dir=fixture.run_dir,
            private_dir=fixture.run_dir / "private",
        )


def test_freeze_records_import_model_prompts_entity_order_and_smoke(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    _import(fixture)
    smoke = _write_smoke_report(fixture)

    result = freeze_v4_run(
        config_path=fixture.config,
        environment_path=fixture.environment,
        run_dir=fixture.run_dir,
        private_dir=fixture.private_dir,
        model_dir=fixture.model_dir,
        batch_size=8,
        smoke_report_path=smoke,
        repo_root=Path(__file__).resolve().parents[2],
        require_clean=False,
    )

    manifest_path = fixture.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["manifest_sha256"] == _sha(manifest_path)
    assert manifest["schema_version"] == 4
    assert manifest["method_version"] == "organic-semantic-fingerprint-v4"
    assert manifest["batch_size"] == 8
    assert manifest["hidden_size"] == 8
    assert manifest["axis_order"] == list(AXES)
    assert manifest["entity_order"] == [
        "axis_baseline",
        "A01",
        "A02",
        "C0001",
        "C0002",
        "C0003",
    ]
    assert "structure_parameters" not in manifest
    assert set(manifest["prompt_sha256"]) == set(AXES) | {"system", "template"}
    assert manifest["inputs"]["smoke_report"]["sha256"] == _sha(smoke)
    assert manifest["inputs"]["model/model.safetensors"]["sha256"] == _sha(
        fixture.model_dir / "model.safetensors"
    )
    assert manifest["smoke_candidate_ids"] == ["C0001", "C0002", "C0003"]

    with pytest.raises(FileExistsError, match="Manifest"):
        freeze_v4_run(
            config_path=fixture.config,
            environment_path=fixture.environment,
            run_dir=fixture.run_dir,
            private_dir=fixture.private_dir,
            model_dir=fixture.model_dir,
            batch_size=8,
            smoke_report_path=smoke,
            repo_root=Path(__file__).resolve().parents[2],
            require_clean=False,
        )


def test_freeze_validates_tensor_bytes_from_safetensors_index_not_file_size(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    weight_path = fixture.model_dir / "model.safetensors"
    weight_path.write_bytes(b"safetensors header" + weight_path.read_bytes())
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["model"]["weight_sha256"]["model.safetensors"] = _sha(weight_path)
    _write_json(fixture.config, config)
    _import(fixture)
    smoke = _write_smoke_report(fixture)

    result = freeze_v4_run(
        config_path=fixture.config,
        environment_path=fixture.environment,
        run_dir=fixture.run_dir,
        private_dir=fixture.private_dir,
        model_dir=fixture.model_dir,
        batch_size=8,
        smoke_report_path=smoke,
        repo_root=Path(__file__).resolve().parents[2],
        require_clean=False,
    )

    assert result["hidden_size"] == 8


@pytest.mark.parametrize("mutation", ["batch", "source", "model", "smoke"])
def test_freeze_rejects_changed_or_invalid_frozen_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    _import(fixture)
    smoke = _write_smoke_report(fixture)
    batch_size = 8
    if mutation == "batch":
        batch_size = 6
    elif mutation == "source":
        (fixture.run_dir / "stage1_scores.csv").write_bytes(
            (fixture.run_dir / "stage1_scores.csv").read_bytes() + b"drift\n"
        )
    elif mutation == "model":
        (fixture.model_dir / "model.safetensors").write_bytes(b"changed")
    elif mutation == "smoke":
        report = json.loads(smoke.read_text(encoding="utf-8"))
        report["hidden_size"] = 7
        _write_json(smoke, report)

    with pytest.raises((TypeError, ValueError), match="batch|hash|Hash|smoke|hidden"):
        freeze_v4_run(
            config_path=fixture.config,
            environment_path=fixture.environment,
            run_dir=fixture.run_dir,
            private_dir=fixture.private_dir,
            model_dir=fixture.model_dir,
            batch_size=batch_size,
            smoke_report_path=smoke,
            repo_root=Path(__file__).resolve().parents[2],
            require_clean=False,
        )
    assert not (fixture.run_dir / "manifest.json").exists()


def test_freeze_requires_clean_tracked_git_worktree(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "fixture")
    _import(fixture)
    smoke = _write_smoke_report(fixture)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=repo,
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    tracked.write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tracked changes"):
        freeze_v4_run(
            config_path=fixture.config,
            environment_path=fixture.environment,
            run_dir=fixture.run_dir,
            private_dir=fixture.private_dir,
            model_dir=fixture.model_dir,
            batch_size=8,
            smoke_report_path=smoke,
            repo_root=repo,
        )


class DeterministicEmbeddingBackend:
    fail_after_calls: int | None = None
    emit_nonfinite = False
    call_count = 0
    encoded_prompt_count = 0
    instances: list["DeterministicEmbeddingBackend"] = []

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str,
        dtype: str,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.hidden_size = 8
        self.system_prompts: list[str] = []
        type(self).instances.append(self)

    @classmethod
    def reset(
        cls,
        *,
        fail_after_calls: int | None = None,
        emit_nonfinite: bool = False,
    ) -> None:
        cls.fail_after_calls = fail_after_calls
        cls.emit_nonfinite = emit_nonfinite
        cls.call_count = 0
        cls.encoded_prompt_count = 0
        cls.instances = []

    def encode_prompts(
        self,
        *,
        system_prompt: str,
        user_prompts: list[str],
        batch_size: int,
    ) -> np.ndarray:
        if (
            self.fail_after_calls is not None
            and type(self).call_count >= self.fail_after_calls
        ):
            raise RuntimeError("simulated batch interruption")
        type(self).call_count += 1
        type(self).encoded_prompt_count += len(user_prompts)
        self.system_prompts.append(system_prompt)
        vectors = np.array(
            [
                [
                    (byte + index + 1) / 300.0
                    for index, byte in enumerate(
                        hashlib.sha256(prompt.encode("utf-8")).digest()[:8]
                    )
                ]
                for prompt in user_prompts
            ],
            dtype=np.float32,
        )
        if self.emit_nonfinite:
            vectors[0, 0] = np.nan
        return vectors


def _frozen_screen_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> WorkflowFixture:
    fixture = _build_fixture(
        tmp_path,
        candidate_count=200,
        anchor_count=10,
        selected_count=200,
    )
    _import(fixture)
    smoke = _write_smoke_report(fixture)
    freeze_v4_run(
        config_path=fixture.config,
        environment_path=fixture.environment,
        run_dir=fixture.run_dir,
        private_dir=fixture.private_dir,
        model_dir=fixture.model_dir,
        batch_size=8,
        smoke_report_path=smoke,
        repo_root=Path(__file__).resolve().parents[2],
        require_clean=False,
    )
    DeterministicEmbeddingBackend.reset()
    monkeypatch.setattr(
        run_control,
        "ChemDFMContextEmbeddingBackend",
        DeterministicEmbeddingBackend,
        raising=False,
    )
    return fixture


def test_screen_writes_fixed_embeddings_scores_diagnostics_top50_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)

    result = screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)

    embeddings = np.load(fixture.run_dir / "semantic_embeddings.npy")
    completion = np.load(fixture.run_dir / "embedding_completion.npy")
    assert embeddings.shape == (3, 211, 8)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert completion.shape == (3, 211)
    assert completion.dtype == np.bool_
    assert completion.all()
    assert result["candidate_count"] == 200
    assert result["top_n"] == 50

    anchor_rows = _read_csv(fixture.run_dir / "anchor_cosines.csv")
    semantic_rows = _read_csv(fixture.run_dir / "semantic_scores.csv")
    top_rows = _read_csv(fixture.run_dir / "top50.csv")
    assert len(anchor_rows) == 200
    assert len(semantic_rows) == 200
    assert len(top_rows) == 50
    cosine_columns = [column for column in anchor_rows[0] if "_anchor_A" in column]
    assert len(cosine_columns) == 30
    expected_semantic_columns = {
        "candidate_id",
        "connectivity_smiles",
        *{f"{axis}_score" for axis in AXES},
        *{f"{axis}_rank" for axis in AXES},
        *{f"{axis}_percentile" for axis in AXES},
        "semantic_score",
        "semantic_rank",
        "final_rank",
    }
    assert set(semantic_rows[0]) == expected_semantic_columns
    numeric_columns = expected_semantic_columns - {
        "candidate_id",
        "connectivity_smiles",
    }
    assert all(
        np.isfinite(float(row[column]))
        for row in semantic_rows
        for column in numeric_columns
    )
    assert [int(row["final_rank"]) for row in top_rows] == list(range(1, 51))

    metadata = json.loads(
        (fixture.run_dir / "checkpoints/embedding.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["shape"] == [3, 211, 8]
    assert metadata["dtype"] == "float32"
    assert metadata["batch_size"] == 8
    lock_path = fixture.run_dir / "result.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert result["result_lock_sha256"] == _sha(lock_path)
    assert set(lock["outputs"]) == {
        "semantic_embeddings.npy",
        "embedding_completion.npy",
        "checkpoint_metadata.json",
        "anchor_cosines.csv",
        "semantic_scores.csv",
        "semantic_scores.jsonl",
        "diagnostics.json",
        "top50.csv",
    }
    assert all(
        record["sha256"] == _sha(Path(record["path"]))
        for record in lock["outputs"].values()
    )
    public_corpus = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in fixture.run_dir.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv"}
    )
    assert "fixture identity" not in public_corpus

    with pytest.raises(FileExistsError, match="locked"):
        screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)


def test_screen_interruption_resumes_only_whole_incomplete_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)
    DeterministicEmbeddingBackend.reset(fail_after_calls=2)

    with pytest.raises(RuntimeError, match="interruption"):
        screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)

    completion = np.load(fixture.run_dir / "embedding_completion.npy")
    assert completion[0, :16].all()
    assert not completion[0, 16:].any()
    assert not completion[1:].any()
    assert not (fixture.run_dir / "result.lock.json").exists()

    DeterministicEmbeddingBackend.reset()
    screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)

    assert DeterministicEmbeddingBackend.encoded_prompt_count == 3 * 211 - 16
    assert np.load(fixture.run_dir / "embedding_completion.npy").all()
    raw_records = _read_csv(fixture.run_dir / "semantic_scores.csv")
    assert len(raw_records) == 200


def test_screen_rejects_torn_completion_bitmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)
    DeterministicEmbeddingBackend.reset(fail_after_calls=1)
    with pytest.raises(RuntimeError, match="interruption"):
        screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)
    completion_path = fixture.run_dir / "embedding_completion.npy"
    completion = np.load(completion_path)
    completion[0, 0] = False
    with completion_path.open("wb") as handle:
        np.save(handle, completion, allow_pickle=False)

    DeterministicEmbeddingBackend.reset()
    with pytest.raises(ValueError, match="torn|batch"):
        screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)


@pytest.mark.parametrize("mutation", ["manifest", "model", "nonfinite"])
def test_screen_rejects_checkpoint_model_or_embedding_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)
    if mutation == "nonfinite":
        DeterministicEmbeddingBackend.reset(emit_nonfinite=True)
        with pytest.raises(ValueError, match="finite"):
            screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)
        assert not np.load(fixture.run_dir / "embedding_completion.npy").any()
        return

    DeterministicEmbeddingBackend.reset(fail_after_calls=1)
    with pytest.raises(RuntimeError, match="interruption"):
        screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)
    if mutation == "manifest":
        manifest_path = fixture.run_dir / "manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b" \n")
        message = "manifest|checkpoint"
    else:
        (fixture.model_dir / "model.safetensors").write_bytes(b"changed")
        message = "hash|model"
    DeterministicEmbeddingBackend.reset()
    with pytest.raises(ValueError, match=message):
        screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)


def test_screen_refuses_duplicate_process_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)
    lock_path = fixture.run_dir / "checkpoints/screen.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(FileExistsError, match="already in progress"):
            screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)


def _completed_screen_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> WorkflowFixture:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)
    screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)
    return fixture


def test_reveal_validates_locked_snapshot_then_reports_v4_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _completed_screen_fixture(tmp_path, monkeypatch)

    result = reveal_v4_target(
        run_dir=fixture.run_dir,
        private_dir=fixture.private_dir,
    )

    assert result["schema_version"] == 4
    assert result["method_version"] == "organic-semantic-fingerprint-v4"
    assert result["candidate_id"] == "C0002"
    assert result["identity"]["candidate_id"] == "C0002"
    assert result["stage1_rank"] == 2
    assert set(result) == {
        "schema_version",
        "method_version",
        "revealed_at",
        "manifest_sha256",
        "result_lock_sha256",
        "candidate_id",
        "identity",
        "stage1_rank",
        "v4",
    }
    assert set(result["v4"]["axes"]) == set(AXES)
    for axis in AXES:
        assert set(result["v4"]["axes"][axis]) == {
            "score",
            "rank",
            "percentile",
        }
    assert result["v4"]["semantic_top15"] is (
        result["v4"]["semantic_rank"] <= 15
    )
    assert result["manifest_sha256"] == _sha(fixture.run_dir / "manifest.json")
    assert result["result_lock_sha256"] == _sha(
        fixture.run_dir / "result.lock.json"
    )
    reveal_path = fixture.private_dir / "target_reveal.json"
    assert json.loads(reveal_path.read_text(encoding="utf-8")) == result
    assert not (fixture.run_dir / "target_reveal.json").exists()

    with pytest.raises(FileExistsError, match="reveal"):
        reveal_v4_target(run_dir=fixture.run_dir, private_dir=fixture.private_dir)


@pytest.mark.parametrize(
    "tampered",
    ["missing_lock", "result_lock", "semantic_scores", "target_registry"],
)
def test_reveal_rejects_missing_or_tampered_public_and_private_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered: str,
) -> None:
    fixture = _completed_screen_fixture(tmp_path, monkeypatch)
    path = {
        "missing_lock": fixture.run_dir / "result.lock.json",
        "result_lock": fixture.run_dir / "result.lock.json",
        "semantic_scores": fixture.run_dir / "semantic_scores.csv",
        "target_registry": fixture.private_dir / "target_registry.json",
    }[tampered]
    if tampered == "missing_lock":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match="lock|hash|snapshot|JSON|missing"):
        reveal_v4_target(run_dir=fixture.run_dir, private_dir=fixture.private_dir)
    assert not (fixture.private_dir / "target_reveal.json").exists()


def test_reveal_rejects_lexical_and_resolved_private_path_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _completed_screen_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="disjoint"):
        reveal_v4_target(run_dir=fixture.run_dir, private_dir=fixture.run_dir)

    alias = tmp_path / "private-alias"
    alias.symlink_to(fixture.private_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link|alias"):
        reveal_v4_target(run_dir=fixture.run_dir, private_dir=alias)


def test_reveal_detects_semantic_snapshot_mutation_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _completed_screen_fixture(tmp_path, monkeypatch)
    original = run_control._load_semantic_score_snapshot

    def mutate_after_snapshot(path: Path, record: dict[str, object]):
        result = original(path, record)
        path.write_bytes(path.read_bytes() + b"changed after verification\n")
        return result

    monkeypatch.setattr(
        run_control,
        "_load_semantic_score_snapshot",
        mutate_after_snapshot,
    )

    with pytest.raises(ValueError, match="snapshot changed"):
        reveal_v4_target(run_dir=fixture.run_dir, private_dir=fixture.private_dir)
    assert not (fixture.private_dir / "target_reveal.json").exists()


def test_reveal_checks_public_lock_before_reading_invalid_private_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _completed_screen_fixture(tmp_path, monkeypatch)
    semantic_path = fixture.run_dir / "semantic_scores.csv"
    semantic_path.write_bytes(semantic_path.read_bytes() + b"public drift\n")
    (fixture.private_dir / "target_registry.json").write_text(
        "not valid JSON",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="locked output|hash"):
        reveal_v4_target(run_dir=fixture.run_dir, private_dir=fixture.private_dir)


def test_exclusive_json_publish_leaves_no_partial_target_on_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "immutable.json"

    def fail_fsync(descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(run_control.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync"):
        run_control._write_json_exclusive(destination, {"complete": True})

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.publish"))


def test_cli_fake_model_runs_import_freeze_screen_and_reveal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _build_fixture(
        tmp_path,
        candidate_count=200,
        anchor_count=10,
        selected_count=200,
    )
    monkeypatch.setattr(
        run_control,
        "_git_state",
        lambda repo_root, require_clean: ("fixture-commit", ""),
    )
    DeterministicEmbeddingBackend.reset()
    monkeypatch.setattr(
        run_control,
        "ChemDFMContextEmbeddingBackend",
        DeterministicEmbeddingBackend,
    )

    assert v4_main(
        [
            "import-source",
            "--run-id",
            fixture.run_dir.name,
            "--runs-root",
            str(fixture.run_dir.parent),
            "--private-root",
            str(fixture.private_dir.parent),
            "--config",
            str(fixture.config),
            "--source-public-dir",
            str(fixture.source_public),
            "--source-private-dir",
            str(fixture.source_private),
        ]
    ) == 0
    smoke = _write_smoke_report(fixture)
    assert v4_main(
        [
            "freeze",
            "--run-id",
            fixture.run_dir.name,
            "--runs-root",
            str(fixture.run_dir.parent),
            "--private-root",
            str(fixture.private_dir.parent),
            "--config",
            str(fixture.config),
            "--environment",
            str(fixture.environment),
            "--model-dir",
            str(fixture.model_dir),
            "--batch-size",
            "8",
            "--smoke-report",
            str(smoke),
        ]
    ) == 0
    assert v4_main(
        [
            "screen",
            "--run-id",
            fixture.run_dir.name,
            "--runs-root",
            str(fixture.run_dir.parent),
            "--model-dir",
            str(fixture.model_dir),
        ]
    ) == 0
    assert v4_main(
        [
            "reveal-target",
            "--run-id",
            fixture.run_dir.name,
            "--runs-root",
            str(fixture.run_dir.parent),
            "--private-root",
            str(fixture.private_dir.parent),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert '"semantic_top15"' in output
    assert (fixture.run_dir / "result.lock.json").is_file()
    assert (fixture.private_dir / "target_reveal.json").is_file()


def test_screen_succeeds_after_private_run_directory_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)
    shutil.rmtree(fixture.private_dir)

    result = screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)

    assert result["candidate_count"] == 200
    assert (fixture.run_dir / "result.lock.json").is_file()


def test_screen_refuses_to_lock_output_changed_during_final_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _frozen_screen_fixture(tmp_path, monkeypatch)
    original = run_control._publish_result_bytes

    def mutate_before_lock(path: Path, content: bytes) -> dict[str, object]:
        record = original(path, content)
        if path.name == "top50.csv":
            semantic_path = path.parent / "semantic_scores.csv"
            semantic_path.write_bytes(semantic_path.read_bytes() + b"late drift\n")
        return record

    monkeypatch.setattr(run_control, "_publish_result_bytes", mutate_before_lock)

    with pytest.raises(ValueError, match="hash|size|snapshot"):
        screen_v4_run(run_dir=fixture.run_dir, model_dir=fixture.model_dir)
    assert not (fixture.run_dir / "result.lock.json").exists()


@pytest.mark.parametrize("mutation", ["declared_count", "nonfinite_other_row"])
def test_reveal_rejects_internally_rehashed_but_invalid_locked_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _completed_screen_fixture(tmp_path, monkeypatch)
    lock_path = fixture.run_dir / "result.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if mutation == "declared_count":
        lock["candidate_count"] = 199
    else:
        semantic_path = fixture.run_dir / "semantic_scores.csv"
        rows = _read_csv(semantic_path)
        rows[0]["ncm811_score"] = "nan"
        _write_csv(semantic_path, rows)
        lock["outputs"]["semantic_scores.csv"] = _record(semantic_path)
    _write_json(lock_path, lock)

    with pytest.raises(ValueError, match="count|finite|NaN"):
        reveal_v4_target(run_dir=fixture.run_dir, private_dir=fixture.private_dir)
