from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "project_config" / "organic_additive_screening.json"

EXPECTED_SYSTEM_PROMPT = (
    "You are a chemistry language model. Encode how the supplied molecular "
    "structure relates to one electrolyte-additive design axis. Do not infer "
    "measured battery performance and do not generate an explanation."
)
EXPECTED_USER_TEMPLATE = (
    "Electrolyte context:\n"
    "LiFSI/DME/TTE localized high-concentration electrolyte for a high-voltage "
    "NCM811||Li-metal full cell.\n\n"
    "Evaluation axis:\n"
    "{criterion}\n\n"
    "Candidate molecular card:\n"
    "{molecular_card}\n\n"
    "Encode the candidate molecular structure for comparison with reference "
    "electrolyte additives."
)
EXPECTED_CRITERIA = {
    "ncm811": (
        "Oxidative and interphase compatibility with high-voltage NCM811, "
        "favoring intrinsic stability or controlled self-limiting oxidation "
        "over persistent electrolyte decomposition and transition-metal "
        "dissolution."
    ),
    "lithium_metal": (
        "Reductive and interphase compatibility with lithium metal, favoring "
        "intrinsic stability or controlled self-limiting reduction, uniform "
        "lithium deposition, and low persistent parasitic consumption."
    ),
    "lhce_solvation": (
        "Moderate and non-disruptive interaction with LiFSI/DME/TTE solvation "
        "aggregates, avoiding both negligible participation and excessive "
        "lithium-ion capture or solvation-network disruption."
    ),
}


def _load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_repository_contains_only_formal_scientific_implementation() -> None:
    assert sorted(
        path.name
        for path in (ROOT / "semantic_fingerprint").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    ) == ["screening"]
    assert sorted(
        path.name for path in (ROOT / "project_config").glob("*.json")
    ) == ["organic_additive_screening.json"]
    assert sorted(path.name for path in (ROOT / "scripts").glob("*.py")) == [
        "run_organic_screening.py"
    ]
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(
            (ROOT / "semantic_fingerprint" / "screening").glob("*.py")
        )
    )
    assert "semantic_fingerprint.screening" in production


def test_config_freezes_context_centered_semantic_method() -> None:
    config = _load_config()

    assert config["schema_version"] == 4
    assert config["method_version"] == "organic-additive-semantic-fingerprint"
    assert config["source_bundle"] == {
        "expected_counts": {
            "candidate_count": 800,
            "anchor_count": 10,
            "stage1_embedding_rows": 810,
            "selected_count": 200,
        },
        "hashes": {
            "candidates": "9825817e08d9c5443d98ec8abb6188efd61884db5a338ba165a16fdc4bd82f06",
            "reference_anchors": "5887c78e1f7633999de1a91b7f5ed94651fe468d0bb8b217a5ab9a415a4056e1",
            "stage1_scores": "7c45c9ac7d144bd19c28fb98295b6aefc31dcfc96e277034197e250fdde92b",
            "stage1_embeddings": "738d7f850366c39906f9abfcbe77d4c1567f0029373c0ced96ba4f242a11c4cf",
            "molecular_cards": "23d29082e2f883f3303e1295d549fda183d344e0390bdd1bd8dfd636c05877ce",
            "target_registry": "8a072581dd89dda62dc6d6809ea573cdc66b924514818378be0c6041d960477a",
        },
    }
    assert config["screening"] == {
        "stage1_keep": 200,
        "top_n": 50,
        "semantic_success_rank": 15,
        "axis_order": ["ncm811", "lithium_metal", "lhce_solvation"],
        "axis_weights": {
            "ncm811": 1 / 3,
            "lithium_metal": 1 / 3,
            "lhce_solvation": 1 / 3,
        },
    }
    assert config["prompts"] == {
        "system": EXPECTED_SYSTEM_PROMPT,
        "user_template": EXPECTED_USER_TEMPLATE,
        "baseline_card": "No candidate molecular structure is provided.",
        "criteria": EXPECTED_CRITERIA,
    }
    assert config["embedding"] == {
        "chat_template": True,
        "add_generation_prompt": True,
        "hidden_layer": "last",
        "token_position": "last_non_padding_prompt_token",
        "raw_dtype": "float32",
        "center": "axis_specific_no_structure_baseline",
        "l2_normalize_after_centering": True,
        "epsilon": 1e-12,
        "entity_order": [
            "axis_baseline",
            "anchors_A01_to_A10",
            "selected_candidates_by_stage1_rank",
        ],
    }
    assert config["semantic_scoring"] == {
        "mode": "context_centered_anchor_cosine",
        "anchor_count": 10,
        "anchor_top_k": 2,
        "anchor_aggregation": "arithmetic_mean",
        "axis_rank_method": "descending_average_ties",
        "axis_percentile_formula": "(200-average_rank)/199",
        "semantic_aggregation": "equal_axis_percentile_mean",
        "semantic_rank_method": "descending_average_ties",
    }
    assert "structural_tie_breaker" not in config
    model = config["model"]
    assert model["repo_id"] == "OpenDFM/ChemDFM-v2.0-14B"
    assert model["revision"] == "b3b1d2143bb3e307e6b6dae794f6e9b0a83a45ef"
    assert model["device"] == "mps"
    assert model["dtype"] == "float16"
    assert model["weight_total_bytes"] == 29540067328
    assert set(model["weight_sha256"]) == {
        f"model-{index:05d}-of-00006.safetensors" for index in range(1, 7)
    }
    assert all(len(value) == 64 for value in model["weight_sha256"].values())
    assert all(len(value) == 64 for value in model["metadata_sha256"].values())


def test_scientific_boundary_contains_no_rejected_scoring_or_target_hint() -> None:
    config = _load_config()
    scientific_paths = [
        CONFIG,
        ROOT / "scripts" / "run_organic_screening.py",
        *sorted((ROOT / "semantic_fingerprint" / "screening").glob("*.py")),
    ]
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in scientific_paths)
    lowered = corpus.lower()

    forbidden = {
        "yes or no",
        "yes_logp",
        "no_logp",
        "positive_token",
        "negative_token",
        "candidate a",
        "candidate b",
        "state_sentence",
        "state likelihood",
        "10219385",
        "c1cocc1c#n",
        "c0686",
    }
    assert all(term not in lowered for term in forbidden)
    assert "target" not in json.dumps(config["semantic_scoring"]).lower()
    assert "structural" not in lowered


def test_cli_has_exact_commands_and_blind_screen_boundary() -> None:
    cli = importlib.import_module("scripts.run_organic_screening")
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )

    assert set(subparsers.choices) == {
        "import-source",
        "freeze",
        "screen",
        "reveal-target",
    }
    screen_options = {
        option
        for action in subparsers.choices["screen"]._actions
        for option in action.option_strings
    }
    assert screen_options == {"-h", "--help", "--run-id", "--runs-root", "--model-dir"}
    assert {"--private-root", "--target-registry", "--target-id"}.isdisjoint(
        screen_options
    )


@pytest.mark.parametrize(
    "command",
    ["import-source", "freeze", "screen", "reveal-target"],
)
def test_cli_requires_run_id(command: str) -> None:
    cli = importlib.import_module("scripts.run_organic_screening")
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args([command])

    assert exc_info.value.code == 2


def test_cli_defaults_are_isolated() -> None:
    cli = importlib.import_module("scripts.run_organic_screening")
    parser = cli.build_parser()

    imported = vars(
        parser.parse_args(
            [
                "import-source",
                "--run-id",
                "screening-run",
                "--source-public-dir",
                "/tmp/public",
                "--source-private-dir",
                "/tmp/private",
            ]
        )
    )
    assert imported == {
        "command": "import-source",
        "run_id": "screening-run",
        "runs_root": cli.DEFAULT_RUNS_ROOT,
        "config": cli.DEFAULT_CONFIG,
        "private_root": cli.DEFAULT_PRIVATE_ROOT,
        "source_public_dir": Path("/tmp/public"),
        "source_private_dir": Path("/tmp/private"),
    }
    frozen = vars(parser.parse_args(["freeze", "--run-id", "screening-run"]))
    assert frozen == {
        "command": "freeze",
        "run_id": "screening-run",
        "runs_root": cli.DEFAULT_RUNS_ROOT,
        "config": cli.DEFAULT_CONFIG,
        "private_root": cli.DEFAULT_PRIVATE_ROOT,
        "environment": cli.DEFAULT_ENVIRONMENT,
        "model_dir": cli.DEFAULT_MODEL_DIR,
        "batch_size": 8,
        "smoke_report": None,
    }
    screened = vars(parser.parse_args(["screen", "--run-id", "screening-run"]))
    assert screened == {
        "command": "screen",
        "run_id": "screening-run",
        "runs_root": cli.DEFAULT_RUNS_ROOT,
        "model_dir": cli.DEFAULT_MODEL_DIR,
    }
    revealed = vars(parser.parse_args(["reveal-target", "--run-id", "screening-run"]))
    assert revealed == {
        "command": "reveal-target",
        "run_id": "screening-run",
        "runs_root": cli.DEFAULT_RUNS_ROOT,
        "private_root": cli.DEFAULT_PRIVATE_ROOT,
    }


def test_run_roots_are_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "runs/" in ignored
    assert "private_runs/" in ignored
