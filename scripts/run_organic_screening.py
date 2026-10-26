from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "project_config" / "organic_additive_screening.json"
DEFAULT_ENVIRONMENT = PROJECT_ROOT / "environment.yml"
DEFAULT_MODEL_DIR = Path("/Users/chenlintao/models/ChemDFM-v2.0-14B")
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_PRIVATE_ROOT = PROJECT_ROOT / "private_runs"


def _common_run_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Context-centered organic-additive semantic screening."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import-source", help="Verify and copy the frozen screening input bundle."
    )
    _common_run_id(import_parser)
    import_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    import_parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    import_parser.add_argument("--source-public-dir", type=Path, required=True)
    import_parser.add_argument("--source-private-dir", type=Path, required=True)

    freeze_parser = subparsers.add_parser(
        "freeze", help="Freeze all verified inputs and inference settings."
    )
    _common_run_id(freeze_parser)
    freeze_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze_parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    freeze_parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    freeze_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    freeze_parser.add_argument("--batch-size", type=int, choices=(4, 8), default=8)
    freeze_parser.add_argument("--smoke-report", type=Path)

    screen_parser = subparsers.add_parser(
        "screen", help="Run only the blind semantic Stage 2."
    )
    _common_run_id(screen_parser)
    screen_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    reveal_parser = subparsers.add_parser(
        "reveal-target", help="Reveal the target after the result is locked."
    )
    _common_run_id(reveal_parser)
    reveal_parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    from semantic_fingerprint.screening.run_control import (
        freeze_run,
        import_source_bundle,
        reveal_target,
        screen_run,
    )

    args = build_parser().parse_args(argv)
    run_dir = args.runs_root / args.run_id
    if args.command == "import-source":
        result = import_source_bundle(
            config_path=args.config,
            source_public_dir=args.source_public_dir,
            source_private_dir=args.source_private_dir,
            run_dir=run_dir,
            private_dir=args.private_root / args.run_id,
        )
    elif args.command == "freeze":
        result = freeze_run(
            config_path=args.config,
            environment_path=args.environment,
            run_dir=run_dir,
            private_dir=args.private_root / args.run_id,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            smoke_report_path=args.smoke_report or run_dir / "smoke_report.json",
            repo_root=PROJECT_ROOT,
        )
    elif args.command == "screen":
        result = screen_run(run_dir=run_dir, model_dir=args.model_dir)
    elif args.command == "reveal-target":
        result = reveal_target(
            run_dir=run_dir,
            private_dir=args.private_root / args.run_id,
        )
    else:
        raise AssertionError(f"Unhandled screening command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
