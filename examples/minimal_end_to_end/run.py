"""Train and evaluate one compact synthetic AMOC workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from machine_learning.evaluate import export_predictions
from machine_learning.train import RunConfig, run


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "generated/data"
DEFAULT_OUTPUT = ROOT / "generated/runs"


def run_regime(regime: str, data_root: Path, output_root: Path) -> dict:
    """Run one epoch and reload the resulting checkpoint for evaluation."""
    summary = run(
        RunConfig(regime, 1, 42, bins=20, max_epochs=1, patience=0,
                  device="cpu", center_init="historical"),
        output_root=output_root,
        data_root=data_root,
    )
    checkpoint = output_root / f"{regime}_20/fold_01/seed_00042/best.pt"
    metrics = export_predictions(
        checkpoint,
        output_root / f"evaluation_{regime}",
        regime=regime,
        bins=20,
        fold=1,
        seed=42,
        data_root=data_root,
    )
    return {"training": summary, "evaluation": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("M", "Q", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.data_root.exists():
        raise SystemExit(
            "Test data not found. Run "
            "'python -m examples.minimal_end_to_end.generate_data' first."
        )
    regimes = ("M", "Q") if args.regime == "both" else (args.regime,)
    existing = [path for regime in regimes
                for path in (args.output_dir / f"{regime}_20/fold_01/seed_00042",
                             args.output_dir / f"evaluation_{regime}")
                if path.exists()]
    if existing:
        raise SystemExit("Refusing to replace existing test output: "
                         + ", ".join(str(path) for path in existing))
    results = {
        regime: run_regime(regime, args.data_root, args.output_dir)
        for regime in regimes
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
