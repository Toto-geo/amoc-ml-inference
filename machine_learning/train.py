"""Train the AMOC model for one blocked fold and seed."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import DATA, SCALES, build_datasets
from .model import build_model


RUNS = Path(__file__).resolve().parent / "runs"
LR = 0.0021761392226802806                                  # base learning rate (HPO-selected)
WEIGHT_DECAY = 0.00029623847446488203                       # AdamW weight decay (HPO-selected)
GRAD_CLIP = 0.9160434996792657                              # global gradient-norm clip (HPO-selected)
# Per-group multipliers applied on top of LR; the sigma group uses a lighter weight decay.
GROUP_MULTIPLIERS = {"gnn": 0.7396125146505655,
                     "surface": 0.34033533774665374,
                     "head": 1.4910276924942758,
                     "sigma": 23.978457168916727,
                     "centers": 0.5078649058256767}
SEEDS = (42, 123, 456, 789, 2024, 2025, 2026, 31415, 27182, 16180,
         7, 99, 314, 1001, 1009, 1234, 4321, 9001, 12021, 65537)   # 20 fixed random seeds


@dataclass(frozen=True)
class RunConfig:
    regime: str
    fold: int
    seed: int
    bins: int = 20
    max_epochs: int = 140
    patience: int = 50
    device: str = "cpu"
    center_init: str = "kmeans"

    def validate(self) -> None:
        if self.regime not in ("M", "Q") or self.fold not in range(1, 9):
            raise ValueError("Use regime M/Q and fold 1..8")
        if self.bins not in SCALES:
            raise ValueError(f"Use a paper surface resolution: one of {tuple(SCALES)} bins")
        if self.seed not in SEEDS:
            raise ValueError("Use one of the 20 paper seeds: " + ", ".join(str(s) for s in SEEDS))
        if self.max_epochs <= 0 or self.patience < 0:
            raise ValueError("max_epochs must be positive and patience non-negative")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda or mps")
        if self.center_init not in ("historical", "kmeans"):
            raise ValueError("center_init must be historical or kmeans")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs so a run is reproducible on the same device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE plus a soft spread-matching penalty on the squared log of the std ratio."""
    eps = torch.tensor(1e-6, device=prediction.device, dtype=prediction.dtype)
    mse = torch.mean((prediction - target) ** 2)
    pred_std = torch.sqrt(torch.mean((prediction - prediction.mean()) ** 2) + eps)
    true_std = torch.sqrt(torch.mean((target - target.mean()) ** 2) + eps)
    ratio = pred_std / true_std.clamp_min(eps)              # ratio of predicted vs target spread
    return mse + 0.1 * torch.log(ratio.clamp_min(eps)) ** 2


def build_optimizer(model) -> torch.optim.Adam:
    """Partition parameters by group with group-specific learning rates and weight decay."""
    assigned = set()

    def collect(parameters):
        result = [p for p in parameters if p.requires_grad and id(p) not in assigned]
        assigned.update(id(p) for p in result)
        return result

    # Order matters: later groups skip parameters already claimed by earlier groups.
    groups = []
    for name, parameters in (
        ("sigma", [model.node_assigner.sigmas]),
        ("centers", [model.node_assigner.centers]),
        ("surface", model.surface_module.parameters()),
        ("head", model.prediction_head.parameters()),
    ):
        groups.append({"name": name, "params": collect(parameters),
                       "lr": LR * GROUP_MULTIPLIERS[name],
                       "weight_decay": 1e-5 if name == "sigma" else WEIGHT_DECAY})
    # Remaining trainable parameters (the GNN) go into the first group.
    groups.insert(0, {"name": "gnn", "params": collect(model.parameters()),
                      "lr": LR * GROUP_MULTIPLIERS["gnn"],
                      "weight_decay": WEIGHT_DECAY})
    if len(assigned) != sum(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Optimizer did not assign every trainable parameter")
    return torch.optim.Adam(groups, betas=(0.9, 0.999))


def train_epoch(model, loader, optimizer, device: torch.device) -> float:
    """Run one training epoch and return the sample-weighted mean loss."""
    model.train()
    total = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(batch)
        loss = loss_fn(prediction, batch["y"])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not torch.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError("Invalid training gradient")
        optimizer.step()
        total += float(loss.item()) * len(batch["y"])
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, stats: dict, device: torch.device) -> dict[str, float]:
    """Run the model over a loader and return normalised and physical-space metrics."""
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction, _ = model(batch)
        predictions.append(prediction.cpu())
        targets.append(batch["y"].cpu())
    pred, true = torch.cat(predictions), torch.cat(targets)
    if not torch.isfinite(pred).all():
        raise RuntimeError("Non-finite evaluation prediction")
    residual = pred - true
    # Undo train-only normalisation to get physical-space residuals.
    pred_phys = pred * stats["y_std_train"] + stats["y_mean_train"]
    true_phys = true * stats["y_std_train"] + stats["y_mean_train"]
    phys_residual = pred_phys - true_phys
    pred_std, true_std = float(pred.std(unbiased=False)), float(true.std(unbiased=False))
    spread_ratio = pred_std / max(true_std, 1e-12)
    pearson = float(np.corrcoef(true.numpy(), pred.numpy())[0, 1]) if (
        float(true_std) > 0 and float(pred_std) > 0) else float("nan")
    result = {
        "loss": float(loss_fn(pred, true)),
        "mse_norm": float(torch.mean(residual.square())),
        "mae_norm": float(torch.mean(residual.abs())),
        "bias_norm": float(torch.mean(residual)),
        "rmse_phys": float(torch.sqrt(torch.mean(phys_residual.square()))),
        "mae_phys": float(torch.mean(phys_residual.abs())),
        "bias_phys": float(torch.mean(phys_residual)),
        "pearson": pearson,
        "pred_std": pred_std, "true_std": true_std,
        "spread_ratio": spread_ratio,
        "spread_distance": abs(spread_ratio - 1.0),
        "r2": float(1 - torch.sum(residual.square()) / torch.sum((true - true.mean()).square())),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("Non-finite evaluation metric")
    return result


def is_better(loss: float, spread: float, best_loss: float, best_spread: float) -> bool:
    """Checkpoint selection tie band: 0.5% of max(abs(best loss), 1); ties broken by spread."""
    if not np.isfinite(best_loss):
        return True
    margin = 0.005 * max(abs(best_loss), 1.0)
    return loss < best_loss - margin or (abs(loss - best_loss) <= margin
                                         and spread < best_spread - 0.005 * max(abs(best_spread), 1.0))


def preflight(model, train, val, device: torch.device) -> float:
    """Validate the initial forward pass and gradient before optimisation begins.

    The order of these checks is load-bearing: training-mode forward calls
    advance dropout RNG, so reordering them changes a seeded run's output.
    """
    loader = DataLoader(train, batch_size=min(16, len(train)), shuffle=False)
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    if not all(bool(torch.isfinite(value).all()) for value in batch.values()):
        raise RuntimeError("Non-finite training input")
    wind = torch.zeros_like(batch["sst"])                         # inactive wind-stress slot
    model.eval()
    prediction, (graph, head) = model(batch)
    surface, diagnostics = model.surface_forward_with_diagnostics(
        wind, batch["sst"], batch["sss"], batch["ssh"], batch["obp"], batch["season"])
    if (prediction.shape != batch["y"].shape or head.shape[-1] != model.head_input_dim
            or surface.shape[-1] != model.surface_latent_dim
            or not torch.isfinite(graph).all() or not torch.isfinite(surface).all()
            or not torch.equal(diagnostics["surface_ablate_mask"].flatten(),
                               torch.tensor([0., 1., 1., 1., 1.], device=device))):
        raise RuntimeError("Initial model contract failed")
    # Training-mode forward and backward to confirm the gradient path is finite.
    model.train()
    model.zero_grad(set_to_none=True)
    prediction, _ = model(batch)
    loss = loss_fn(prediction, batch["y"])
    loss.backward()
    initial_loss = float(loss.detach())
    if not torch.isfinite(loss) or not all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
    ):
        raise RuntimeError("Invalid initial loss or gradient")
    model.zero_grad(set_to_none=True)

    # Walk the validation loader once and re-check that the wind slot stays inactive.
    for _ in DataLoader(val, batch_size=16, shuffle=False):
        pass
    loader = DataLoader(train, batch_size=min(16, len(train)), shuffle=False)
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    with torch.no_grad():
        model(batch)
        _, diagnostics = model.surface_forward_with_diagnostics(
            torch.zeros_like(batch["sst"]), batch["sst"], batch["sss"],
            batch["ssh"], batch["obp"], batch["season"])
    if not torch.equal(diagnostics["surface_ablate_mask"].flatten(),
                       torch.tensor([0., 1., 1., 1., 1.], device=device)):
        raise RuntimeError("Inactive wind slot became active")
    return initial_loss


def run(config: RunConfig, *, output_root: Path = RUNS,
        data_root: Path = DATA) -> dict:
    """Train one (regime, fold, seed) run end to end and write checkpoint, history and summary."""
    config.validate()
    torch.set_num_threads(1)
    seed_everything(config.seed)
    device = torch.device(config.device)
    train, val, test, stats = build_datasets(
        config.regime, config.fold, n_bins=config.bins, data_root=data_root
    )
    model = build_model(train, center_init=config.center_init).to(device)
    initial_centers = model.node_assigner.centers.detach().cpu().numpy().copy()
    initial_loss = preflight(model, train, val, device)
    optimizer = build_optimizer(model)
    model.node_assigner.sigmas.requires_grad_(False)             # sigmas frozen for the first 14 epochs

    run_dir = Path(output_root) / f"{config.regime}_{config.bins}" / f"fold_{config.fold:02d}" / f"seed_{config.seed:05d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    train_loader = DataLoader(train, batch_size=16, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(val, batch_size=16)
    test_loader = DataLoader(test, batch_size=16)
    best_loss, best_spread, best_epoch = float("inf"), float("inf"), -1
    history = []

    for epoch in range(config.max_epochs):
        if epoch == 14:
            model.node_assigner.sigmas.requires_grad_(True)      # unfreeze Gaussian sigmas after warm-up
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, stats, device)
        improved = is_better(val_metrics["loss"], val_metrics["spread_distance"],
                             best_loss, best_spread)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_spread_distance": val_metrics["spread_distance"],
                        "improved": int(improved)})
        if improved:
            best_loss, best_spread, best_epoch = (val_metrics["loss"],
                                                  val_metrics["spread_distance"], epoch)
            checkpoint_stats = {**stats, "locked_hpo_trial": "HPO1_M/trial_0340"}
            torch.save({"state_dict": {key: value.detach().cpu().clone()
                                       for key, value in model.state_dict().items()},
                        "config": {**asdict(config), "surface_n_bins": config.bins,
                                   "surface_lat_rows": SCALES[config.bins],
                                   "surface_temporal_mode": stats["surface_temporal_mode"],
                                   "surface_include_ws": False},
                        "norm_stats": checkpoint_stats, "epoch": epoch,
                        "best_checkpoint_metric_value": best_loss,
                        "init_centers": initial_centers}, run_dir / "best.pt")
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_metrics['loss']:.6f}"
              + (" *" if improved else ""), flush=True)
        if epoch - best_epoch >= config.patience:
            break

    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    # Reload the best checkpoint before the final test evaluation.
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_metrics = evaluate(model, test_loader, stats, device)
    summary = {"config": asdict(config), "initial_loss": initial_loss,
               "best_epoch": best_epoch,
               "epochs_completed": len(history), "best_val_loss": best_loss,
               "test_metrics": test_metrics,
               "test_used_for_model_selection": False}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("M", "Q"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bins", type=int, choices=tuple(SCALES), default=20)
    parser.add_argument("--max-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--center-init", choices=("historical", "kmeans"),
                        default="kmeans")
    parser.add_argument("--output-root", type=Path, default=RUNS)
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    run(RunConfig(args.regime, args.fold, args.seed, args.bins,
                  args.max_epochs, args.patience, args.device, args.center_init),
        output_root=args.output_root, data_root=args.data_root)


if __name__ == "__main__":
    main()
