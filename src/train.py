"""src/train.py
Single-run training / hyper-parameter optimisation with Hydra, Optuna & WandB.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

from .model import build_model
from .preprocess import get_dataloaders

import wandb  # type: ignore

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return top-1 accuracy for a mini-batch."""
    return (pred.argmax(dim=1) == target).float().mean().item()


# -----------------------------------------------------------------------------
# Cost-aware compression helpers (BOIL-C)
# -----------------------------------------------------------------------------

def sigmoid_score(metric: float, m0: float = 0.5, g0: float = 0.1) -> float:
    return 1.0 / (1.0 + np.exp(-(metric - m0) / g0))


def compress_curve(score: float, cumulative_cost: float, beta: float) -> float:
    """Equation (BOIL-C): penalise by compute cost (seconds)."""
    return score - beta * np.log1p(cumulative_cost)


# -----------------------------------------------------------------------------
# Epoch-level routines
# -----------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg,
    wdb_run,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        if cfg.mode == "trial" and batch_idx >= cfg.trial_limited_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimiser.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimiser.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(out.detach(), y) * bs
        num_samples += bs

    epoch_loss = running_loss / num_samples
    epoch_acc = running_acc / num_samples
    if wdb_run is not None:
        wdb_run.log({"train_loss": epoch_loss, "train_acc": epoch_acc, "epoch": epoch})
    return epoch_loss, epoch_acc


def _eval_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split: str,
    wdb_run,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    num_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            acc_sum += accuracy(out, y) * bs
            num_samples += bs
    loss_avg = loss_sum / num_samples
    acc_avg = acc_sum / num_samples
    if wdb_run is not None:
        wdb_run.log({f"{split}_loss": loss_avg, f"{split}_acc": acc_avg, "epoch": epoch})
    return loss_avg, acc_avg


# -----------------------------------------------------------------------------
# Optuna objective – returns best validation accuracy of a trial
# -----------------------------------------------------------------------------

def build_objective(cfg, device):
    beta = float(cfg.run.method.get("beta", 0.0))

    def objective(trial: optuna.Trial):
        # Sample hyper-parameters ------------------------------------------------
        lr_params = dict(cfg.run.optuna.search_space.learning_rate)
        lr_type = lr_params.pop("type", None)
        if lr_type == "loguniform":
            lr_params["log"] = True
        lr = trial.suggest_float("learning_rate", **lr_params)
        
        batch_size = trial.suggest_categorical("batch_size", cfg.run.optuna.search_space.batch_size.choices)
        
        dropout_params = dict(cfg.run.optuna.search_space.dropout)
        dropout_type = dropout_params.pop("type", None)
        if dropout_type == "loguniform":
            dropout_params["log"] = True
        dropout = trial.suggest_float("dropout", **dropout_params)

        # Data loaders ----------------------------------------------------------
        train_loader, val_loader, _ = get_dataloaders(cfg, batch_size=batch_size)

        # Model / optimiser -----------------------------------------------------
        model = build_model(cfg, dropout).to(device)
        crit = nn.CrossEntropyLoss()
        opt = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.run.training.momentum,
            weight_decay=cfg.run.training.weight_decay,
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.run.training.epochs)

        best_val_acc = 0.0
        start_time = time.time()
        for ep in range(cfg.run.training.epochs):
            _train_one_epoch(model, train_loader, crit, opt, device, ep, cfg, wdb_run=None)
            _, v_acc = _eval_split(model, val_loader, crit, device, ep, "val", wdb_run=None)
            sch.step()
            best_val_acc = max(best_val_acc, v_acc)

            score = sigmoid_score(v_acc)
            scalar = (
                compress_curve(score, time.time() - start_time, beta)
                if cfg.run.method.name.lower() == "boil-c"
                else score
            )
            trial.report(scalar, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val_acc

    return objective


# -----------------------------------------------------------------------------
# Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):  # type: ignore
    # ------------------------------------------------------------------
    # Mode switch (trial / full)
    # ------------------------------------------------------------------
    if cfg.mode not in ("trial", "full"):
        raise ValueError("mode must be 'trial' or 'full'")
    OmegaConf.set_struct(cfg, False)
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.run.optuna.n_trials = 0
        cfg.run.training.epochs = 1
    else:
        cfg.wandb.mode = "online"

    # ------------------------------------------------------------------
    # Directories & config snapshot
    # ------------------------------------------------------------------
    results_dir = Path(cfg.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(results_dir / "config.yaml"))

    # ------------------------------------------------------------------
    # WandB session ----------------------------------------------------
    # ------------------------------------------------------------------
    wdb_run = None
    if cfg.wandb.mode != "disabled":
        wdb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            dir=str(results_dir),
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB URL: {wdb_run.url}")

    # ------------------------------------------------------------------
    # Reproducibility & device ----------------------------------------
    # ------------------------------------------------------------------
    set_seed(cfg.run.method.seeds[0] if "seeds" in cfg.run.method else 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data loaders (initial hyper-params from cfg) ---------------------
    # ------------------------------------------------------------------
    train_loader_full, val_loader_full, test_loader = get_dataloaders(cfg)

    # ------------------------------------------------------------------
    # Optuna hyper-parameter tuning -----------------------------------
    # ------------------------------------------------------------------
    best_params = {
        "learning_rate": float(cfg.run.training.learning_rate),
        "batch_size": int(cfg.run.training.batch_size),
        "dropout": float(cfg.run.model.dropout),
    }

    if int(cfg.run.optuna.n_trials) > 0:
        study = optuna.create_study(
            direction=cfg.run.optuna.direction,
            sampler=optuna.samplers.TPESampler(seed=0),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(build_objective(cfg, device), n_trials=int(cfg.run.optuna.n_trials))
        best_params.update(study.best_params)
        if wdb_run is not None:
            wdb_run.summary.update({f"optuna/best_{k}": v for k, v in study.best_params.items()})

    # ------------------------------------------------------------------
    # Final training with best hyper-params ----------------------------
    # ------------------------------------------------------------------
    train_loader, val_loader, _ = get_dataloaders(cfg, batch_size=int(best_params["batch_size"]))
    model = build_model(cfg, dropout=float(best_params["dropout"])).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(),
        lr=float(best_params["learning_rate"]),
        momentum=cfg.run.training.momentum,
        weight_decay=cfg.run.training.weight_decay,
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.run.training.epochs)

    best_val_acc = 0.0
    for ep in range(cfg.run.training.epochs):
        _train_one_epoch(model, train_loader, crit, opt, device, ep, cfg, wdb_run)
        _, v_acc = _eval_split(model, val_loader, crit, device, ep, "val", wdb_run)
        best_val_acc = max(best_val_acc, v_acc)
        sch.step()

    # ------------------------------------------------------------------
    # Final test evaluation & confusion matrix -------------------------
    # ------------------------------------------------------------------
    test_loss, test_acc = _eval_split(model, test_loader, crit, device, cfg.run.training.epochs, "test", wdb_run)

    all_preds: List[int] = []
    all_targets: List[int] = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.tolist())
    cm = confusion_matrix(all_targets, all_preds)
    class_names = test_loader.dataset.classes  # type: ignore[attr-defined]

    if wdb_run is not None:
        cm_plot = wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_targets,
            preds=all_preds,
            class_names=class_names,
        )
        wdb_run.log({"confusion_matrix": cm_plot})
        wdb_run.summary.update(
            {
                "best_val_acc": best_val_acc,
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
                "confusion_matrix": cm.tolist(),
            }
        )
        wdb_run.finish()


if __name__ == "__main__":
    main()
