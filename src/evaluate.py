"""src/evaluate.py
Independent evaluation / visualisation.
Fetches metrics from WandB and writes per-run & aggregated artefacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb  # type: ignore
from scipy import stats

sns.set(style="whitegrid")

# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------

def _plot_learning_curve(path: Path, history: pd.DataFrame, run_id: str) -> str:
    plt.figure(figsize=(6, 4))
    if "train_acc" in history:
        sns.lineplot(x=history["epoch"], y=history["train_acc"], label="train_acc")
    if "val_acc" in history:
        sns.lineplot(x=history["epoch"], y=history["val_acc"], label="val_acc")
    if "test_acc" in history:
        sns.lineplot(x=history["epoch"], y=history["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    fname = f"{run_id}_learning_curve.pdf"
    plt.savefig(path / fname)
    plt.close()
    return str(path / fname)


def _plot_confusion_matrix(cm: np.ndarray, classes: List[str], path: Path, run_id: str) -> str:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fname = f"{run_id}_confusion_matrix.pdf"
    plt.savefig(path / fname)
    plt.close()
    return str(path / fname)


def _bar_chart(path: Path, metric_dict: Dict[str, float], metric_name: str) -> str:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metric_dict.keys()), y=list(metric_dict.values()))
    for i, v in enumerate(metric_dict.values()):
        plt.text(i, v + (0.01 if metric_name != "relative_improvement" else 0.001), f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel(metric_name)
    plt.tight_layout()
    fname = f"comparison_{metric_name}_bar_chart.pdf"
    plt.savefig(path / fname)
    plt.close()
    return str(path / fname)


def _box_plot(path: Path, metric_dict: Dict[str, float], metric_name: str) -> str:
    plt.figure(figsize=(6, 4))
    df = pd.DataFrame({"run": list(metric_dict.keys()), metric_name: list(metric_dict.values())})
    sns.boxplot(x="run", y=metric_name, data=df)
    plt.tight_layout()
    fname = f"comparison_{metric_name}_boxplot.pdf"
    plt.savefig(path / fname)
    plt.close()
    return str(path / fname)

# -----------------------------------------------------------------------------
# Main evaluation pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WandB runs & generate artefacts")
    parser.add_argument("results_dir", type=str, help="Directory containing experiment metadata")
    parser.add_argument("run_ids", type=str, help="JSON list of WandB run IDs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    run_ids: List[str] = json.loads(args.run_ids)

    # ---------------------------------------------------------------------
    # Load global WandB project/entity information
    # ---------------------------------------------------------------------
    import yaml  # local import to avoid mandatory dependency when not needed

    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found – cannot resolve WandB project info")
    with cfg_path.open() as f:
        global_cfg = yaml.safe_load(f)
    entity = global_cfg["wandb"]["entity"]
    project = global_cfg["wandb"]["project"]

    api = wandb.Api()

    per_run_metric: Dict[str, float] = {}
    run_configs: Dict[str, dict] = {}
    generated_files: List[str] = []

    # ---------------------------------------------------------------------
    # STEP 1: Per-run processing
    # ---------------------------------------------------------------------
    for rid in run_ids:
        print(f"\nProcessing run {rid}")
        run = api.run(f"{entity}/{project}/{rid}")
        history = run.history(keys=["train_acc", "val_acc", "test_acc", "epoch"], pandas=True)
        summary = dict(run.summary._json_dict)
        config = dict(run.config)
        run_configs[rid] = config

        run_dir = results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics ----------------------------------------------------
        metrics_out = {
            "history": history.to_dict(orient="list"),
            "summary": summary,
            "config": config,
        }
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(metrics_out, f, indent=2)

        # Figures ---------------------------------------------------------
        lc_fig = _plot_learning_curve(run_dir, history, rid)
        generated_files.append(lc_fig)

        if "confusion_matrix" in summary:
            cm_arr = np.array(summary["confusion_matrix"])  # type: ignore[arg-type]
            class_names = config.get("dataset", {}).get("class_names", []) or [str(i) for i in range(cm_arr.shape[0])]
            cm_fig = _plot_confusion_matrix(cm_arr, class_names, run_dir, rid)
            generated_files.append(cm_fig)

        # Collect metric for aggregation ---------------------------------
        metric_value = summary.get("final_test_acc", summary.get("best_val_acc", 0.0))
        per_run_metric[rid] = float(metric_value)

    # ---------------------------------------------------------------------
    # STEP 2: Aggregated analysis
    # ---------------------------------------------------------------------
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Determine baseline (first non-proposed or first run if none) -------
    baseline_id = None
    for rid in run_ids:
        mtype = run_configs[rid].get("method", {}).get("type", "").lower()
        if mtype in {"baseline", "comparative", "comparative-1", "comparative_1"}:
            baseline_id = rid
            break
    if baseline_id is None:
        baseline_id = run_ids[0]
    baseline_value = per_run_metric[baseline_id]

    # Compute derived metrics -------------------------------------------
    absolute_diff: Dict[str, float] = {}
    relative_improvement: Dict[str, float] = {}
    for rid, value in per_run_metric.items():
        abs_diff = value - baseline_value
        rel_imp = abs_diff / baseline_value if baseline_value != 0 else float("nan")
        absolute_diff[rid] = abs_diff
        relative_improvement[rid] = rel_imp

    # Combine & save aggregated metrics ---------------------------------
    aggregated = {
        "per_run": per_run_metric,
        "baseline_id": baseline_id,
        "baseline_value": baseline_value,
        "absolute_difference": absolute_diff,
        "relative_improvement": relative_improvement,
    }

    with (comp_dir / "aggregated_metrics.json").open("w") as f:
        json.dump(aggregated, f, indent=2)

    # Visualisations -----------------------------------------------------
    bar_fig_acc = _bar_chart(comp_dir, per_run_metric, "accuracy")
    box_fig_acc = _box_plot(comp_dir, per_run_metric, "accuracy")
    bar_fig_rel = _bar_chart(comp_dir, relative_improvement, "relative_improvement")

    generated_files.extend([bar_fig_acc, box_fig_acc, bar_fig_rel])

    # Statistical significance tests (vs baseline) -----------------------
    sig_results = {}
    for k, v in per_run_metric.items():
        if k == baseline_id:
            continue
        # With single observations bootstrap w/1000 resamples could be used; simple Welch t-test here
        try:
            t_stat, p_val = stats.ttest_ind([baseline_value], [v], equal_var=False)
        except Exception:
            t_stat, p_val = float("nan"), float("nan")
        sig_results[k] = {"t_stat": t_stat, "p_val": p_val}

    with (comp_dir / "significance_tests.json").open("w") as f:
        json.dump(sig_results, f, indent=2)

    # ---------------------------------------------------------------------
    # Report paths of generated artefacts
    # ---------------------------------------------------------------------
    print("\nGenerated files:")
    for fp in generated_files:
        print(fp)
    print(comp_dir / "aggregated_metrics.json")
    print(comp_dir / "significance_tests.json")


if __name__ == "__main__":
    main()
