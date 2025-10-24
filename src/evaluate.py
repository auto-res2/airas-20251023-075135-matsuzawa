"""src/evaluate.py
Independent evaluation / visualisation.
Fetches metrics from WandB and writes per-run & aggregated artefacts.
"""
from __future__ import annotations

import argparse
import json
import sys
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
    plt.figure(figsize=(8, 5))
    has_data = False

    # Check for step column (either "Step" or "_step")
    step_col = "Step" if "Step" in history.columns else "_step" if "_step" in history.columns else None
    
    if step_col:
        history = history.dropna(subset=[step_col])

        if "train_acc" in history.columns and not history["train_acc"].isna().all():
            valid_data = history[[step_col, "train_acc"]].dropna()
            if len(valid_data) > 0:
                plt.plot(valid_data[step_col], valid_data["train_acc"], label="Train Accuracy", linewidth=2, marker='o', markersize=3)
                has_data = True
        if "val_acc" in history.columns and not history["val_acc"].isna().all():
            valid_data = history[[step_col, "val_acc"]].dropna()
            if len(valid_data) > 0:
                plt.plot(valid_data[step_col], valid_data["val_acc"], label="Val Accuracy", linewidth=2, marker='s', markersize=3)
                has_data = True
        if "test_acc" in history.columns and not history["test_acc"].isna().all():
            valid_data = history[[step_col, "test_acc"]].dropna()
            if len(valid_data) > 0:
                plt.plot(valid_data[step_col], valid_data["test_acc"], label="Test Accuracy", linewidth=2, marker='^', markersize=3)
                has_data = True
    else:
        # If no Step column, use index as x-axis
        x_values = range(len(history))
        if "train_acc" in history.columns and not history["train_acc"].isna().all():
            valid_indices = history["train_acc"].notna()
            if valid_indices.any():
                plt.plot(np.array(x_values)[valid_indices], history.loc[valid_indices, "train_acc"], label="Train Accuracy", linewidth=2.5, marker='o', markersize=4)
                has_data = True
        if "val_acc" in history.columns and not history["val_acc"].isna().all():
            valid_indices = history["val_acc"].notna()
            if valid_indices.any():
                plt.plot(np.array(x_values)[valid_indices], history.loc[valid_indices, "val_acc"], label="Val Accuracy", linewidth=2.5, marker='s', markersize=4)
                has_data = True
        if "test_acc" in history.columns and not history["test_acc"].isna().all():
            valid_indices = history["test_acc"].notna()
            if valid_indices.any():
                plt.plot(np.array(x_values)[valid_indices], history.loc[valid_indices, "test_acc"], label="Test Accuracy", linewidth=2.5, marker='^', markersize=4)
                has_data = True

    plt.xlabel("Step", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    if has_data:
        plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    else:
        # Add a note when no data is available
        plt.text(0.5, 0.5, 'No training history available\nfrom WandB run',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                transform=plt.gca().transAxes)
    plt.title(f"Learning Curve: {run_id}", fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    fname = f"{run_id}_learning_curve.pdf"
    plt.savefig(path / fname, dpi=300, bbox_inches='tight')
    plt.close()
    return str(path / fname)


def _plot_confusion_matrix(cm: np.ndarray, classes: List[str], path: Path, run_id: str) -> str:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                annot_kws={"fontsize": 11}, cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor='gray')
    plt.ylabel("True label", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted label", fontsize=14, fontweight='bold')
    plt.title("Confusion Matrix", fontsize=16, fontweight='bold', pad=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    fname = f"{run_id}_confusion_matrix.pdf"
    plt.savefig(path / fname, dpi=300, bbox_inches='tight')
    plt.close()
    return str(path / fname)


def _bar_chart(path: Path, metric_dict: Dict[str, float], metric_name: str) -> str:
    # Increase width to accommodate longer labels
    plt.figure(figsize=(14, 7))
    x_labels = list(metric_dict.keys())
    values = list(metric_dict.values())

    # Use different colors for better distinction
    colors = plt.cm.Set2(range(len(x_labels)))
    bars = plt.bar(range(len(x_labels)), values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    # Annotate bars with values
    for i, v in enumerate(values):
        if metric_name == "relative_improvement":
            offset = 0.0005 if v >= 0 else -0.0005
            va = "bottom" if v >= 0 else "top"
            plt.text(i, v + offset, f"{v:.4f}", ha="center", va=va, fontsize=12, fontweight='bold')
        else:
            offset = max(values) * 0.02 if metric_name == "accuracy" else max(values) * 0.02
            plt.text(i, v + offset, f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight='bold')

    # Better label handling: use abbreviated names
    shortened_labels = []
    for label in x_labels:
        # For long labels, replace common parts with abbreviations
        if len(label) > 25:
            # Keep first part (proposed/comparative) and last meaningful parts
            parts = label.split('-')
            if len(parts) >= 3:
                # Format: "proposed-Small-CNN-1.2M-CIFAR-10" -> "proposed\nSmall-CNN\n1.2M-CIFAR-10"
                if parts[0] in ['proposed', 'comparative'] and len(parts) > 4:
                    shortened = f"{parts[0]}\n{parts[1]}-{parts[2]}\n{'-'.join(parts[3:])}"
                else:
                    shortened = label
            else:
                shortened = label
        else:
            shortened = label
        shortened_labels.append(shortened)

    plt.xticks(range(len(x_labels)), shortened_labels, rotation=0, ha='center', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.title(f"Comparison: {metric_name.replace('_', ' ').title()}", fontsize=16, fontweight='bold', pad=15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tick_params(axis='y', labelsize=12)

    # Set y-axis for better visibility of differences
    if metric_name == "accuracy":
        # Start from 0 for proper visual representation
        plt.ylim(bottom=0, top=1.0)
    else:
        # For relative improvement, add padding to y-axis
        y_range = max(values) - min(values)
        padding = y_range * 0.2 if y_range > 0 else 0.001
        plt.ylim(bottom=min(values) - padding, top=max(values) + padding)

    plt.tight_layout()
    fname = f"comparison_{metric_name}_bar_chart.pdf"
    plt.savefig(path / fname, dpi=300, bbox_inches='tight')
    plt.close()
    return str(path / fname)


def _box_plot(path: Path, metric_dict: Dict[str, float], metric_name: str) -> str:
    # Increase width to accommodate longer labels
    plt.figure(figsize=(14, 7))
    df = pd.DataFrame({"run": list(metric_dict.keys()), metric_name: list(metric_dict.values())})

    # Use hue parameter to avoid deprecation warning
    sns.boxplot(x="run", y=metric_name, hue="run", data=df, palette="Set2", legend=False, linewidth=2)

    # Add individual points to show actual values
    sns.stripplot(x="run", y=metric_name, data=df, color='black', size=8, alpha=0.6, jitter=False)

    # Better label handling: use abbreviated names
    x_labels = list(metric_dict.keys())
    shortened_labels = []
    for label in x_labels:
        # For long labels, replace common parts with abbreviations
        if len(label) > 25:
            # Keep first part (proposed/comparative) and last meaningful parts
            parts = label.split('-')
            if len(parts) >= 3:
                # Format: "proposed-Small-CNN-1.2M-CIFAR-10" -> "proposed\nSmall-CNN\n1.2M-CIFAR-10"
                if parts[0] in ['proposed', 'comparative'] and len(parts) > 4:
                    shortened = f"{parts[0]}\n{parts[1]}-{parts[2]}\n{'-'.join(parts[3:])}"
                else:
                    shortened = label
            else:
                shortened = label
        else:
            shortened = label
        shortened_labels.append(shortened)

    plt.xticks(range(len(x_labels)), shortened_labels, rotation=0, ha='center', fontsize=12)
    plt.xlabel("Run", fontsize=14, fontweight='bold')
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    plt.title(f"Comparison: {metric_name.replace('_', ' ').title()} Distribution", fontsize=16, fontweight='bold', pad=15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tick_params(axis='y', labelsize=12)

    # Set y-axis for better visibility of differences
    if metric_name == "accuracy":
        # Start from 0 for proper visual representation
        plt.ylim(bottom=0, top=1.0)

    plt.tight_layout()
    fname = f"comparison_{metric_name}_boxplot.pdf"
    plt.savefig(path / fname, dpi=300, bbox_inches='tight')
    plt.close()
    return str(path / fname)

# -----------------------------------------------------------------------------
# Main evaluation pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    # Parse arguments in the format: key="value" or key=value
    # This handles both traditional argparse and key=value style arguments
    parsed_args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Remove quotes if present
            value = value.strip('"').strip("'")
            parsed_args[key] = value

    # If we have key=value style arguments, use them directly
    if 'results_dir' in parsed_args and 'run_ids' in parsed_args:
        results_dir = Path(parsed_args['results_dir']).expanduser().resolve()
        run_ids: List[str] = json.loads(parsed_args['run_ids'])
    else:
        # Fall back to traditional argparse
        parser = argparse.ArgumentParser(description="Evaluate WandB runs & generate artefacts")
        parser.add_argument("--results_dir", type=str, required=True, help="Directory containing experiment metadata")
        parser.add_argument("--run_ids", type=str, required=True, help="JSON list of WandB run IDs")
        args = parser.parse_args()
        results_dir = Path(args.results_dir).expanduser().resolve()
        run_ids = json.loads(args.run_ids)

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
        history = run.history(pandas=True)
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

    # Determine baseline (comparative or first run if none) -------
    baseline_id = None
    for rid in run_ids:
        if "comparative" in rid.lower():
            baseline_id = rid
            break
    if baseline_id is None:
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
