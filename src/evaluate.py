import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")


def metric_direction(metric_name: str) -> str:
    name = metric_name.lower()
    if any(
        key in name
        for key in ["loss", "error", "perplexity", "forward", "expanded", "rollback", "latency"]
    ):
        return "min"
    return "max"


def make_json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        val = obj.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val
    if isinstance(obj, (np.ndarray,)):
        return [make_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if obj is pd.NA:
        return None
    return str(obj)


def save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe = make_json_safe(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)


def parse_cli_args(argv: List[str]) -> Tuple[str, str]:
    kv_args: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            kv_args[key.lstrip("-")] = value
    if "results_dir" in kv_args and "run_ids" in kv_args:
        return kv_args["results_dir"], kv_args["run_ids"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--run_ids", type=str)
    parser.add_argument("results_dir_pos", nargs="?")
    parser.add_argument("run_ids_pos", nargs="?")
    args = parser.parse_args(argv)
    results_dir = args.results_dir or args.results_dir_pos
    run_ids = args.run_ids or args.run_ids_pos
    if results_dir is None or run_ids is None:
        raise ValueError(
            "Usage: python -m src.evaluate results_dir=<path> run_ids='[\"run-1\"]'"
        )
    return results_dir, run_ids


def run_mode_from_config(config: Dict) -> str | None:
    for key in ["mode", "cfg.mode", "cfg/mode"]:
        if key in config:
            return str(config[key])
    return None


def get_nested(config: Dict, keys: List[str], default=None):
    cur = config
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def is_classification_task(config: Dict, history: pd.DataFrame) -> bool:
    norm = get_nested(config, ["dataset", "preprocessing", "normalization"])
    if isinstance(norm, str):
        norm_l = norm.lower()
        if any(tok in norm_l for tok in ["int", "number", "numeric"]):
            return False
        if any(tok in norm_l for tok in ["ab", "choice", "label", "class", "option", "bool"]):
            return True
    task_type = get_nested(config, ["dataset", "task_type"]) or config.get("task_type")
    if isinstance(task_type, str) and any(tok in task_type.lower() for tok in ["class", "binary"]):
        return True
    dataset_name = get_nested(config, ["dataset", "name"], "")
    if isinstance(dataset_name, str):
        dname = dataset_name.lower()
        if "gsm8k" in dname:
            return False
        if any(tok in dname for tok in ["coinflip", "bool", "sentiment", "classification"]):
            return True
    if "test_gold" in history.columns and "test_pred" in history.columns:
        vals = pd.concat([history["test_gold"], history["test_pred"]], axis=0).dropna()
        uniq = pd.unique(vals.astype(str))
        if len(uniq) <= 10:
            numeric_like = all(v.lstrip("-").isdigit() for v in uniq)
            if numeric_like and len(uniq) > 5:
                return False
            return True
    return False


def plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths = []
    if "test_accuracy_running" in history.columns:
        x = history["_step"] if "_step" in history.columns else history.index
        y = history["test_accuracy_running"].ffill()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, label="Running Accuracy")
        if not y.empty:
            ax.scatter([x.iloc[-1]], [y.iloc[-1]], color="red", s=20)
            ax.annotate(f"{y.iloc[-1]:.3f}", (x.iloc[-1], y.iloc[-1]))
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("Running Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    if "risk_train_loss" in history.columns:
        x = history["_step"] if "_step" in history.columns else history.index
        y = history["risk_train_loss"].ffill()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, label="Risk Train Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Risk Model Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{run_id}_risk_train_loss.pdf")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_histogram(history: pd.DataFrame, column: str, run_id: str, out_dir: str) -> str | None:
    if column not in history.columns:
        return None
    vals = history[column].dropna().values
    if len(vals) == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=20, alpha=0.8, color="#4c72b0")
    ax.set_title(f"{column} Distribution")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_{column}_hist.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_scatter_forward_passes(history: pd.DataFrame, run_id: str, out_dir: str) -> str | None:
    if "test_forward_passes" not in history.columns or "test_accuracy_step" not in history.columns:
        return None
    vals = history[["test_forward_passes", "test_accuracy_step"]].dropna()
    if vals.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x=vals["test_forward_passes"],
        y=vals["test_accuracy_step"],
        ax=ax,
        alpha=0.6,
    )
    ax.set_xlabel("Forward Passes")
    ax.set_ylabel("Correct (0/1)")
    ax.set_title("Forward Passes vs Correctness")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_forward_passes_scatter.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_distribution(history: pd.DataFrame, run_id: str, out_dir: str) -> List[str]:
    paths = []
    if "test_error" in history.columns:
        vals = history["test_error"].dropna().values
        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(vals, bins=20, alpha=0.8, color="#55a868")
            ax.set_title("Signed Error Distribution")
            ax.set_xlabel("Error")
            ax.set_ylabel("Count")
            fig.tight_layout()
            path = os.path.join(out_dir, f"{run_id}_error_hist.pdf")
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    if "test_abs_error" in history.columns:
        vals = history["test_abs_error"].dropna().values
        if len(vals) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(vals, bins=20, alpha=0.8, color="#c44e52")
            ax.set_title("Absolute Error Distribution")
            ax.set_xlabel("Absolute Error")
            ax.set_ylabel("Count")
            fig.tight_layout()
            path = os.path.join(out_dir, f"{run_id}_abs_error_hist.pdf")
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
            fig, ax = plt.subplots(figsize=(6, 4))
            sorted_vals = np.sort(vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf)
            ax.set_title("CDF of Absolute Error")
            ax.set_xlabel("Absolute Error")
            ax.set_ylabel("CDF")
            fig.tight_layout()
            cdf_path = os.path.join(out_dir, f"{run_id}_abs_error_cdf.pdf")
            fig.savefig(cdf_path)
            plt.close(fig)
            paths.append(cdf_path)
    return paths


def plot_risk_coverage(history: pd.DataFrame, run_id: str, out_dir: str) -> str | None:
    if "test_selected_risk" not in history.columns or "test_accuracy_step" not in history.columns:
        return None
    vals = history[["test_selected_risk", "test_accuracy_step"]].dropna()
    if vals.empty:
        return None
    risks = vals["test_selected_risk"].values
    correct = vals["test_accuracy_step"].values
    thresholds = np.linspace(np.nanmin(risks), np.nanmax(risks), 11)
    coverages = []
    errors = []
    for t in thresholds:
        mask = risks <= t
        if mask.sum() == 0:
            coverages.append(0.0)
            errors.append(0.0)
            continue
        coverages.append(mask.mean())
        errors.append(1.0 - correct[mask].mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverages, errors, marker="o")
    ax.set_xlabel("Coverage (risk <= t)")
    ax.set_ylabel("Selective Error Rate")
    ax.set_title("Risk-Coverage Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_risk_coverage_curve.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_risk_calibration(history: pd.DataFrame, run_id: str, out_dir: str) -> str | None:
    if "test_selected_risk" not in history.columns or "test_accuracy_step" not in history.columns:
        return None
    vals = history[["test_selected_risk", "test_accuracy_step"]].dropna()
    if vals.empty:
        return None
    risks = vals["test_selected_risk"].values
    errors = 1.0 - vals["test_accuracy_step"].values
    bins = np.linspace(0.0, 1.0, 6)
    bin_ids = np.digitize(risks, bins) - 1
    bin_centers = []
    bin_errors = []
    for i in range(len(bins) - 1):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_errors.append(errors[mask].mean())
    if not bin_centers:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bin_centers, bin_errors, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Predicted Risk")
    ax.set_ylabel("Empirical Error")
    ax.set_title("Risk Calibration")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_risk_calibration.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(history: pd.DataFrame, run_id: str, out_dir: str) -> str | None:
    if "test_pred" not in history.columns or "test_gold" not in history.columns:
        return None
    vals = history[["test_pred", "test_gold"]].dropna()
    if vals.empty:
        return None
    y_true = vals["test_gold"].astype(str)
    y_pred = vals["test_pred"].astype(str)
    all_labels = pd.concat([y_true, y_pred], axis=0)
    counts = all_labels.value_counts()
    top_k = min(10, len(counts))
    top_labels = list(counts.index[:top_k])

    def map_label(x: str) -> str:
        return x if x in top_labels else "other"

    y_true_m = y_true.map(map_label)
    y_pred_m = y_pred.map(map_label)
    labels = top_labels.copy()
    if "other" in set(y_true_m) or "other" in set(y_pred_m):
        labels.append("other")
    cm = confusion_matrix(y_true_m, y_pred_m, labels=labels)
    size = max(6, int(0.6 * len(labels)) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion Matrix (Top Labels)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    results_dir, run_ids_raw = parse_cli_args(sys.argv[1:])

    cfg = OmegaConf.load("config/config.yaml")
    if getattr(cfg, "mode", "full") == "trial":
        raise RuntimeError("Evaluation cannot run in trial mode.")
    if cfg.wandb.mode == "disabled":
        raise RuntimeError("WandB is disabled; evaluation requires online runs with WandB logging.")
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    run_ids = json.loads(run_ids_raw)
    api = wandb.Api()

    generated_paths: List[str] = []
    summaries: Dict[str, Dict] = {}
    histories: Dict[str, pd.DataFrame] = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)
        run_mode = run_mode_from_config(config)
        if run_mode == "trial":
            raise RuntimeError(f"Run {run_id} was executed in trial mode; evaluation aborted.")

        run_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        history_clean = history.where(pd.notnull(history), None)
        history_records = [make_json_safe(row) for row in history_clean.to_dict(orient="records")]
        metrics_path = os.path.join(run_dir, "metrics.json")
        save_json(
            metrics_path,
            {
                "run_id": run_id,
                "summary": make_json_safe(summary),
                "config": make_json_safe(config),
                "history": history_records,
            },
        )
        generated_paths.append(metrics_path)

        generated_paths.extend(plot_learning_curve(history, run_id, run_dir))

        for column in ["test_forward_passes", "test_expanded_hypotheses", "test_rollback_events"]:
            p = plot_histogram(history, column, run_id, run_dir)
            if p:
                generated_paths.append(p)

        scatter_path = plot_scatter_forward_passes(history, run_id, run_dir)
        if scatter_path:
            generated_paths.append(scatter_path)

        generated_paths.extend(plot_error_distribution(history, run_id, run_dir))

        risk_curve = plot_risk_coverage(history, run_id, run_dir)
        if risk_curve:
            generated_paths.append(risk_curve)

        risk_cal = plot_risk_calibration(history, run_id, run_dir)
        if risk_cal:
            generated_paths.append(risk_cal)

        if is_classification_task(config, history):
            confusion_path = plot_confusion_matrix(history, run_id, run_dir)
            if confusion_path:
                generated_paths.append(confusion_path)

        summaries[run_id] = make_json_safe(summary)
        histories[run_id] = history

    comp_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    metrics = defaultdict(dict)
    for run_id, summary in summaries.items():
        for key, value in summary.items():
            if isinstance(value, (int, float, np.number, np.bool_)):
                metrics[key][run_id] = float(make_json_safe(value))

    primary_metric = "accuracy"
    accuracy_map = metrics.get(primary_metric, {})

    proposed_runs = {k: v for k, v in accuracy_map.items() if "proposed" in k}
    baseline_runs = {
        k: v for k, v in accuracy_map.items() if ("baseline" in k or "comparative" in k)
    }

    best_proposed = max(proposed_runs.items(), key=lambda x: x[1]) if proposed_runs else (None, None)
    best_baseline = max(baseline_runs.items(), key=lambda x: x[1]) if baseline_runs else (None, None)

    gap = None
    if best_proposed[0] and best_baseline[0] and best_baseline[1] not in [0, None]:
        gap = (best_proposed[1] - best_baseline[1]) / best_baseline[1] * 100
        if metric_direction(primary_metric) == "min":
            gap = -gap

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": metrics,
        "best_proposed": {"run_id": best_proposed[0], "value": best_proposed[1]},
        "best_baseline": {"run_id": best_baseline[0], "value": best_baseline[1]},
        "gap": gap,
    }
    agg_path = os.path.join(comp_dir, "aggregated_metrics.json")
    save_json(agg_path, aggregated)
    generated_paths.append(agg_path)

    if accuracy_map:
        fig, ax = plt.subplots(figsize=(7, 4))
        run_labels = sorted(accuracy_map.keys())
        values = [accuracy_map[k] for k in run_labels]
        sns.barplot(x=run_labels, y=values, ax=ax)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Comparison")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        for i, val in enumerate(values):
            ax.text(i, val + 0.001, f"{val:.3f}", ha="center")
        fig.tight_layout()
        path = os.path.join(comp_dir, "comparison_accuracy_bar_chart.pdf")
        fig.savefig(path)
        plt.close(fig)
        generated_paths.append(path)

    if histories:
        fp_records = []
        exp_records = []
        for run_id, history in histories.items():
            if "test_forward_passes" in history.columns:
                for val in history["test_forward_passes"].dropna().values:
                    fp_records.append({"run_id": run_id, "forward_passes": val})
            if "test_expanded_hypotheses" in history.columns:
                for val in history["test_expanded_hypotheses"].dropna().values:
                    exp_records.append({"run_id": run_id, "expanded": val})
        if fp_records:
            df_fp = pd.DataFrame(fp_records)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(x="run_id", y="forward_passes", data=df_fp, ax=ax)
            ax.set_title("Forward Passes Comparison")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            path = os.path.join(comp_dir, "comparison_forward_passes_boxplot.pdf")
            fig.savefig(path)
            plt.close(fig)
            generated_paths.append(path)
        if exp_records:
            df_exp = pd.DataFrame(exp_records)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(x="run_id", y="expanded", data=df_exp, ax=ax)
            ax.set_title("Expanded Hypotheses Comparison")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            path = os.path.join(comp_dir, "comparison_expanded_hypotheses_boxplot.pdf")
            fig.savefig(path)
            plt.close(fig)
            generated_paths.append(path)

    if "selective_error_rate_no_rollback" in metrics:
        ser_map = metrics["selective_error_rate_no_rollback"]
        fig, ax = plt.subplots(figsize=(7, 4))
        run_labels = sorted(ser_map.keys())
        values = [ser_map[k] for k in run_labels]
        sns.barplot(x=run_labels, y=values, ax=ax)
        ax.set_ylabel("Selective Error Rate")
        ax.set_title("Selective Error Rate (No Rollback)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        for i, val in enumerate(values):
            ax.text(i, val + 0.001, f"{val:.3f}", ha="center")
        fig.tight_layout()
        path = os.path.join(comp_dir, "comparison_selective_error_rate_bar_chart.pdf")
        fig.savefig(path)
        plt.close(fig)
        generated_paths.append(path)

    table_metrics = ["accuracy", "mean_forward_passes", "avg_expanded_hypotheses"]
    table_data = {metric: metrics.get(metric, {}) for metric in table_metrics}
    if table_data:
        df_table = pd.DataFrame(table_data)
        if not df_table.empty:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.axis("off")
            tbl = ax.table(
                cellText=df_table.values,
                rowLabels=df_table.index,
                colLabels=df_table.columns,
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            fig.tight_layout()
            path = os.path.join(comp_dir, "comparison_metrics_table.pdf")
            fig.savefig(path)
            plt.close(fig)
            generated_paths.append(path)

    if best_proposed[0] and best_baseline[0]:
        prop_hist = histories.get(best_proposed[0])
        base_hist = histories.get(best_baseline[0])
        if prop_hist is not None and base_hist is not None:
            if "test_accuracy_step" in prop_hist.columns and "test_accuracy_step" in base_hist.columns:
                prop_vals = prop_hist["test_accuracy_step"].dropna().values
                base_vals = base_hist["test_accuracy_step"].dropna().values
                if len(prop_vals) > 1 and len(base_vals) > 1:
                    stat, pval = ttest_ind(prop_vals, base_vals, equal_var=False)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.text(0.1, 0.5, f"t={stat:.3f}, p={pval:.4f}")
                    ax.set_axis_off()
                    ax.set_title("Significance Test (Accuracy)")
                    fig.tight_layout()
                    path = os.path.join(comp_dir, "comparison_significance_test.pdf")
                    fig.savefig(path)
                    plt.close(fig)
                    generated_paths.append(path)

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
