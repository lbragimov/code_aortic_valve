import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from data_preprocessing.text_worker import add_info_logging

def summarize_and_plot(metrics: Dict[str, List[float]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for metric_name, values in metrics.items():
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        median = np.median(values)

        add_info_logging(f"\n {metric_name} metric:", "result_logger")
        add_info_logging(f"  Mean: {mean:.4f}", "result_logger")
        add_info_logging(f"  Std: {std:.4f}", "result_logger")
        add_info_logging(f"  Median: {median:.4f}", "result_logger")

        # Visualization
        plt.figure(figsize=(6, 4))
        plt.boxplot(values, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
        plt.scatter(values, np.ones_like(values), alpha=0.6, color='darkblue', label="Per-case values")
        plt.axvline(mean, color='red', linestyle='--', label=f"Mean = {mean:.3f}")
        plt.title(f"{metric_name} across all cases")
        plt.xlabel(metric_name)
        plt.legend()
        plt.tight_layout()

        # Save figure
        plot_path = os.path.join(save_dir, f"{metric_name}_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

def plot_group_comparison(metrics_by_group: Dict[str, Dict[str, List[float]]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    metrics = ["Dice", "IoU", "HD", "ASSD"]
    group_labels = ["all", "H", "p", "n"]
    colors = ["gray", "skyblue", "lightgreen", "salmon"]

    for metric in metrics:
        means = []
        stds = []
        for group in group_labels:
            values = metrics_by_group.get(group, {}).get(metric, [])
            means.append(np.nanmean(values))
            stds.append(np.nanstd(values))

        x = np.arange(len(group_labels))
        plt.figure(figsize=(6, 5))
        plt.bar(x, means, yerr=stds, color=colors, capsize=5)
        plt.xticks(x, group_labels)
        plt.ylabel(metric)
        plt.title(f"{metric} — comparison by group")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_barplot.png"), dpi=300)
        plt.close()
