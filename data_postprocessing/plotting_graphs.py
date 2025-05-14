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
    group_label_map = {
        "all": "All",
        "H": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }
    colors = ["lightgray", "skyblue", "lightgreen", "salmon"]

    # Добавим "all", если он не посчитан
    if "all" not in metrics_by_group:
        metrics_by_group["all"] = {}
        for metric in metrics:
            combined = []
            for group in metrics_by_group:
                if group == "all":
                    continue
                combined.extend(metrics_by_group[group].get(metric, []))
            metrics_by_group["all"][metric] = combined

    for metric in metrics:
        data = []
        labels = []
        means = []
        stds = []

        for group in group_labels:
            values = metrics_by_group.get(group, {}).get(metric, [])
            if values:
                data.append(values)
                labels.append(group_label_map.get(group, group))
                means.append(np.mean(values))
                stds.append(np.std(values))

        if not data:
            continue  # Пропуск метрики без данных

        plt.figure(figsize=(7, 5))
        bplot = plt.boxplot(data, patch_artist=True)

        # Цвета boxplot'ов
        for patch, color in zip(bplot['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)

        # Добавление scatter-значений
        for i, values in enumerate(data):
            y = values
            x = np.random.normal(loc=i+1, scale=0.05, size=len(values))  # чтобы точки не накладывались
            plt.scatter(x, y, alpha=0.6, color='black', s=20)

        # Добавление линий среднего и текстовых аннотаций
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.plot([i+0.8, i+1.2], [mean, mean], color='red', linestyle='--', linewidth=1.5)
            # plt.text(i+1, mean + std * 0.1, f"{mean:.3f} ± {std:.3f}",
            #          ha='center', va='bottom', fontsize=9, color='darkred')

        # plt.xticks(ticks=np.arange(1, len(labels)+1), labels=labels)
        # plt.ylabel(metric)
        # plt.title(f"{metric} — per-group distribution with mean ± std")
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, f"{metric}_group_boxplot_mean_std.png"), dpi=300)
        # plt.close()

        # Добавление легенды под графиком
        plt.plot([], [], color='red', linestyle='--', linewidth=1.5, label='Mean')
        plt.scatter([], [], color='black', alpha=0.6, s=20, label='Individual cases')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, shadow=False, ncol=2, fontsize=8)

        plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels)
        plt.ylabel(metric)
        plt.title(f"{metric} — per-group distribution with mean ± std")

        # Увеличиваем нижний отступ для легенды
        plt.subplots_adjust(bottom=0.25)

        plt.savefig(os.path.join(save_dir, f"{metric}_group_boxplot_mean_std.png"), dpi=300)
        plt.close()
