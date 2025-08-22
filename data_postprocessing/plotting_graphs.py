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


def plot_group_comparison(filter_column, metric_name, group_label_map, data_pd, save_dir: str,
                          label_file_name: str=None):
    os.makedirs(save_dir, exist_ok=True)

    if not label_file_name:
        label_file_name = metric_name

    colors = ["lightgray", "skyblue", "lightgreen", "salmon", "orange", "violet", "gold"]

    data = []
    labels = []
    means = []
    stds = []

    for group_id, group_name in group_label_map.items():
        if group_id == "all":
            # Собираем все значения метрики из всех групп
            values = data_pd[metric_name].dropna().tolist()
        else:
            # Отбираем по конкретной группе
            values = data_pd[data_pd[filter_column] == group_id][metric_name].dropna().tolist()

        if values:
            data.append(values)
            labels.append(group_name)
            means.append(np.mean(values))
            stds.append(np.std(values))

    if not data:
        add_info_logging(f"No data found for metric {metric_name}. Skipping plot.", "work_logger")
        return  # Пропуск метрики без данных

    plt.figure(figsize=(7, 5))
    bplot = plt.boxplot(data, patch_artist=True)

    # Цвета boxplot'ов
    for patch, color in zip(bplot['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)

    # Добавление scatter-значений
    for i, values in enumerate(data):
        y = values
        x = np.random.normal(loc=i + 1, scale=0.05, size=len(values))  # чтобы точки не накладывались
        plt.scatter(x, y, alpha=0.6, color='black', s=20)

    # Добавление линий среднего и текстовых аннотаций
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.plot([i + 0.8, i + 1.2], [mean, mean], color='red', linestyle='--', linewidth=1.5)

    # Добавление легенды под графиком
    plt.plot([], [], color='red', linestyle='--', linewidth=1.5, label='Mean')
    plt.scatter([], [], color='black', alpha=0.6, s=20, label='Individual cases')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=False, ncol=2, fontsize=8)

    plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels)
    plt.ylabel(label_file_name)
    plt.title(f"{label_file_name} — per-group distribution with mean ± std")

    # Увеличиваем нижний отступ для легенды
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(os.path.join(save_dir, f"{label_file_name}.png"), dpi=300)
    plt.close()

def plot_table(data, columns, table_path):
    fig, ax = plt.subplots()
    ax.axis("off")  # убираем оси

    # Создаем таблицу
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.3)

    plt.savefig(table_path, dpi=300, bbox_inches="tight")  # сохраняем в PNG