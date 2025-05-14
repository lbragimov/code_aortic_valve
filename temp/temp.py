import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from data_preprocessing.text_worker import add_info_logging
from typing import Dict, List
import matplotlib.pyplot as plt
from data_postprocessing.mask_analysis import mask_comparison
from data_postprocessing.plotting_graphs import summarize_and_plot


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
            plt.text(i+1, mean + std * 0.1, f"{mean:.3f} ± {std:.3f}",
                     ha='center', va='bottom', fontsize=9, color='darkred')

        plt.xticks(ticks=np.arange(1, len(labels)+1), labels=labels)
        plt.ylabel(metric)
        plt.title(f"{metric} — per-group distribution with mean ± std")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_group_boxplot_mean_std.png"), dpi=300)
        plt.close()


def mask_analysis(data_path, result_path, type_mask, folder_name):
    per_case_csv = os.path.join(result_path, "per_case_metrics.csv")
    aggregated_csv = os.path.join(result_path, "aggregated_metrics.csv")

    if os.path.exists(per_case_csv) and os.path.exists(aggregated_csv):
        # Загружаем данные из файлов
        add_info_logging("Using cached metrics from CSV files", "work_logger")
        df = pd.read_csv(per_case_csv)
        # df.columns = df.columns.str.strip().str.lower()
        df_summary = pd.read_csv(aggregated_csv)
        # df_summary.columns = df_summary.columns.str.strip().str.lower()

        # Восстановим структуру metrics_by_group
        metrics_by_group: Dict[str, Dict[str, list]] = {}
        for _, row in df_summary.iterrows():
            group = row["Group"]
            metric = row["Metric"]
            if group not in metrics_by_group:
                metrics_by_group[group] = {}
            if metric not in metrics_by_group[group]:
                # Найдём все значения этой метрики и группы из df
                values = df[df["group"] == group][metric].dropna().tolist()
                metrics_by_group[group][metric] = values
    else:
        # Пересчитываем метрики
        metrics_by_group, per_case_data = mask_comparison(data_path, type_mask=type_mask, folder_name=folder_name)

        # Сохраняем метрики по кейсам
        df = pd.DataFrame(per_case_data)
        df.to_csv(per_case_csv, index=False)

        # Сохраняем агрегированные метрики
        summary = []
        for group, metrics in metrics_by_group.items():
            for metric_name, values in metrics.items():
                mean = np.nanmean(values)
                std = np.nanstd(values)
                summary.append({"Group": group, "Metric": metric_name, "Mean": mean, "Std": std})
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(aggregated_csv, index=False)

    # Строим графики
    for group, metrics in metrics_by_group.items():
        save_subdir = os.path.join(result_path, f"group_{group}")
        summarize_and_plot(metrics, save_subdir)

    plot_group_comparison(metrics_by_group, os.path.join(result_path, "group_comparison"))
    add_info_logging("Analysis completed", "work_logger")


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    add_info_logging("Start", "work_logger")

    mask_analysis(data_path, result_path, type_mask="aortic_valve")
    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    current_time = datetime.now()
    # Логгер для хода программы
    log_file_name = current_time.strftime("log_%H_%M__%d_%m_%Y.log")
    work_log_path = os.path.join(data_path, log_file_name)
    work_logger = logging.getLogger("work_logger")
    work_logger.setLevel(logging.INFO)
    work_handler = logging.FileHandler(work_log_path, mode='w')
    work_handler.setFormatter(logging.Formatter('%(message)s'))
    work_logger.addHandler(work_handler)

    controller(data_path)