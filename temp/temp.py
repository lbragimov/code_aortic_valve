import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from data_preprocessing.text_worker import add_info_logging
from typing import Dict, List
import matplotlib.pyplot as plt
from data_postprocessing.mask_analysis import LandmarkCentersCalculator
from data_postprocessing.mask_analysis import mask_comparison

def plot_group_comparison(metric_name, group_label_map, data_pd, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    colors = ["lightgray", "skyblue", "lightgreen", "salmon", "orange", "violet", "gold"]

    data = []
    labels = []
    means = []
    stds = []

    for group_id, group_name in group_label_map:
        if group_id == "all":
            # Собираем все значения метрики из всех групп
            values = data_pd[metric_name].dropna().tolist()
        else:
            # Отбираем по конкретной группе
            values = data_pd[data_pd['group'] == group_id][metric_name].dropna().tolist()

        if values:
            data.append(values)
            labels.append(group_name)
            means.append(np.mean(values))
            stds.append(np.std(values))

    if not data:
        return # Пропуск метрики без данных

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

    # Добавление легенды под графиком
    plt.plot([], [], color='red', linestyle='--', linewidth=1.5, label='Mean')
    plt.scatter([], [], color='black', alpha=0.6, s=20, label='Individual cases')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=False, ncol=2, fontsize=8)

    plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} — per-group distribution with mean ± std")

    # Увеличиваем нижний отступ для легенды
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(os.path.join(save_dir, f"{metric_name}.png"), dpi=300)
    plt.close()


def mask_analysis(data_path, result_path, type_mask, folder_name):
    per_case_csv = os.path.join(result_path, "per_case_metrics.csv")

    group_label_map = {
        "all": "All",
        "H": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }
    metrics = ["Dice", "IoU", "HD", "ASSD"]

    if os.path.exists(per_case_csv):# and os.path.exists(aggregated_csv):
        # Загружаем данные из файлов
        add_info_logging("Using cached metrics from CSV files", "work_logger")
        df = pd.read_csv(per_case_csv)
    else:
        # Пересчитываем метрики
        _, per_case_data = mask_comparison(data_path, type_mask=type_mask, folder_name=folder_name)

        # Сохраняем метрики по кейсам
        df = pd.DataFrame(per_case_data)
        df.to_csv(per_case_csv, index=False)

    for metric_name in metrics:
        data_for_plot = df[['group', metric_name]].dropna(how='any')
        plot_group_comparison(metric_name, group_label_map, data_for_plot,
                              os.path.join(result_path, "group_comparison"))
    add_info_logging("Analysis completed", "work_logger")


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    add_info_logging("Start", "work_logger")

    mask_analysis(data_path, result_path, type_mask="aortic_valve", folder_name="Dataset401_AorticValve")
    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)