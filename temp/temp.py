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
    plt.ylabel(label_file_name)
    plt.title(f"{label_file_name} — per-group distribution with mean ± std")

    # Увеличиваем нижний отступ для легенды
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(os.path.join(save_dir, f"{label_file_name}.png"), dpi=300)
    plt.close()


def landmarks_analysis(data_path, ds_folder_name,
                       find_center_mass=False,
                       find_monte_carlo=False,
                       probabilities_map=False):

    def process_file(file, original_mask_folder, probabilities_map):
        res_test = LandmarkCentersCalculator()
        if probabilities_map:
            file_name = file.name[:-4] + ".nii.gz"
            pred = res_test.compute_metrics_direct_npz(original_mask_folder / file_name, file)
        else:
            pred = res_test.compute_metrics_direct_nii(file)
        true = res_test.compute_metrics_direct_nii(original_mask_folder / file.name)
        return true, pred

    def compute_errors(true, pred, error_list, r, l, n, rlc, rnc, lnc, results, file_name, group):
        not_found = 0
        for key in true:
            if key in pred:
                dist = np.linalg.norm(true[key] - pred[key]) # Евклидово расстояние
                error_list.append(dist)
                results.append({
                    "filename": file_name,
                    "group": group,
                    "point_id": point_name[key],
                    "error": dist
                })
                if key == 1: r.append(dist)
                elif key == 2: l.append(dist)
                elif key == 3: n.append(dist)
                elif key == 4: rlc.append(dist)
                elif key == 5: rnc.append(dist)
                elif key == 6: lnc.append(dist)
            else:
                not_found += 1
        return not_found

    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    result_folder_path = data_path / "result"
    results_csv_path = result_folder_path / f"landmark_errors_{ds_folder_name}.csv"

    point_name = {"all": "All", 1:"r", 2:"l", 3:"n", 4:"rlc", 5:"rnc", 6:"lnc"}
    type_label = {
        "all": "All",
        "H": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }

    results = []  # список словарей

    if find_center_mass:
        if os.path.exists(results_csv_path):
            # Загружаем данные из файлов
            add_info_logging("Using cached metrics from CSV files", "work_logger")
            results_df = pd.read_csv(results_csv_path)
        else:
            if probabilities_map:
                files = list(result_landmarks_folder.glob("*.npz"))
            else:
                files = list(result_landmarks_folder.glob("*.nii.gz"))
            errors_ger_pat, errors_slo_pat, errors_slo_norm = [], [], []
            not_found_ger_pat, not_found_slo_pat, not_found_slo_norm = 0, 0, 0
            num_img_ger_pat, num_img_slo_pat, num_img_slo_norm = 0, 0, 0
            r_errors, l_errors, n_errors = [], [], []
            rlc_errors, rnc_errors, lnc_errors = [], [], []
            for file in files:

                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")

                first_char = file.name[0]
                if first_char == "H":
                    num_img_ger_pat += 1
                    not_found_ger_pat += compute_errors(landmarks_true, landmarks_pred, errors_ger_pat,
                                                        r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                        results, file.name, "H")
                elif first_char == "p":
                    # if file.name[1] == "9":
                    #     continue
                    num_img_slo_pat += 1
                    not_found_slo_pat += compute_errors(landmarks_true, landmarks_pred, errors_slo_pat,
                                                        r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                        results, file.name, "p")
                elif first_char == "n":
                    num_img_slo_norm += 1
                    not_found_slo_norm += compute_errors(landmarks_true, landmarks_pred, errors_slo_norm,
                                                        r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                        results, file.name, "n")

            # Сохраняем подробный CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_csv_path, index=False)

        for key, type_name in type_label.items():
            if key == "all":
                data_for_plot = results_df[["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id", "error", point_name, data_for_plot,
                                      str(result_folder_path / "landmarks_comparsion"), type_name)
            else:
                data_for_plot = results_df[results_df['group'] == key][["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id","error", point_name, data_for_plot,
                                      str(result_folder_path / "landmarks_comparsion"), type_name)


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    add_info_logging("Start", "work_logger")

    landmarks_analysis(Path(data_path), ds_folder_name="Dataset499_AortaLandmarks",
                       find_center_mass=True, probabilities_map=True)
    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)