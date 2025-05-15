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

def plot_group_comparison(metrics_by_group, save_dir: str, mode: str):
    os.makedirs(save_dir, exist_ok=True)

    if mode == "segmentation":
        metrics = ["Dice", "IoU", "HD", "ASSD"]
        group_labels = ["all", "H", "p", "n"]
        group_label_map = {
            "all": "All",
            "H": "Ger. path.",
            "p": "Slo. path.",
            "n": "Slo. norm."
        }
    elif mode == "landmarks":
        metrics = ["All cases", "Ger. path.", "Slo. path.", "Slo. norm."]
        group_labels = ["all", "r", "l", "n", "rnc", "rlc", "lnc"]
        group_label_map = {
            "all": "All landmarks",
            "r": "R",
            "l": "L",
            "n": "N",
            "rnc": "RNC",
            "rlc": "RLC",
            "lnc": "LNC"
        }
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    colors = ["lightgray", "skyblue", "lightgreen", "salmon", "orange", "violet", "gold"]

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
                    "point_id": key,
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

    results = []  # список словарей

    if find_center_mass:
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
            first_char = file.name[0]
            landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
            if len(landmarks_pred.keys()) < 5:
                add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                 "result_logger")
            if first_char == "H":
                num_img_ger_pat += 1
                not_found_ger_pat += compute_errors(landmarks_true, landmarks_pred, errors_ger_pat,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                    results, file.name, "H")
            if first_char == "p":
                # if file.name[1] == "9":
                #     continue
                num_img_slo_pat += 1
                not_found_slo_pat += compute_errors(landmarks_true, landmarks_pred, errors_slo_pat,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                    results, file.name, "p")
            if first_char == "n":
                num_img_slo_norm += 1
                not_found_slo_norm += compute_errors(landmarks_true, landmarks_pred, errors_slo_norm,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                    results, file.name, "n")

        # Сохраняем подробный CSV
        results_df = pd.DataFrame(results)
        results_csv_path = result_folder_path / f"landmark_errors_{ds_folder_name}.csv"
        results_df.to_csv(results_csv_path, index=False)

        mean_error_ger_pat = np.mean(errors_ger_pat) if errors_ger_pat else None
        not_found_ger_pat = (not_found_ger_pat / (num_img_ger_pat * 6)) * 100

        mean_error_slo_pat = np.mean(errors_slo_pat) if errors_slo_pat else None
        not_found_slo_pat = (not_found_slo_pat / (num_img_slo_pat * 6)) * 100

        mean_error_slo_norm = np.mean(errors_slo_norm) if errors_slo_norm else None
        not_found_slo_norm = (not_found_slo_norm / (num_img_slo_norm * 6)) * 100

        mean_error = np.mean(np.concatenate([errors_ger_pat, errors_slo_pat, errors_slo_norm]))
        num_img = num_img_ger_pat + num_img_slo_pat + num_img_slo_norm
        not_found = ((not_found_ger_pat + not_found_slo_pat + not_found_slo_norm) / (num_img * 6)) * 100
        # add_info_logging("finish analysis", "work_logger")
        mean_r_error = np.mean(r_errors) if r_errors else None
        mean_l_error = np.mean(l_errors) if l_errors else None
        mean_n_error = np.mean(n_errors) if n_errors else None
        mean_rlc_error = np.mean(rlc_errors) if rlc_errors else None
        mean_rnc_error = np.mean(rnc_errors) if rnc_errors else None
        mean_lnc_error = np.mean(lnc_errors) if lnc_errors else None

        add_info_logging("German pathology", "result_logger")
        add_info_logging(
            f"Mean Euclidean Distance: {mean_error_ger_pat:.4f} mm, not found: {not_found_ger_pat: .2f}%. Number of images:{num_img_ger_pat}",
            "result_logger")
        add_info_logging("Slovenian pathology", "result_logger")
        add_info_logging(
            f"Mean Euclidean Distance: {mean_error_slo_pat:.4f} mm, not found: {not_found_slo_pat: .2f}%. Number of images:{num_img_slo_pat}",
            "result_logger")
        add_info_logging("Slovenian normal", "result_logger")
        add_info_logging(
            f"Mean Euclidean Distance: {mean_error_slo_norm:.4f} mm, not found: {not_found_slo_norm: .2f}%. Number of images:{num_img_slo_norm}",
            "result_logger")
        add_info_logging("Sum", "result_logger")
        add_info_logging(
            f"Mean Euclidean Distance: {mean_error:.4f} mm, not found: {not_found: .2f}%. Number of images:{num_img}",
            "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'R' point: {mean_r_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'L' point: {mean_l_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'N' point: {mean_n_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'RLC' point: {mean_rlc_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'RNC' point: {mean_rnc_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'LNC' point: {mean_lnc_error}", "result_logger")

        plot_group_comparison(landmark_errors, str(result_folder_path), mode="landmarks")


def controller(data_path):
    add_info_logging("Start", "work_logger")

    landmarks_analysis(Path(data_path), ds_folder_name="Dataset499_AortaLandmarks",
                       find_center_mass=True, probabilities_map=True)
    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)