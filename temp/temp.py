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


def process_analysis(data_path, ds_folder_name,
                     find_center_mass=False,
                     find_monte_carlo=False,
                     probabilities_map=False):

    def process_file(file, original_mask_folder, probabilities_map):
        res_test = LandmarkCentersCalculator()
        if probabilities_map:
            file_name = file.name[:-4] + ".nii.gz"
            pred = res_test.compute_metrics_direct_npz(original_mask_folder / file_name, file)
            true = res_test.compute_metrics_direct_nii(original_mask_folder / file_name)
        else:
            pred = res_test.compute_metrics_direct_nii(file)
            true = res_test.compute_metrics_direct_nii(original_mask_folder / file.name)
        return true, pred

    def compute_errors(true, pred, error_list, r, l, n, rlc, rnc, lnc):
        # Вычисляем среднюю ошибку
        not_found = 0
        for key in true:
            if key in pred:
                dist = np.linalg.norm(true[key] - pred[key]) # Евклидово расстояние
                error_list.append(dist)
                if key == 1:
                    r.append(dist)
                elif key == 2:
                    l.append(dist)
                elif key == 3:
                    n.append(dist)
                elif key == 4:
                    rlc.append(dist)
                elif key == 5:
                    rnc.append(dist)
                elif key == 6:
                    lnc.append(dist)
            else:
                not_found += 1
        return not_found

    # add_info_logging("start analysis", "work_logger")
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    json_path = data_path / "nnUNet_folder" / "json_info"

    if find_center_mass:
        if probabilities_map:
            files = list(result_landmarks_folder.glob("*.npz"))
        else:
            files = list(result_landmarks_folder.glob("*.nii.gz"))
        errors_ger_pat = []
        not_found_ger_pat = 0
        num_img_ger_pat = 0
        errors_slo_pat = []
        not_found_slo_pat = 0
        num_img_slo_pat = 0
        errors_slo_norm = []
        not_found_slo_norm = 0
        num_img_slo_norm = 0
        r_errors = []
        l_errors = []
        n_errors = []
        rlc_errors = []
        rnc_errors = []
        lnc_errors = []
        for file in files:
            first_char = file.name[0]
            if first_char == "H":
                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                num_img_ger_pat += 1
                not_found_ger_pat += compute_errors(landmarks_true, landmarks_pred, errors_ger_pat,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors)
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")
            if first_char == "p":
                # if file.name[1] == "9":
                #     continue
                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                num_img_slo_pat += 1
                not_found_slo_pat += compute_errors(landmarks_true, landmarks_pred, errors_slo_pat,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors)
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")
            if first_char == "n":
                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                num_img_slo_norm += 1
                not_found_slo_norm += compute_errors(landmarks_true, landmarks_pred, errors_slo_norm,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors)
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")

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


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    add_info_logging("Start", "work_logger")

    ds_folder_name = "Dataset404_AortaLandmarks"
    data_path_2 = Path(data_path)
    process_analysis(data_path=data_path_2, ds_folder_name=ds_folder_name, find_center_mass=True, probabilities_map=True)
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