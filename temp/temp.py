import os
import numpy as np
import pandas as pd
import logging
import nibabel as nib
from datetime import datetime
from data_preprocessing.text_worker import add_info_logging
from sklearn.metrics import jaccard_score, f1_score
from typing import Literal, Dict, List
import matplotlib.pyplot as plt
from medpy.metric.binary import hd, assd


def summarize_and_plot(metrics: Dict[str, List[float]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for metric_name, values in metrics.items():
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        median = np.median(values)

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


def evaluate_segmentation(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = 1,
                          average: Literal['macro', 'weighted'] = 'macro'):
    """
    Вычисляет Dice и IoU между масками.

    Parameters:
        true_mask (np.ndarray): Ground truth маска (2D или 3D).
        pred_mask (np.ndarray): Предсказанная маска (2D или 3D).
        num_classes (int): Количество классов. Если 1 — бинарная сегментация.
        average (str): Способ усреднения ('macro' или 'weighted').

    Returns:
        dict: {'Dice': ..., 'IoU': ...}
    """
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()

    metrics = {}

    if num_classes == 1:
        dice = f1_score(true_flat, pred_flat)
        iou = jaccard_score(true_flat, pred_flat)

        metrics["Dice"] = dice
        metrics["IoU"] = iou

        try:
            # Преобразуем в бинарные маски
            true_bin = (true_mask > 0).astype(np.bool_)
            pred_bin = (pred_mask > 0).astype(np.bool_)

            if np.count_nonzero(true_bin) == 0 or np.count_nonzero(pred_bin) == 0:
                metrics["HD"] = np.nan
                metrics["ASSD"] = np.nan
            else:
                metrics["HD"] = hd(pred_bin, true_bin)
                metrics["ASSD"] = assd(pred_bin, true_bin)
        except Exception as e:
            metrics["HD"] = np.nan
            metrics["ASSD"] = np.nan
    else:
        dice = f1_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        iou = jaccard_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        metrics["Dice"] = dice
        metrics["IoU"] = iou
        metrics["HD"] = np.nan  # многоклассовую HD/ASSD сложнее интерпретировать
        metrics["ASSD"] = np.nan

    return metrics


def mask_comparison(data_path, type_mask, folder_name):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
    original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)

    # dice_scores = []
    # iou_scores = []
    metric_groups = {
        "all": {"Dice": [], "IoU": [], "HD": [], "ASSD": []},
        "H": {"Dice": [], "IoU": [], "HD": [], "ASSD": []},
        "p": {"Dice": [], "IoU": [], "HD": [], "ASSD": []},
        "n": {"Dice": [], "IoU": [], "HD": [], "ASSD": []},
    }

    per_case_data = []

    for case in os.listdir(result_mask_folder):
        if not case.endswith(".nii.gz"):
            continue
        case_name = case[:-7]
        first_char = case_name[0]

        result_mask_path = os.path.join(result_mask_folder, case)
        original_mask_path = os.path.join(original_mask_folder, f"{case_name}.nii.gz")

        if not os.path.exists(original_mask_path):
            add_info_logging(f" Пропущен {case_name} — нет оригинальной маски", "work_logger")
            continue

        result_mask = nib.load(result_mask_path).get_fdata()
        mask_img = nib.load(original_mask_path).get_fdata()

        if result_mask.shape != mask_img.shape:
            add_info_logging(f" Размеры не совпадают в кейсе: {case_name}", "work_logger")
            add_info_logging(f" result_mask shape:  {result_mask.shape}", "work_logger")
            add_info_logging(f" original_mask shape:{mask_img.shape}", "work_logger")
            continue  # пропустить

        try:
            metrics = evaluate_segmentation(mask_img, result_mask)
            # Сохраняем по кейсу
            metrics["case"] = case_name
            metrics["group"] = first_char
            per_case_data.append(metrics)
            for key in metric_groups[first_char]:
                metric_groups[first_char][key].append(metrics[key])
                metric_groups["all"][key].append(metrics[key])
            # dice_scores.append(metrics["Dice"])
            # iou_scores.append(metrics["IoU"])
        except Exception as e:
            add_info_logging(f" Ошибка при сравнении {case_name}: {str(e)}", "work_logger")

    return metric_groups, per_case_data


def mask_analysis(data_path, result_path, type_mask):
    metrics_by_group, per_case_data = mask_comparison(data_path, type_mask=type_mask)
    for group, metrics in metrics_by_group.items():
        save_subdir = os.path.join(result_path, f"group_{group}")
        summarize_and_plot(metrics, save_subdir)
    # summarize_and_plot(metrics, result_path)
    # Сохраняем CSV по кейсам
    df = pd.DataFrame(per_case_data)
    df.to_csv(os.path.join(result_path, "per_case_metrics.csv"), index=False)

    # Сохраняем агрегированные данные
    summary = []
    for group, metrics in metrics_by_group.items():
        for metric_name, values in metrics.items():
            mean = np.nanmean(values)
            std = np.nanstd(values)
            summary.append({"Group": group, "Metric": metric_name, "Mean": mean, "Std": std})
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(result_path, "aggregated_metrics.csv"), index=False)

    # Рисуем сравнение групп
    plot_group_comparison(metrics_by_group, os.path.join(result_path, "group_comparison"))


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