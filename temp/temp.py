import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
from typing import Literal, Dict, List
import logging
from datetime import datetime

def add_info_logging(text_info, type_logger="work_logger"):
    current_time = datetime.now()
    str_time = current_time.strftime("%H:%M")
    if type_logger == "work_logger":
        logger = logging.getLogger("work_logger")
        logger.info(f"time:  {str_time} {text_info}")
    elif type_logger == "result_logger":
        logger = logging.getLogger("result_logger")
        logger.info(f"time:  {str_time} {text_info}")


def summarize_and_plot(metrics: Dict[str, List[float]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for metric_name, values in metrics.items():
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        median = np.median(values)

        add_info_logging(f"\n📊 {metric_name} метрика:", "result_logger")
        add_info_logging(f"  ▸ Среднее: {mean:.4f}", "result_logger")
        add_info_logging(f"  ▸ Ст. отклонение: {std:.4f}", "result_logger")
        add_info_logging(f"  ▸ Медиана: {median:.4f}", "result_logger")

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

    if num_classes == 1:
        dice = f1_score(true_flat, pred_flat)
        iou = jaccard_score(true_flat, pred_flat)
    else:
        dice = f1_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        iou = jaccard_score(true_flat, pred_flat, average=average, labels=range(num_classes))

    return {
        'Dice': dice,
        'IoU': iou
    }

def aortic_mask_comparison(data_path):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", "Dataset401_AorticValve")
    original_mask_folder = os.path.join(nnUNet_folder, "original_mask", "Dataset401_AorticValve")

    dice_scores = []
    iou_scores = []

    for case in os.listdir(result_mask_folder):
        if not case.endswith(".nii.gz"):
            continue
        case_name = case[:-7]
        result_mask_path = os.path.join(result_mask_folder, case)
        original_mask_path = os.path.join(original_mask_folder, f"{case_name}.nii.gz")

        if not os.path.exists(original_mask_path):
            add_info_logging(f"⚠️ Пропущен {case_name} — нет оригинальной маски", "work_logger")
            continue

        result_mask = nib.load(result_mask_path).get_fdata()
        mask_img = nib.load(original_mask_path).get_fdata()

        if result_mask.shape != mask_img.shape:
            add_info_logging(f"❌ Размеры не совпадают в кейсе: {case_name}", "work_logger")
            add_info_logging(f"   → result_mask shape:  {result_mask.shape}", "work_logger")
            add_info_logging(f"   → original_mask shape:{mask_img.shape}", "work_logger")
            continue  # пропустить

        try:
            metrics = evaluate_segmentation(mask_img, result_mask)
            dice_scores.append(metrics["Dice"])
            iou_scores.append(metrics["IoU"])
        except Exception as e:
            add_info_logging(f"❗ Ошибка при сравнении {case_name}: {str(e)}", "work_logger")

    return {
        "Dice": dice_scores,
        "IoU": iou_scores
    }



def controller(data_path):
    result_path = os.path.join(data_path, "result")
    add_info_logging("Start", "work_logger")
    metrics = aortic_mask_comparison(data_path)
    summarize_and_plot(metrics, result_path)
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

    # Логгер для результатов
    result_file_name = current_time.strftime("result_%H_%M__%d_%m_%Y.log")
    result_log_path = os.path.join(data_path, result_file_name)
    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)
    result_handler = logging.FileHandler(result_log_path, mode='w')
    result_handler.setFormatter(logging.Formatter('%(message)s'))  # без времени
    result_logger.addHandler(result_handler)
    controller(data_path)