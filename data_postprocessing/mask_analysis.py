import os
import nibabel as nib
from data_postprocessing.evaluation_analysis import  evaluate_segmentation
from data_preprocessing.text_worker import add_info_logging


def mask_comparison(data_path, type_mask, folder_name):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    if type_mask == "aortic_valve":
        result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
        original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)
    elif type_mask == "aoric_landmarks":
        result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
        original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)

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