from data_preprocessing.text_worker import add_info_logging
import os
import shutil
import math
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from data_preprocessing.text_worker import yaml_reader


def clear_folder(folder_path):
    """Очищает папку, удаляя все файлы и подпапки"""
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        # add_info_logging(f"The folder '{folder_path}' did not exist, so it was created.",
        #                  "work_logger")
        return

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()  # Удаляем файл или символическую ссылку
        elif item.is_dir():
            shutil.rmtree(item)  # Удаляем папку рекурсивно



def controller():

    for file_name in controller_dump["train_cases_list"]:
        if file_name.startswith("H"):
            sub_dir_name = "Homburg pathology"
        elif file_name.startswith("n"):
            sub_dir_name = "Normal"
        else:
            sub_dir_name = "Pathology"

        shutil.copy(str(os.path.join(crop_markers_mask_path, sub_dir_name, f"{file_name}.nii.gz")),
                    str(os.path.join(mask_train_folder, f"{file_name}.nii.gz")))

    # add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    controller_path = "C:/Users/Kamil/Aortic_valve/code_aortic_valve/controller.yaml"
    crop_markers_mask_path = os.path.join(data_path, "crop_markers_mask")
    ds_folder_name = "Dataset489_AortaLandmarks"
    mask_org_folder = os.path.join(nnUNet_folder, "original_mask", ds_folder_name)
    mask_train_folder = os.path.join(nnUNet_folder, "nnUNet_raw", ds_folder_name, "labelsTr")
    clear_folder(mask_train_folder)
    controller_dump = yaml_reader(controller_path)
    controller()