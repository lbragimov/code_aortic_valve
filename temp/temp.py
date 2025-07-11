from data_preprocessing.text_worker import add_info_logging
import os
import shutil
import math
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from data_preprocessing.text_worker import yaml_reader


def convert_dcm_to_nii(dicom_folder: str, nii_folder: str, zip: bool = False):

    if zip:
        output_nii_file = nii_folder + ".nii.gz"
    else:
        output_nii_file = nii_folder + ".nii"

    new_spacing = [0.4, 0.4, 0.4]

    # Reading a series of DICOM files
    reader = sitk.ImageSeriesReader()

    # Getting a list of DICOM files in the specified folder
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

    # Installing files in the reader
    reader.SetFileNames(dicom_series)

    # Reading images
    image = reader.Execute()

    # Saving an image in NIfTI format
    sitk.WriteImage(image, output_nii_file)


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
    dicom_folder = os.path.join(data_path, "dicom")
    result_folder = os.path.join(data_path, "result")
    img_folder = os.path.join(data_path, "img_nii")
    train_test_lists = read_csv(result_folder, "train_test_lists.csv") # дата фрейм

    dicom_folders = [p for p in Path(dicom_folder).iterdir() if p.is_dir()]

    clear_folder(img_folder)

    for folder_path in dicom_folders:
        org_folder_name = folder_path.name
        new_file_name = train_test_lists[train_test_lists['case_name'] == org_folder_name]['used_case_name']
        img_convert_path = os.path.join(img_folder, new_file_name)
        convert_dcm_to_nii(file_path, img_convert_path, zip=True)

    # add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller_path = "C:/Users/Kamil/Aortic_valve/code_aortic_valve/controller.yaml"
    controller_dump = yaml_reader(controller_path)
    controller()