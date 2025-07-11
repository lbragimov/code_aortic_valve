from data_preprocessing.text_worker import add_info_logging
import os
import shutil
import math
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from data_preprocessing.text_worker import yaml_reader


def _find_series_folders(root_folder, types_file):
    if isinstance(types_file, str):
        types_file = [types_file]
    series_folders = set()
    for ext in types_file:
        for file_path in Path(root_folder).rglob(f"*.{ext}"):
            series_folders.add(file_path.parent)
    return list(series_folders)


def controller():
    txt_points_folder = os.path.join(data_path, "txt_points")
    txt_files = _find_series_folders(txt_points_folder, "txt")
    train_test_lists = read_csv(result_folder, "train_test_lists.csv")

    for txt_file_path in txt_files:


    # add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller_path = "C:/Users/Kamil/Aortic_valve/code_aortic_valve/controller.yaml"
    controller_dump = yaml_reader(controller_path)
    controller()