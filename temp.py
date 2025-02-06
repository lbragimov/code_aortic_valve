import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from data_preprocessing.text_worker import json_reader


def find_global_size(image_paths, padding=16):
    """
    Находит общий size box для всех изображений.
    :param image_paths: Список путей к изображениям в формате NIfTI (.nii)
    :param padding: Дополнительный запас вокселей вокруг найденной маски
    :return: [(z_min, z_max), (y_min, y_max), (x_min, x_max)]
    """
    global_size_z, global_size_y, global_size_x = 0, 0, 0

    for image_path in image_paths:
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)

        nonzero_indices = np.argwhere(image_array > 0)  # Координаты всех ненулевых пикселей

        if nonzero_indices.size == 0:
            continue  # Если маска пустая, пропускаем

        # Определяем min/max координаты
        z_min, y_min, x_min = nonzero_indices.min(axis=0)
        z_max, y_max, x_max = nonzero_indices.max(axis=0)

        global_size_z = max(global_size_z, z_max - z_min)
        global_size_y = max(global_size_y, y_max - y_min)
        global_size_x = max(global_size_x, x_max - x_min)

    global_size_z = global_size_z + (2*padding)
    global_size_y = global_size_y + (2*padding)
    global_size_x = global_size_x + (2*padding)

    return [global_size_z, global_size_y, global_size_x]


def controller(data_path):
    script_dir = Path(__file__).resolve().parent
    data_structure_path = os.path.join(script_dir, "dir_structure.json")
    mask_aorta_segment_cut_path = os.path.join(data_path, "mask_aorta_segment_cut")
    dir_structure = json_reader(data_structure_path)
    all_image_paths = []
    for sub_dir in dir_structure["mask_aorta_segment_cut"]:
        for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
            image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
            all_image_paths.append(image_path)

    padding = 10
    # Найти общий bounding box для всех изображений
    global_size = find_global_size(all_image_paths, padding)


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)