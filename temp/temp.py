from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
from data_preprocessing.text_worker import (json_reader, yaml_reader, read_csv, clear_folder,
                                            add_info_logging)


def _load_image(image_path):
    return sitk.ReadImage(image_path)


def _save_image(image, output_path):
    """Saves the image to a file."""
    sitk.WriteImage(image, output_path)


def _create_sphere_mask(shape, center, radius):#, spacing):
    """
    Creates a sphere at the given point with the given radius.
    :param shape: Mask dimensions (voxels).
    :param center: Center coordinates (voxels).
    :param radius: Sphere radius (in millimeters).
    :param spacing: Voxel dimensions (image spacing).
    :return: SimpleITK image with mask.
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]  # Z, Y, X
    distance = np.sqrt((z - center[2]) ** 2 + (y - center[1]) ** 2 + (x - center[0]) ** 2)
    sphere = distance <= radius
    return sphere.astype(np.uint8)


def _world_to_voxel(coord, image):
    """
    Converts physical coordinates (world) to voxel indices.
    :param coord: Coordinates in physical units (millimeters).
    :param image: SimpleITK image object.
    :return: Coordinates in voxel indices.
    """
    origin = np.array(image.GetOrigin())  # Точка отсчёта
    spacing = np.array(image.GetSpacing())  # Размер вокселя
    direction = np.array(image.GetDirection()).reshape(3, 3)  # Ориентация

    # Преобразование
    voxel_coord = np.linalg.inv(direction).dot(np.array(coord) - origin) / spacing
    return np.round(voxel_coord).astype(int)


def process_markers(image_path, dict_case, output_path, radius, keys_to_need=None):
    """
    Processes one pair (image and coordinate table).
    Creates a mask and saves it.
    """
    image = _load_image(image_path)
    shape = image.GetSize()

    # Создаём пустую маску
    mask = np.zeros(shape[::-1], dtype=np.uint8)  # Меняем порядок: Z, Y, X

    # keys_to_need = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']
    # keys_to_need = {
    #     'R': 1, 'L': 2, 'N': 3,
    #     'RLC': 4, 'RNC': 5, 'LNC': 6
    # }
    # keys_to_need = {'GH': 1}
    for key, coord in dict_case.items():
        if key not in keys_to_need:
            continue
        voxel_coord = _world_to_voxel(coord, image)

        # Проверка на выход за границы
        if not all(0 <= voxel_coord[d] < shape[d] for d in range(3)):
            add_info_logging(f"Point {coord} (voxel coordinates {voxel_coord}) is outside the image volume.",
                             "work_logger")
            continue

        # Создаём сферу и добавляем её в маску
        sphere = _create_sphere_mask(mask.shape, voxel_coord, radius)
        # Заполняем маску уникальным числом
        mask[sphere > 0] = keys_to_need[key]
        # mask = np.maximum(mask, sphere)

    # Преобразуем маску в SimpleITK-объект
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetSpacing(image.GetSpacing())
    mask_image.SetOrigin(image.GetOrigin())
    mask_image.SetDirection(image.GetDirection())

    # Сохраняем маску
    _save_image(mask_image, output_path)


def controller():
    result_folder = os.path.join(data_path, "result")
    train_test_lists = read_csv(result_folder, "train_test_lists.csv")
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")
    dict_all_case = json_reader(dict_all_case_path)
    image_crop_folder = os.path.join(data_path, "image_nii_crop")
    mask_6_landmarks_folder = os.path.join(data_path, "mask_6_landmarks")

    clear_folder(os.path.join(mask_6_landmarks_folder))
    for case_name, points_dict in dict_all_case.items():
        process_markers(image_path=os.path.join(image_crop_folder, f"{case_name}.nii.gz"),
                        dict_case=points_dict,
                        output_path=os.path.join(mask_6_landmarks_folder, f"{case_name}.nii.gz"),
                        radius=9, keys_to_need={'R': 1, 'L': 2, 'N': 3, 'RLC': 4, 'RNC': 5, 'LNC': 6})

    # add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller_path = "C:/Users/Kamil/Aortic_valve/code_aortic_valve/controller.yaml"
    controller_dump = yaml_reader(controller_path)
    controller()