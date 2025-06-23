import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import json


def min_max_value(folder_path):
    min_val = 0.0
    max_val = 0.0

    files = list(folder_path.glob("*.nii.gz"))
    for file in files:

        # Загрузка изображения
        img = nib.load(file)

        # Получение numpy-массива с данными интенсивности
        data = img.get_fdata()

        # # Чтение изображения
        # image = sitk.ReadImage(file)
        #
        # # Преобразование в numpy-массив
        # data = sitk.GetArrayFromImage(image)  # (Z, Y, X)

        # Вычисление минимума и максимума
        if np.min(data) < min_val:
            min_val = np.min(data)
        if np.max(data) > max_val:
            max_val = np.max(data)

    print(f"Минимальное значение интенсивности: {min_val}")
    print(f"Максимальное значение интенсивности: {max_val}")


def _extract_image_info(nii_path):
    """Извлекает параметры изображения из .nii файла"""
    image = sitk.ReadImage(str(nii_path))
    info = {
        "shape": list(reversed(image.GetSize())),  # Z, Y, X
        "spacing": list(image.GetSpacing()),  # X, Y, Z
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection())
    }
    return info


def check_files_in_folder(folder_path):
    folder_path = Path(folder_path)
    nii_files = list(folder_path.glob("*.nii")) + list(folder_path.glob("*.nii.gz"))

    all_info = {}

    for nii_file in nii_files:
        info = _extract_image_info(nii_file)
        all_info[nii_file.name] = info

    output_path = folder_path / "image_info.json"
    with open(output_path, 'w') as f:
        json.dump(all_info, f, indent=4)


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    check_folder = Path(data_path) / "temp_check_info_nii"
    # min_max_value(check_folder)
    check_files_in_folder(check_folder)