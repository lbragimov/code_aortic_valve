import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import json
import pandas as pd


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
    """Извлекает параметры изображения из .nii/.nii.gz файла"""
    image = sitk.ReadImage(str(nii_path))
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    info = {
        "filename": nii_path.name,
        "shape": tuple(reversed(size)),       # (Z, Y, X)
        "spacing": spacing,                   # (X, Y, Z)
        "origin": origin,                     # (X, Y, Z)
        "direction": image.GetDirection(),    # 3x3 flattened
        "pixel_type": sitk.GetPixelIDValueAsString(image.GetPixelID())
    }
    return info


def check_files_in_folder(check_folder, result_folder):
    nii_files = list(check_folder.rglob("*.nii")) + list(check_folder.rglob("*.nii.gz"))

    info_list = []

    for nii_file in nii_files:
        info = _extract_image_info(nii_file)
        info_list.append(info)

    df = pd.DataFrame(info_list)
    csv_path = result_folder / f"{check_folder.name}_image_info.csv"
    df.to_csv(csv_path, index=False)

    # 🔍 Собираем уникальные значения по каждому столбцу
    unique_info = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in df.columns
        if col not in ['filename']
    }

    # Преобразуем к JSON-сериализуемым типам
    def convert(o):
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        return o

    json_path = result_folder / f"{check_folder.name}_image_info_summary.json"
    with open(json_path, 'w') as f:
        json.dump(unique_info, f, indent=4, default=convert)


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    check_folder = Path(data_path) / "nii_resample"
    result_folder = Path(data_path) / "temp" / "temp_check_info_nii"
    # min_max_value(check_folder)
    check_files_in_folder(check_folder, result_folder)