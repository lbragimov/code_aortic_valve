import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk


def controller(folder_path):
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


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    check_folder = Path(data_path) /"nii_resample"/"Normal"
    controller(check_folder)