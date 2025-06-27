import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np

def check_dataset(images_folder, masks_folder):
    images_folder = Path(images_folder)
    masks_folder = Path(masks_folder)

    image_files = sorted(images_folder.glob("*.nii*"))
    mask_files = sorted(masks_folder.glob("*.nii*"))

    if len(image_files) != len(mask_files):
        print(f"⚠️ Количество изображений и масок не совпадает! {len(image_files)} изображений, {len(mask_files)} масок.")
        return

    for img_path, mask_path in zip(image_files, mask_files):

        img = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))

        img_shape = img.GetSize()
        mask_shape = mask.GetSize()

        if img_shape != mask_shape:
            print(f"\nПроверка пары:\n  Изображение: {img_path.name}\n  Маска: {mask_path.name}")
            print(f"❌ Размеры не совпадают! Изображение: {img_shape}, Маска: {mask_shape}")

        mask_array = sitk.GetArrayFromImage(mask)
        # unique_vals = np.unique(mask_array)

        if np.all(mask_array == 0):
            print(f"\nПроверка пары:\n  Изображение: {img_path.name}\n  Маска: {mask_path.name}")
            print(f"⚠️ Маска полностью пустая (все значения = 0).")
        # else:
        #     print(f"✅ Размеры совпадают. Маска содержит классы: {unique_vals}")

if __name__ == "__main__":
    images_dir = r"C:\Users\Kamil\Aortic_valve\data\nnUNet_folder\nnUNet_raw\Dataset479_GeometricHeight\imagesTr"
    masks_dir = r"C:\Users\Kamil\Aortic_valve\data\nnUNet_folder\nnUNet_raw\Dataset479_GeometricHeight\labelsTr"

    check_dataset(images_dir, masks_dir)