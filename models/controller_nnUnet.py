import os
import shutil
from pathlib import Path

from data_preprocessing.text_worker import add_info_logging
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from models.implementation_nnUnet import nnUnet_trainer


def _clear_folder(folder):
    """Очищает папку, удаляя все файлы и подпапки"""
    if not folder.exists():
        add_info_logging(f"Папка '{str(folder)}' не существует.", "work_logger")
        return

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()  # Удаляем файл или символическую ссылку
        elif item.is_dir():
            shutil.rmtree(item)  # Удаляем папку рекурсивно


def _configure_folder(base_folder, ds_folder_name):
    subfolders = ["imagesTr", "imagesTs", "labelsTr"]

    (base_folder/"nnUNet_preprocessed").mkdir(exist_ok=True)
    (base_folder/"nnUNet_raw"/ds_folder_name).mkdir(parents=True, exist_ok=True)
    (base_folder/"nnUNet_results").mkdir(exist_ok=True)
    (base_folder/"nnUNet_test").mkdir(exist_ok=True)
    (base_folder/"original_mask"/ds_folder_name).mkdir(parents=True, exist_ok=True)

    # Создаем папки, если их нет
    for subfolder in subfolders:
        (base_folder/"nnUNet_raw"/ds_folder_name/subfolder).mkdir(exist_ok=True)


def _copy_img(input_imgs_path, output_folder, rename=False):
    _clear_folder(output_folder)
    for img_path in input_imgs_path:
        if rename:
            case_name = img_path.name[:-7]
            shutil.copy(img_path, output_folder/f"{case_name}_0000.nii.gz")
        else:
            shutil.copy(img_path, output_folder/img_path.name)


def process_nnunet(folder, ds_folder_name, id_case, folder_image_path,
                   folder_mask_path, dict_dataset, pct_test=0.15, num_test=None, test_folder=None, create_ds=False,
                   training_mod=False, testing_mod=False, save_probabilities=False):

    folder = Path(folder)

    if create_ds:
        folder_image_path = Path(folder_image_path)
        folder_mask_path = Path(folder_mask_path)
        _configure_folder(folder, ds_folder_name)

        list_train_case, list_test_case = [], []
        list_train_mask, list_test_mask = [], []
        for subfolder in folder_image_path.iterdir():
            if subfolder.is_dir():
                if subfolder.name == test_folder:
                    for case in (folder_image_path / subfolder).iterdir():
                        list_test_case.append(case)
                        list_test_mask.append(folder_mask_path / subfolder.name / case.name)
                else:
                    file_count = len([f for f in (folder_image_path/subfolder).iterdir()])
                    if not num_test:
                        limit_files = file_count - num_test
                    else:
                        limit_files = int(file_count * (1.0 - pct_test))
                    n = 0
                    for case in (folder_image_path/subfolder).iterdir():
                        if limit_files >= n:
                            list_train_case.append(case)
                            list_train_mask.append(folder_mask_path / subfolder.name / case.name)
                        else:
                            list_test_case.append(case)
                            list_test_mask.append(folder_mask_path / subfolder.name / case.name)
                        n += 1

        _copy_img(list_train_case, folder/ "nnUNet_raw" / ds_folder_name / "imagesTr", rename=True)
        _copy_img(list_train_mask, folder / "nnUNet_raw" / ds_folder_name / "labelsTr")
        _copy_img(list_test_case, folder/ "nnUNet_raw" / ds_folder_name / "imagesTs", rename=True)
        _copy_img(list_test_mask, folder / "original_mask" / ds_folder_name)

        generate_dataset_json(str(folder/ "nnUNet_raw" / ds_folder_name),
                              channel_names=dict_dataset["channel_names"],
                              labels=dict_dataset["labels"],
                              num_training_cases=len(list_train_case),
                              file_ending=dict_dataset["file_ending"])

    if training_mod:
        model_nnUnet = nnUnet_trainer(str(folder))
        model_nnUnet.preprocessing(task_id=id_case)
        model_nnUnet.train(task_id=id_case, fold="all")

    if testing_mod:
        input_folder = Path(folder / "nnUNet_raw" / ds_folder_name / "imagesTs")
        output_folder = Path(folder / "nnUNet_test" / ds_folder_name)
        model_nnUnet = nnUnet_trainer(str(folder))
        model_nnUnet.predicting(input_folder=str(input_folder),
                                output_folder=str(output_folder),
                                task_id=id_case, fold="all",
                                save_probabilities=save_probabilities)
