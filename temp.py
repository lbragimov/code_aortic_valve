import os
import shutil
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from subprocess import call
import torch

from data_preprocessing.text_worker import add_info_logging


class nnUnet_trainer:

    def __init__(self, nnUnet_path):
        # Set paths for nnUNet environment variables
        os.environ["nnUNet_raw"] = os.path.join(nnUnet_path, "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = os.path.join(nnUnet_path, "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = os.path.join(nnUnet_path, "nnUNet_results")
        os.environ["nnUNet_compile"] = "f"

    def preprocessing(self, task_id):
        command = [
            "nnUNetv2_plan_and_preprocess",
            '-d ' + str(task_id),
            "--verify_dataset_integrity",  # Optional: Save softmax predictions
            "-np", "1",
        ]

        # Execute preprocessing
        try:
            add_info_logging("Starting nnU-Net preprocessing")
            call(command)
            add_info_logging("Preprocessing completed successfully")
        except Exception as e:
            add_info_logging(f"An error occurred during preprocessing: {e}")

    def train(self, task_id, fold=0, network="3d_fullres"):

        print(torch.device(type='cuda', index=2))

        # Define the task ID and fold
        command = [
            # "CUDA_VISIBLE_DEVICES=2",
            "nnUNetv2_train",
            str(task_id),
            # trainer,
            network,
            # f"Task{task_id}",
            str(fold),
            "--npz",  # Optional: Save softmax predictions
            #"-device cpu"
            "-device", "cuda",
            "-num_gpus", "2"
        ]

        # Execute the training
        try:
            add_info_logging("Starting nnU-Net training")
            call(command)
            add_info_logging("Training completed successfully")
        except Exception as e:
            add_info_logging(f"An error occurred during training: {e}")

    def predicting(self, input_folder, output_folder, task_id, fold=0, network="3d_fullres"):
        # input_folder = os.path.join(nnUnet_path, "nnUNet_raw", "Dataset401_AorticValve", "imagesTs")
        # output_folder = os.path.join(nnUnet_path, "nnUNet_test", "Dataset401_AorticValve")

        command = [
            "nnUNetv2_predict",
            "-i" + input_folder,
            "-o" + output_folder,
            "-d" + str(task_id),
            "-c" + network,
            "-f" + str(fold),
        ]

        # Execute the predicting
        try:
            add_info_logging("Starting nnU-Net predict")
            call(command)
            add_info_logging("Predicting completed successfully")
        except Exception as e:
            add_info_logging(f"An error occurred during predicting: {e}")

    def evaluation(self, input_folder, output_folder, task_id, fold=0, network="3d_fullres"):
        # input_folder = os.path.join(nnUnet_path, "nnUNet_raw", "Dataset401_AorticValve", "imagesTs")
        # output_folder = os.path.join(nnUnet_path, "nnUNet_test", "Dataset401_AorticValve")

        command = [
            "nnUNetv2_evaluate_folder",
            "-i" + input_folder,
            "-o" + output_folder,
            "-d" + str(task_id),
            "-c" + network,
            "-f" + str(fold),
        ]

        # Execute the predicting
        try:
            add_info_logging("Starting nnU-Net evaluation")
            call(command)
            add_info_logging("evaluation completed successfully")
        except Exception as e:
            add_info_logging(f"An error occurred during evaluation: {e}")

        # nnUNetv2_evaluate_folder / nnUNet_tests / gt / / nnUNet_tests / predictions / -djfile
        # Dataset300_Aorta / nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres / dataset.json - pfile
        # Dataset300_Aorta / nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres / plans.json

    def reassembling_model(self, nnUnet_path, case_path):
        model_path = os.path.join(nnUnet_path, "nnUNet_results", case_path, "nnUNetTrainer__nnUNetPlans__3d_fullres",
                                  "fold_all", "checkpoint_final.pth")
        # Загружаем старый чекпоинт (весы + другие данные)
        checkpoint = torch.load(model_path, weights_only=False)  # Включаем загрузку всех данных

        # Извлекаем только state_dict (веса)
        model_weights = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        # Пересохраняем только веса
        torch.save(model_weights, model_path)


def _clear_folder(folder):
    """Очищает папку, удаляя все файлы и подпапки"""
    if not folder.exists():
        add_info_logging(f"Папка '{str(folder)}' не существует.")
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
                   folder_mask_path, dict_dataset, pct_test, test_folder=None, create_ds=False, activate_mod=False):

    folder = Path(folder)
    folder_image_path = Path(folder_image_path)
    folder_mask_path = Path(folder_mask_path)

    if create_ds:
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
                    n = 0
                    for case in (folder_image_path/subfolder).iterdir():
                        if int(file_count * (1.0 - pct_test)) >= n:
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

    if activate_mod:
        input_folder = Path(folder / "nnUNet_raw" / ds_folder_name / "imagesTs")
        output_folder = Path(folder / "nnUNet_test" / ds_folder_name)
        model_nnUnet = nnUnet_trainer(folder)
        model_nnUnet.preprocessing(task_id=id_case)
        model_nnUnet.train(task_id=id_case, fold="all")
        model_nnUnet.predicting(input_folder=input_folder,
                                output_folder=output_folder,
                                task_id=id_case, fold="all")


def controller(data_path):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    crop_nii_image_path = os.path.join(data_path, "crop_nii_image")
    crop_markers_mask_path = os.path.join(data_path, "crop_markers_mask")
    dict_dataset = {
        "channel_names": {0: "CT"},
        "labels": {
            "background": 0,
            "R": 1,
            "L": 2,
            "N": 3,
            "RLC": 4,
            "RNC": 5,
            "LNC": 6
        },
        "file_ending": ".nii.gz"
    }
    process_nnunet(folder=nnUNet_folder, ds_folder_name="Dataset403_AortaLandmarks", id_case=403,
                   folder_image_path=crop_nii_image_path, folder_mask_path=crop_markers_mask_path,
                   dict_dataset=dict_dataset, pct_test=0.15, test_folder="Homburg pathology",
                   create_ds=False, activate_mod=True)


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)