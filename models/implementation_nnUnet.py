import os
from subprocess import call
import logging
# import platform
import torch

from datetime import datetime

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
