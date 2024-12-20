import os
from subprocess import call
import logging
# import platform
import torch

from datetime import datetime

from data_preprocessing.log_worker import add_info_logging


class nnUnet_trainer:

    def __init__(self, nnUnet_path):
        # Set paths for nnUNet environment variables
        os.environ["nnUNet_raw"] = os.path.join(nnUnet_path, "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = os.path.join(nnUnet_path, "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = os.path.join(nnUnet_path, "nnUNet_results")
        os.environ["nnUNet_compile"] = "f"

    def train_nnUnet(self, task_id, nnUnet_path, fold=0, network="3d_fullres"):

        preprocessing = True
        training = True
        predicting = True
        evaluation = False
        print(torch.device(type='cuda', index=2))

        # Define the task ID and fold

        if preprocessing:
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

        if training:
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
                "-num_gpus", "0"
            ]

            # Execute the training
            try:
                add_info_logging("Starting nnU-Net training")
                call(command)
                add_info_logging("Training completed successfully")
            except Exception as e:
                add_info_logging(f"An error occurred during training: {e}")

        if predicting:
            input_folder = os.path.join(nnUnet_path, "nnUNet_raw", "Dataset401_AorticValve", "imagesTs")
            output_folder = os.path.join(nnUnet_path, "nnUNet_test", "Dataset401_AorticValve")

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

        if evaluation:
            input_folder = os.path.join(nnUnet_path, "nnUNet_raw", "Dataset401_AorticValve", "imagesTs")
            output_folder = os.path.join(nnUnet_path, "nnUNet_test", "Dataset401_AorticValve")

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
                add_info_logging("Starting nnU-Net predict")
                call(command)
                add_info_logging("Predicting completed successfully")
            except Exception as e:
                add_info_logging(f"An error occurred during predicting: {e}")

        # nnUNetv2_evaluate_folder / nnUNet_tests / gt / / nnUNet_tests / predictions / -djfile
        # Dataset300_Aorta / nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres / dataset.json - pfile
        # Dataset300_Aorta / nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres / plans.json


