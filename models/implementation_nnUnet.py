import os
from subprocess import call
import logging
# import platform
import torch

from datetime import datetime


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
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"Starting nnU-Net preprocessing: {str_time}")
                call(command)
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"Preprocessing completed successfully: {str_time}")
            except Exception as e:
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"An error occurred during preprocessing: {str_time}: {e}")

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
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"Starting nnU-Net training: {str_time}")
                call(command)
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"Training completed successfully: {str_time}")
            except Exception as e:
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"An error occurred during training: {str_time}: {e}")

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
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"Starting nnU-Net predict: {str_time}")
                call(command)
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"Predicting completed successfully: {str_time}")
            except Exception as e:
                current_time = datetime.now()
                str_time = current_time.strftime("%d:%H:%M")
                logging.info(f"An error occurred during predicting: {str_time}: {e}")

