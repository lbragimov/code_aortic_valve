import os
from subprocess import call
import logging
# import platform

from datetime import datetime


class nnUnet_trainer:

    def __init__(self, nnUnet_path):
        # Set paths for nnUNet environment variables
        os.environ["nnUNet_raw"] = nnUnet_path + '/nnUNet_raw'
        os.environ["nnUNet_preprocessed"] = nnUnet_path + '/nnUNet_preprocessed'
        os.environ["nnUNet_results"] = nnUnet_path + '/nnUNet_results'
        pass

    def train_nnUnet(self, task_id, nnUnet_path, fold=0, network="3d_fullres"):
        # Define the task ID and fold

        command = [
            "nnUNetv2_plan_and_preprocess",
            '-d ' + str(task_id),
            "--verify_dataset_integrity",  # Optional: Save softmax predictions
            # "-np " + str(1),
        ]

        # Execute preprocessing
        try:
            current_time = datetime.now()
            str_time = current_time.strftime("%d:%H:%M")
            logging.info(f"time: {str_time}")
            print("Starting nnU-Net preprocessing...")
            call(command)
            print("Preprocessing completed successfully.")
        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")

        command = [
            "nnUNetv2_train",
            str(task_id),
            # trainer,
            network,
            # f"Task{task_id}",
            str(fold),
            # "--npz",  # Optional: Save softmax predictions
            #"-device cpu"
            "--device cuda:2"
        ]

        # Execute the training
        try:
            current_time = datetime.now()
            str_time = current_time.strftime("%d:%H:%M")
            logging.info(f"time: {str_time}")
            print("Starting nnU-Net training...")
            call(command)
            print("Training completed successfully.")
        except Exception as e:
            print(f"An error occurred during training: {e}")

        # input_folder = nnUnet_path + "/nnUNet_raw/Dataset401_AorticValve/imagesTs/"
        # output_folder = nnUnet_path + "/nnUNet_test/"
        #
        # command = [
        #     "nnUNetv2_predict",
        #     "-i" + input_folder,
        #     "-o" + output_folder,
        #     "-d" + str(task_id),
        #     "-c" + network,
        #     "-f" + str(fold),
        # ]
        #
        # # Execute the predicting
        # try:
        #     current_time = datetime.now()
        #     str_time = current_time.strftime("%d:%H:%M")
        #     logging.info(f"time: {str_time}")
        #     print("Starting nnU-Net predict...")
        #     call(command)
        #     print("Predicting completed successfully.")
        # except Exception as e:
        #     print(f"An error occurred during predicting: {e}")

