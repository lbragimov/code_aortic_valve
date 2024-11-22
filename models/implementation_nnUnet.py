

# Пути
# dataset_name = "Dataset001_MyTask"
# raw_data_dir = "/path/to/nnUNet_raw/Dataset001_MyTask"
# preprocessed_dir = "/path/to/nnUNet_preprocessed"

# Запуск предобработки
# preprocess_dataset(dataset_name, raw_data_dir, preprocessed_dir)

# Название задачи
dataset_name = "Dataset001_MyTask"

import os
from subprocess import call
import logging

from datetime import datetime


class nnUnet_trainer:

    def __init__(self, nnUnet_path):
        # Set paths for nnUNet environment variables
        os.environ["nnUNet_raw"] = nnUnet_path + r'\nnUNet_raw'
        os.environ["nnUNet_preprocessed"] = nnUnet_path + r'\nnUNet_preprocessed'
        os.environ["nnUNet_results"] = nnUnet_path + r'\nnUNet_results'
        pass

    def train_nnUnet(self, task_id, fold=0, network="3d_fullres"):
        # Define the task ID and fold

        # command = [
        #     "nnUNetv2_plan_and_preprocess",
        #     '-d ' + str(task_id),
        #     "--verify_dataset_integrity",  # Optional: Save softmax predictions
        #     # "-np " + str(1),
        # ]
        #
        # # Execute preprocessing
        # try:
        #     current_time = datetime.now()
        #     str_time = current_time.strftime("%d:%H:%M")
        #     logging.info(f"time: {str_time}")
        #     print("Starting nnU-Net preprocessing...")
        #     call(command)
        #     print("Preprocessing completed successfully.")
        # except Exception as e:
        #     print(f"An error occurred during preprocessing: {e}")
        # pass

        # subprocess.run([
        #     "nnUNetv2_plan_and_preprocess",
        #     "--dataset_name", dataset_name,# task_id
        #     "--verify_integrity"
        # ])

        # Set the configuration for the training
        # trainer = "nnUNetTrainerV2"  # Replace if using a custom trainer

        command = [
            "nnUNetv2_train",
            str(task_id),
            # trainer,
            network,
            # f"Task{task_id}",
            str(fold),
            # "--npz",  # Optional: Save softmax predictions
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


        command = [
            "nnUNetv2_predict",
            "-i" + "C:/Users/Kamil/Aortic_valve/data/nnUNet_folder/nnUNet_raw/Dataset401_AorticValve/imagesTs/",
            "-o" + "C:/Users/Kamil/Aortic_valve/data/nnUNet_folder/nnUNet_test/",
            "-d" + str(task_id),
            "-c" + network,
            "-f" + str(fold),
        ]

        # Execute the training
        try:
            current_time = datetime.now()
            str_time = current_time.strftime("%d:%H:%M")
            logging.info(f"time: {str_time}")
            print("Starting nnU-Net predict...")
            call(command)
            print("Predicting completed successfully.")
        except Exception as e:
            print(f"An error occurred during predicting: {e}")

        # # Конфигурация
        # config_name = "3d_fullres"
        # trainer_name = "nnUNetTrainer"
        # plans_identifier = "nnUNetPlans"
        # fold = 0
        # dataset_name = "Dataset001_MyTask"
        #
        # # Инициализация тренера
        # trainer = nnUNetTrainer(config_name=config_name,
        #                         trainer_name=trainer_name,
        #                         plans_identifier=plans_identifier,
        #                         dataset_name=dataset_name,
        #                         fold=fold,
        #                         results_folder=nnUNet_results)
        #
        # # Обучение
        # trainer.run_training()

        # # Параметры
        # config_name = "3d_fullres"
        # trainer_name = "nnUNetTrainer"
        # plans_identifier = "nnUNetPlans"
        # fold = 0
        # input_folder = "/path/to/test/images"
        # output_folder = "/path/to/output"
        # dataset_name = "Dataset001_MyTask"
        #
        # # Запуск предсказания
        # predict_from_raw_data(
        #     input_folder=input_folder,
        #     output_folder=output_folder,
        #     config_name=config_name,
        #     trainer_name=trainer_name,
        #     plans_identifier=plans_identifier,
        #     dataset_name=dataset_name,
        #     fold=fold,
        # )
