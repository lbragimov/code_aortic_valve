import os
from nnunet.inference.predict import predict_from_folder
from nnunet.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunetv2.inference.predict_from_raw_data import predict

# Обучение модели
def train_model():

    task_id = int(task_name.split("_")[0][4:])
    fold = 0  # Определите fold (0-4 для 5-fold cross-validation)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage = get_default_configuration(
        "3d_fullres", task_id, "nnUNetTrainerV2", fold
    )
    trainer = nnUNetTrainerV2(plans_file, fold, output_folder_name, dataset_directory, stage, batch_dice)
    trainer.initialize()
    trainer.run_training()

# Предсказание
def predict_images(input_folder, output_folder):
    model = "3d_fullres"
    folds = [0]  # Определите, какие folds использовать
    predict_from_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        model=model,
        folds=folds,
        save_npz=True,
        num_threads_preprocessing=4,
        num_threads_nifti_save=2,
    )

import subprocess
from nnunetv2.experiment_planning.utils import preprocess_dataset
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.paths import nnUNet_results
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data

# Пути
# dataset_name = "Dataset001_MyTask"
# raw_data_dir = "/path/to/nnUNet_raw/Dataset001_MyTask"
# preprocessed_dir = "/path/to/nnUNet_preprocessed"

# Запуск предобработки
preprocess_dataset(dataset_name, raw_data_dir, preprocessed_dir)

# Название задачи
dataset_name = "Dataset001_MyTask"

def preprocess_nnUnet():
    # Запуск предобработки
    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "--dataset_name", dataset_name,
        "--verify_integrity"
    ])


# Конфигурация
config_name = "3d_fullres"
trainer_name = "nnUNetTrainer"
plans_identifier = "nnUNetPlans"
fold = 0
dataset_name = "Dataset001_MyTask"

# Инициализация тренера
trainer = nnUNetTrainer(config_name=config_name,
                        trainer_name=trainer_name,
                        plans_identifier=plans_identifier,
                        dataset_name=dataset_name,
                        fold=fold,
                        results_folder=nnUNet_results)

# Обучение
trainer.run_training()

# Параметры
config_name = "3d_fullres"
trainer_name = "nnUNetTrainer"
plans_identifier = "nnUNetPlans"
fold = 0
input_folder = "/path/to/test/images"
output_folder = "/path/to/output"
dataset_name = "Dataset001_MyTask"

# Запуск предсказания
predict_from_raw_data(
    input_folder=input_folder,
    output_folder=output_folder,
    config_name=config_name,
    trainer_name=trainer_name,
    plans_identifier=plans_identifier,
    dataset_name=dataset_name,
    fold=fold,
)


import os
from subprocess import call


class nnUnet_trainer:

    def __init__(self, nnUnet_path):
        # Set paths for nnUNet environment variables
        os.environ["nnUNet_raw"] = nnUnet_path + '/nnUNet_raw'
        os.environ["nnUNet_preprocessed"] = nnUnet_path + '/nnUNet_preprocessed'
        os.environ["nnUNet_results"] = nnUnet_path + '/nnUNet_results'
        pass

    def train_nnUnet(self, task_id, fold=0, network="3d_fullres"):
        # Define the task ID and fold

        command = [
            "nnUNetv2_plan_and_preprocess",
            '-d ' + task_id,
            "--verify_dataset_integrity",  # Optional: Save softmax predictions
        ]

        # Execute preprocessing
        try:
            print("Starting nnU-Net preprocessing...")
            call(command)
            print("Preprocessing completed successfully.")
        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
        pass

        # Set the configuration for the training
        # trainer = "nnUNetTrainerV2"  # Replace if using a custom trainer

        command = [
            "nnUNetv2_train",
            task_id,
            # trainer,
            network,
            # f"Task{task_id}",
            str(fold),
            # "--npz",  # Optional: Save softmax predictions
        ]

        # Execute the training
        try:
            print("Starting nnU-Net training...")
            call(command)
            print("Training completed successfully.")
        except Exception as e:
            print(f"An error occurred during training: {e}")
        pass
