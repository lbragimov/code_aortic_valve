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