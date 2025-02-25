import numpy as np
from pathlib import Path
from data_postprocessing.evaluation_analysis import landmarking_testing, landmarking_MonteCarlo
from data_preprocessing.text_worker import add_info_logging


def process_analysis(data_path, ds_folder_name):
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    json_path = data_path / "nnUNet_folder" / "json_info"
    files = list(result_landmarks_folder.glob("*.nii.gz"))
    errors = []
    not_found = 0
    num_img = 0
    for file in files:
        if not file.name[0] == "H":
            continue
        # for file in (result_landmarks_folder/sub_dir).iterdir():
        # for case in os.listdir(os.path.join(crop_nii_image_path, sub_dir)):
        simulation = landmarking_MonteCarlo(str(json_path/file.name[:-7]) + ".json",
                                            file)
        simulation.run_simulation()
        res_test = landmarking_testing()
        landmarks_pred = res_test.compute_metrics_direct(file)
        landmarks_true = res_test.compute_metrics_direct(original_mask_folder / file.name)
        num_img += 1
        # Вычисляем среднюю ошибку
        for key in landmarks_true:
            if key in landmarks_pred:
                dist = np.linalg.norm(landmarks_true[key] - landmarks_pred[key])  # Евклидово расстояние
                errors.append(dist)
            else:
                not_found += 1

    mean_error = np.mean(errors) if errors else None
    not_found = (not_found / (num_img * 6)) * 100

    add_info_logging(
        f"Mean Euclidean Distance: {mean_error:.4f} mm, not found: {not_found: .2f}%. Number of images:{num_img}")
