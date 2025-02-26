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
    errors_ger_pat = []
    not_found_ger_pat = 0
    num_img_ger_pat = 0
    errors_slo_pat = []
    not_found_slo_pat = 0
    num_img_slo_pat = 0
    errors_slo_norm = []
    not_found_slo_norm = 0
    num_img_slo_norm = 0
    for file in files:
        if file.name[0] == "H":
            res_test = landmarking_testing()
            landmarks_pred = res_test.compute_metrics_direct(file)
            landmarks_true = res_test.compute_metrics_direct(original_mask_folder / file.name)
            num_img_ger_pat += 1
            # Вычисляем среднюю ошибку
            for key in landmarks_true:
                if key in landmarks_pred:
                    dist = np.linalg.norm(landmarks_true[key] - landmarks_pred[key])  # Евклидово расстояние
                    errors_ger_pat.append(dist)
                else:
                    not_found_ger_pat += 1
        if file.name[0] == "p":
            res_test = landmarking_testing()
            landmarks_pred = res_test.compute_metrics_direct(file)
            landmarks_true = res_test.compute_metrics_direct(original_mask_folder / file.name)
            num_img_slo_pat += 1
            # Вычисляем среднюю ошибку
            for key in landmarks_true:
                if key in landmarks_pred:
                    dist = np.linalg.norm(landmarks_true[key] - landmarks_pred[key])  # Евклидово расстояние
                    errors_slo_pat.append(dist)
                else:
                    not_found_slo_pat += 1
        if file.name[0] == "n":
            res_test = landmarking_testing()
            landmarks_pred = res_test.compute_metrics_direct(file)
            landmarks_true = res_test.compute_metrics_direct(original_mask_folder / file.name)
            num_img_slo_norm += 1
            # Вычисляем среднюю ошибку
            for key in landmarks_true:
                if key in landmarks_pred:
                    dist = np.linalg.norm(landmarks_true[key] - landmarks_pred[key])  # Евклидово расстояние
                    errors_slo_norm.append(dist)
                else:
                    not_found_slo_norm += 1
        # for file in (result_landmarks_folder/sub_dir).iterdir():
        # for case in os.listdir(os.path.join(crop_nii_image_path, sub_dir)):
        # simulation = landmarking_MonteCarlo(str(json_path/file.name[:-7]) + ".json", file)
        # simulation.run_simulation()


    mean_error_ger_pat = np.mean(errors_ger_pat) if errors_ger_pat else None
    not_found_ger_pat = (not_found_ger_pat / (num_img_ger_pat * 6)) * 100

    mean_error_slo_pat = np.mean(errors_slo_pat) if errors_slo_pat else None
    not_found_slo_pat = (not_found_slo_pat / (num_img_slo_pat * 6)) * 100

    mean_error_slo_norm = np.mean(errors_slo_norm) if errors_slo_norm else None
    not_found_slo_norm = (not_found_slo_norm / (num_img_slo_norm * 6)) * 100

    mean_error = np.mean(np.concatenate([errors_ger_pat, errors_slo_pat, errors_slo_norm]))
    num_img = num_img_ger_pat + not_found_slo_pat + num_img_slo_norm
    not_found = ((not_found_ger_pat + not_found_slo_pat + not_found_slo_norm) / (num_img * 6)) * 100

    add_info_logging("German pathology")
    add_info_logging(
        f"Mean Euclidean Distance: {mean_error_ger_pat:.4f} mm, not found: {not_found_ger_pat: .2f}%. Number of images:{num_img_ger_pat}")
    add_info_logging("Slovenian pathology")
    add_info_logging(
        f"Mean Euclidean Distance: {mean_error_slo_pat:.4f} mm, not found: {not_found_slo_pat: .2f}%. Number of images:{num_img_slo_pat}")
    add_info_logging("Slovenian normal")
    add_info_logging(
        f"Mean Euclidean Distance: {mean_error_slo_norm:.4f} mm, not found: {not_found_slo_norm: .2f}%. Number of images:{num_img_slo_norm}")
    add_info_logging("Sum")
    add_info_logging(
        f"Mean Euclidean Distance: {mean_error:.4f} mm, not found: {not_found: .2f}%. Number of images:{num_img}")
