import os
import numpy as np
from pathlib import Path
from data_postprocessing.evaluation_analysis import landmarking_testing, landmarking_MonteCarlo
from data_preprocessing.text_worker import add_info_logging
from models.controller_nnUnet import process_nnunet


def experiment(data_path):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    # list_radius = [10, 9, 8, 7, 6, 5, 4]
    dict_id_case = {10: 491, 9: 499, 8: 498, 7: 497, 6: 496, 5: 495, 4: 494}
    list_radius = [10, 9, 8, 7, 6]
    # dict_id_case = {10: 481, 9: 489, 8: 488, 7: 487, 6: 486}
    list_result_type = ["binary_map", "probability_map"]
    for radius in list_radius:
        ds_folder_name = f"Dataset{dict_id_case[radius]}_AortaLandmarks"
        for type_map in list_result_type:
            if type_map == "probability_map":
                save_probabilities = True
            else:
                save_probabilities = False
            process_nnunet(folder=nnUNet_folder, ds_folder_name=ds_folder_name,
                           id_case=dict_id_case[radius], folder_image_path=None, folder_mask_path=None, dict_dataset={},
                           pct_test=None, testing_mod=True, save_probabilities=save_probabilities)
            add_info_logging(f"radius: {radius}, type predicting: {type_map}")
            process_analysis(data_path, ds_folder_name, find_center_mass=True, probabilities_map=save_probabilities)
            # data_path = Path(data_path)
            # result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
            # original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
            # json_path = data_path / "nnUNet_folder" / "json_info"


def process_analysis(data_path, ds_folder_name, find_center_mass=False, find_monte_carlo=False, probabilities_map=False):
    # add_info_logging("start analysis")
    # print("start analysis")
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    json_path = data_path / "nnUNet_folder" / "json_info"

    if find_center_mass:
        if probabilities_map:
            files = list(result_landmarks_folder.glob("*.npz"))
        else:
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
        r_errors = []
        l_errors = []
        n_errors = []
        rlc_errors = []
        rnc_errors = []
        lnc_errors = []
        for file in files:
            if file.name[0] == "H":
                res_test = landmarking_testing()
                if probabilities_map:
                    file_name = file.name[:-4] + ".nii.gz"
                    landmarks_pred = res_test.compute_metrics_direct_npz(original_mask_folder / file_name, file)
                    landmarks_true = res_test.compute_metrics_direct_nii(original_mask_folder / file_name)
                else:
                    landmarks_pred = res_test.compute_metrics_direct_nii(file)
                    landmarks_true = res_test.compute_metrics_direct_nii(original_mask_folder / file.name)
                num_img_ger_pat += 1
                # Вычисляем среднюю ошибку
                for key in landmarks_true:
                    if key in landmarks_pred:
                        dist = np.linalg.norm(landmarks_true[key] - landmarks_pred[key])  # Евклидово расстояние
                        errors_ger_pat.append(dist)
                        if key == 1:
                            r_errors.append(dist)
                        elif key == 2:
                            l_errors.append(dist)
                        elif key == 3:
                            n_errors.append(dist)
                        elif key == 4:
                            rlc_errors.append(dist)
                        elif key == 5:
                            rnc_errors.append(dist)
                        elif key == 6:
                            lnc_errors.append(dist)
                    else:
                        not_found_ger_pat += 1
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}")
            if file.name[0] == "p":
                # if file.name[1] == "9":
                #     continue
                res_test = landmarking_testing()
                if probabilities_map:
                    file_name = file.name[:-4] + ".nii.gz"
                    landmarks_pred = res_test.compute_metrics_direct_npz(original_mask_folder / file_name, file)
                    landmarks_true = res_test.compute_metrics_direct_nii(original_mask_folder / file_name)
                else:
                    landmarks_pred = res_test.compute_metrics_direct_nii(file)
                    landmarks_true = res_test.compute_metrics_direct_nii(original_mask_folder / file.name)
                num_img_slo_pat += 1
                # Вычисляем среднюю ошибку
                for key in landmarks_true:
                    if key in landmarks_pred:
                        dist = np.linalg.norm(landmarks_true[key] - landmarks_pred[key])  # Евклидово расстояние
                        errors_slo_pat.append(dist)
                    if key == 1:
                        r_errors.append(dist)
                    elif key == 2:
                        l_errors.append(dist)
                    elif key == 3:
                        n_errors.append(dist)
                    elif key == 4:
                        rlc_errors.append(dist)
                    elif key == 5:
                        rnc_errors.append(dist)
                    elif key == 6:
                        lnc_errors.append(dist)
                    else:
                        not_found_slo_pat += 1
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}")
            if file.name[0] == "n":
                res_test = landmarking_testing()
                if probabilities_map:
                    file_name = file.name[:-4] + ".nii.gz"
                    landmarks_pred = res_test.compute_metrics_direct_npz(original_mask_folder / file_name, file)
                    landmarks_true = res_test.compute_metrics_direct_nii(original_mask_folder / file_name)
                else:
                    landmarks_pred = res_test.compute_metrics_direct_nii(file)
                    landmarks_true = res_test.compute_metrics_direct_nii(original_mask_folder / file.name)
                num_img_slo_norm += 1
                # Вычисляем среднюю ошибку
                for key in landmarks_true:
                    if key in landmarks_pred:
                        dist = np.linalg.norm(landmarks_true[key] - landmarks_pred[key])  # Евклидово расстояние
                        errors_slo_norm.append(dist)
                        if key == 1:
                            r_errors.append(dist)
                        elif key == 2:
                            l_errors.append(dist)
                        elif key == 3:
                            n_errors.append(dist)
                        elif key == 4:
                            rlc_errors.append(dist)
                        elif key == 5:
                            rnc_errors.append(dist)
                        elif key == 6:
                            lnc_errors.append(dist)
                    else:
                        not_found_slo_norm += 1
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}")

        mean_error_ger_pat = np.mean(errors_ger_pat) if errors_ger_pat else None
        not_found_ger_pat = (not_found_ger_pat / (num_img_ger_pat * 6)) * 100

        mean_error_slo_pat = np.mean(errors_slo_pat) if errors_slo_pat else None
        not_found_slo_pat = (not_found_slo_pat / (num_img_slo_pat * 6)) * 100

        mean_error_slo_norm = np.mean(errors_slo_norm) if errors_slo_norm else None
        not_found_slo_norm = (not_found_slo_norm / (num_img_slo_norm * 6)) * 100

        mean_error = np.mean(np.concatenate([errors_ger_pat, errors_slo_pat, errors_slo_norm]))
        num_img = num_img_ger_pat + num_img_slo_pat + num_img_slo_norm
        not_found = ((not_found_ger_pat + not_found_slo_pat + not_found_slo_norm) / (num_img * 6)) * 100
        # print("finish analysis")
        mean_r_error = np.mean(r_errors) if r_errors else None
        mean_l_error = np.mean(l_errors) if l_errors else None
        mean_n_error = np.mean(n_errors) if n_errors else None
        mean_rlc_error = np.mean(rlc_errors) if rlc_errors else None
        mean_rnc_error = np.mean(rnc_errors) if rnc_errors else None
        mean_lnc_error = np.mean(lnc_errors) if lnc_errors else None

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
        add_info_logging(f"Mean Euclidean Distance 'R' point: {mean_r_error}")
        add_info_logging(f"Mean Euclidean Distance 'L' point: {mean_l_error}")
        add_info_logging(f"Mean Euclidean Distance 'N' point: {mean_n_error}")
        add_info_logging(f"Mean Euclidean Distance 'RLC' point: {mean_rlc_error}")
        add_info_logging(f"Mean Euclidean Distance 'RNC' point: {mean_rnc_error}")
        add_info_logging(f"Mean Euclidean Distance 'LNC' point: {mean_lnc_error}")

    if find_monte_carlo:
        arr_mean_angles_ger_pat = np.array([]).reshape(0, 3)
        arr_mean_dists_ger_pat = np.array([]).reshape(0, 3)
        arr_mean_angles_slo_pat = np.array([]).reshape(0, 3)
        arr_mean_dists_slo_pat = np.array([]).reshape(0, 3)
        arr_mean_angles_slo_norm = np.array([]).reshape(0, 3)
        arr_mean_dists_slo_norm = np.array([]).reshape(0, 3)
        files = list(result_landmarks_folder.glob("*.npz"))
        for file in files:
            if file.name[0] == "H":
                simulation = landmarking_MonteCarlo(json_file=str(json_path/file.name[:-4]) + ".json",
                                                    nii_file=str(result_landmarks_folder/file.name[:-4]) + ".nii.gz" ,
                                                    npy_file=str(file))
                cur_angles, cur_dists =  simulation.run_simulation()
                arr_mean_angles_ger_pat = np.vstack([arr_mean_angles_ger_pat, cur_angles])
                arr_mean_dists_ger_pat = np.vstack([arr_mean_dists_ger_pat, cur_dists])
            if file.name[0] == "p":
                simulation = landmarking_MonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
                                                    nii_file=str(result_landmarks_folder / file.name[:-4]) + ".nii.gz",
                                                    npy_file=str(file))
                cur_angles, cur_dists = simulation.run_simulation()
                arr_mean_angles_slo_pat = np.vstack([arr_mean_angles_slo_pat, cur_angles])
                arr_mean_dists_slo_pat = np.vstack([arr_mean_dists_slo_pat, cur_dists])
            if file.name[0] == "n":
                simulation = landmarking_MonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
                                                    nii_file=str(result_landmarks_folder / file.name[:-4]) + ".nii.gz",
                                                    npy_file=str(file))
                cur_angles, cur_dists = simulation.run_simulation()
                arr_mean_angles_slo_norm = np.vstack([arr_mean_angles_slo_norm, cur_angles])
                arr_mean_dists_slo_norm = np.vstack([arr_mean_dists_slo_norm, cur_dists])
        add_info_logging("German pathology")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_ger_pat, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_ger_pat, axis=0)}'")
        add_info_logging("Slovenian pathology")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_slo_pat, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_slo_pat, axis=0)}'")
        add_info_logging("Slovenian normal")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_slo_norm, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_slo_norm, axis=0)}'")
        add_info_logging("Sum")
        arr_mean_angles = np.vstack([arr_mean_angles_ger_pat, arr_mean_angles_slo_pat, arr_mean_angles_slo_norm])
        arr_mean_dists = np.vstack([arr_mean_dists_ger_pat, arr_mean_dists_slo_pat, arr_mean_dists_slo_norm])
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists, axis=0)}'")
