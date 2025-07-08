import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List
from data_postprocessing.evaluation_analysis import evaluate_segmentation
from data_postprocessing.montecarlo import LandmarkingMonteCarlo
from data_postprocessing.mask_analysis import mask_comparison, LandmarkCentersCalculator
from data_postprocessing.plotting_graphs import summarize_and_plot, plot_group_comparison
from data_preprocessing.text_worker import add_info_logging
from models.controller_nnUnet import process_nnunet
from data_postprocessing.metrics_config import metric_to_landmarks
from data_postprocessing.geometric_metrics import controller_metrics


def mask_analysis(data_path, result_path, type_mask, folder_name):
    per_case_csv = os.path.join(result_path, "per_case_metrics.csv")

    group_label_map = {
        "all": "All",
        "H": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }
    metrics = ["Dice", "IoU", "HD", "ASSD"]

    if os.path.exists(per_case_csv):
        # Загружаем данные из файлов
        add_info_logging("Using cached metrics from CSV files", "work_logger")
        df = pd.read_csv(per_case_csv)
    else:
        # Пересчитываем метрики
        per_case_data = mask_comparison(data_path, type_mask=type_mask, folder_name=folder_name)

        # Сохраняем метрики по кейсам
        df = pd.DataFrame(per_case_data)
        df.to_csv(per_case_csv, index=False)

    for metric_name in metrics:
        data_for_plot = df[['group', metric_name]].dropna(how='any')
        assd_mean = df["ASSD"].mean(numeric_only=True)
        assd_median = df["ASSD"].median(numeric_only=True)
        assd_std = df["ASSD"].std(numeric_only=True)
        dice_mean = df["Dice"].mean(numeric_only=True)
        dice_median = df["Dice"].median(numeric_only=True)
        dice_std = df["Dice"].std(numeric_only=True)
        assd_mean_groupby = df.groupby("group")["ASSD"].mean(numeric_only=True)
        assd_median_groupby = df.groupby("group")["ASSD"].median(numeric_only=True)
        assd_std_groupby = df.groupby("group")["ASSD"].std(numeric_only=True)
        dice_mean_groupby = df.groupby("group")["Dice"].mean(numeric_only=True)
        dice_median_groupby = df.groupby("group")["Dice"].median(numeric_only=True)
        dice_std_groupby = df.groupby("group")["Dice"].std(numeric_only=True)
        plot_group_comparison('group', metric_name, group_label_map, data_for_plot,
                              os.path.join(result_path, "aorta_root_comparison"))
    add_info_logging("Analysis completed", "work_logger")


def experiment_analysis(data_path,
                        dict_case,
                        generate_result=False,
                        find_center_mass=False,
                        find_monte_carlo=False):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    list_result_type = ["binary_map", "probability_map"]
    for radius, case_name in dict_case.items():
        ds_folder_name = f"Dataset{case_name}_AortaLandmarks"
        for type_map in list_result_type:
            if type_map == "probability_map":
                save_probabilities = True
            else:
                save_probabilities = False
            if generate_result:
                process_nnunet(folder=nnUNet_folder, ds_folder_name=ds_folder_name,
                               id_case=case_name, folder_image_path=None, folder_mask_path=None, dict_dataset={},
                               pct_test=None, testing_mod=True, save_probabilities=save_probabilities)
                add_info_logging(f"radius: {radius}, type predicting: {type_map}", "result_logger")
            landmarks_analysis(data_path, ds_folder_name,
                             find_center_mass=find_center_mass,
                             find_monte_carlo=find_monte_carlo,
                             probabilities_map=save_probabilities)
            # data_path = Path(data_path)
            # result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
            # original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
            # json_path = data_path / "nnUNet_folder" / "json_info"


def landmarks_analysis(data_path, ds_folder_name,
                       find_center_mass=False,
                       find_monte_carlo=False,
                       probabilities_map=False,
                       type_set="six_landmarks"):

    def process_file(file, original_mask_folder, probabilities_map):
        res_test = LandmarkCentersCalculator()
        file_name = file.name[:-4] + ".nii.gz"
        if probabilities_map:
            pred = res_test.extract_landmarks_com_npz(original_mask_folder / file_name, file)
        else:
            pred = res_test.extract_landmarks_com_nii(file)
        true = res_test.extract_landmarks_com_nii(original_mask_folder / file_name)
        return true, pred

    def compute_errors(true, pred, error_list, r, l, n, rlc, rnc, lnc, results, file_name, group):
        not_found = 0
        for key in true:
            if key in pred:
                if file_name.endswith(".npz"):
                    cur_file_name = file_name[:-4]
                elif file_name.endswith(".nii.gz"):
                    cur_file_name = file_name[:-7]
                dist = np.linalg.norm(true[key] - pred[key]) # Евклидово расстояние
                error_list.append(dist)
                results.append({
                    "filename": cur_file_name,
                    "group": group,
                    "point_id": point_name[key],
                    "error": dist
                })
                if key == 1: r.append(dist)
                elif key == 2: l.append(dist)
                elif key == 3: n.append(dist)
                elif key == 4: rlc.append(dist)
                elif key == 5: rnc.append(dist)
                elif key == 6: lnc.append(dist)
            else:
                not_found += 1
        return not_found

    def compute_gh_errors(true, pred, error_list, results, file_name, group):
        not_found = 0
        for key in true:
            if key in pred:
                if file_name.endswith(".npz"):
                    cur_file_name = file_name[:-4]
                elif file_name.endswith(".nii.gz"):
                    cur_file_name = file_name[:-7]
                dist = np.linalg.norm(true[key] - pred[key]) # Евклидово расстояние
                error_list.append(dist)
                results.append({
                    "filename": cur_file_name,
                    "group": group,
                    "point_id": point_name[key],
                    "error": dist
                })
            else:
                not_found += 1
        return not_found

    # add_info_logging("start analysis", "work_logger")
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    json_path = data_path / "nnUNet_folder" / "json_info"
    result_folder_path = data_path / "result" / ds_folder_name
    result_folder_path.mkdir(parents=True, exist_ok=True)
    results_csv_path = result_folder_path / f"landmark_errors_{ds_folder_name}.csv"

    if type_set == "six_landmarks":
        point_name = {"all": "All", 1:"r", 2:"l", 3:"n", 4:"rlc", 5:"rnc", 6:"lnc"}
    elif type_set == "gh_landmark":
        point_name = {1: "gh"}
    type_label = {
        "all": "All",
        "H": "Ger. path.",
        "p": "Slo. path.",
        "n": "Slo. norm."
    }

    results = []  # список словарей

    if find_center_mass:
        if os.path.exists(results_csv_path):
            # Загружаем данные из файлов
            add_info_logging("Using cached metrics from CSV files", "work_logger")
            results_df = pd.read_csv(results_csv_path)
        else:
            if probabilities_map:
                files = list(result_landmarks_folder.glob("*.npz"))
            else:
                files = list(result_landmarks_folder.glob("*.nii.gz"))
            errors_ger_pat, errors_slo_pat, errors_slo_norm = [], [], []
            not_found_ger_pat, not_found_slo_pat, not_found_slo_norm = 0, 0, 0
            num_img_ger_pat, num_img_slo_pat, num_img_slo_norm = 0, 0, 0
            if type_set == "six_landmarks":
                r_errors, l_errors, n_errors = [], [], []
                rlc_errors, rnc_errors, lnc_errors = [], [], []
            for file in files:
                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                if len(landmarks_pred.keys()) < 5 and type_set == "six_landmarks":
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")
                elif len(landmarks_pred.keys()) < 1 and type_set == "gh_landmark":
                    add_info_logging(f"img: {file.name}, not found landmark", "result_logger")

                first_char = file.name[0]
                if first_char == "H":
                    if file.name[1] == "HOM_M23_H175_W68_YA":
                        continue
                    num_img_ger_pat += 1
                    if type_set == "six_landmarks":
                        not_found_ger_pat += compute_errors(landmarks_true, landmarks_pred, errors_ger_pat,
                                                            r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                            results, file.name, "H")
                    elif type_set == "gh_landmark":
                        not_found_ger_pat += compute_gh_errors(landmarks_true, landmarks_pred, errors_ger_pat,
                                                               results, file.name, "H")
                elif first_char == "p":
                    if file.name[1] == "9":
                        continue
                    num_img_slo_pat += 1
                    if type_set == "six_landmarks":
                        not_found_slo_pat += compute_errors(landmarks_true, landmarks_pred, errors_slo_pat,
                                                            r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                            results, file.name, "p")
                    elif type_set == "gh_landmark":
                        not_found_slo_pat += compute_gh_errors(landmarks_true, landmarks_pred, errors_slo_pat,
                                                               results, file.name, "p")
                elif first_char == "n":
                    num_img_slo_norm += 1
                    if type_set == "six_landmarks":
                        not_found_slo_norm += compute_errors(landmarks_true, landmarks_pred, errors_slo_norm,
                                                             r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                             results, file.name, "n")
                    elif type_set == "gh_landmark":
                        not_found_slo_norm += compute_gh_errors(landmarks_true, landmarks_pred, errors_slo_norm,
                                                                results, file.name, "n")

            # Сохраняем подробный CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_csv_path, index=False)

            mean_error_ger_pat = np.mean(errors_ger_pat) if errors_ger_pat else None
            not_found_ger_pat = (not_found_ger_pat / (num_img_ger_pat * 6)) * 100

            mean_error_slo_pat = np.mean(errors_slo_pat) if errors_slo_pat else None
            not_found_slo_pat = (not_found_slo_pat / (num_img_slo_pat * 6)) * 100

            mean_error_slo_norm = np.mean(errors_slo_norm) if errors_slo_norm else None
            not_found_slo_norm = (not_found_slo_norm / (num_img_slo_norm * 6)) * 100

            mean_error = np.mean(np.concatenate([errors_ger_pat, errors_slo_pat, errors_slo_norm]))
            num_img = num_img_ger_pat + num_img_slo_pat + num_img_slo_norm
            not_found = ((not_found_ger_pat + not_found_slo_pat + not_found_slo_norm) / (num_img * 6)) * 100
            # add_info_logging("finish analysis", "work_logger")
            if type_set == "six_landmarks":
                mean_r_error = np.mean(r_errors) if r_errors else None
                mean_l_error = np.mean(l_errors) if l_errors else None
                mean_n_error = np.mean(n_errors) if n_errors else None
                mean_rlc_error = np.mean(rlc_errors) if rlc_errors else None
                mean_rnc_error = np.mean(rnc_errors) if rnc_errors else None
                mean_lnc_error = np.mean(lnc_errors) if lnc_errors else None

            add_info_logging("German pathology", "result_logger")
            add_info_logging(
                f"Mean Euclidean Distance: {mean_error_ger_pat:.4f} mm, not found: {not_found_ger_pat: .2f}%. Number of images:{num_img_ger_pat}",
                "result_logger")
            add_info_logging("Slovenian pathology", "result_logger")
            add_info_logging(
                f"Mean Euclidean Distance: {mean_error_slo_pat:.4f} mm, not found: {not_found_slo_pat: .2f}%. Number of images:{num_img_slo_pat}",
                "result_logger")
            add_info_logging("Slovenian normal", "result_logger")
            add_info_logging(
                f"Mean Euclidean Distance: {mean_error_slo_norm:.4f} mm, not found: {not_found_slo_norm: .2f}%. Number of images:{num_img_slo_norm}",
                "result_logger")
            add_info_logging("Sum", "result_logger")
            add_info_logging(
                f"Mean Euclidean Distance: {mean_error:.4f} mm, not found: {not_found: .2f}%. Number of images:{num_img}",
                "result_logger")
            if type_set == "six_landmarks":
                add_info_logging(f"Mean Euclidean Distance 'R' point: {mean_r_error}", "result_logger")
                add_info_logging(f"Mean Euclidean Distance 'L' point: {mean_l_error}", "result_logger")
                add_info_logging(f"Mean Euclidean Distance 'N' point: {mean_n_error}", "result_logger")
                add_info_logging(f"Mean Euclidean Distance 'RLC' point: {mean_rlc_error}", "result_logger")
                add_info_logging(f"Mean Euclidean Distance 'RNC' point: {mean_rnc_error}", "result_logger")
                add_info_logging(f"Mean Euclidean Distance 'LNC' point: {mean_lnc_error}", "result_logger")

        if type_set == "six_landmarks":
            point_name = {"all": "All", "r": "R", "l": "L", "n": "N", "rlc": "RLC", "rnc": "RNC", "lnc": "LNC"}
            graph_folder = result_folder_path / "six_landmarks_comparsion"
            graph_folder.mkdir(parents=True, exist_ok=True)
        elif type_set == "gh_landmark":
            point_name = {"gh": "Geometric Height"}
            graph_folder = result_folder_path / "gh_landmark_comparsion"
            graph_folder.mkdir(parents=True, exist_ok=True)
        for key, type_name in type_label.items():
            err_std = results_df["error"].std(numeric_only=True)
            err_std_groupby = results_df.groupby("group")["error"].std(numeric_only=True)
            if key == "all":
                data_for_plot = results_df[["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id", "error", point_name, data_for_plot,
                                      str(graph_folder), type_name)
            else:
                data_for_plot = results_df[results_df['group'] == key][["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id","error", point_name, data_for_plot,
                                      str(graph_folder), type_name)

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
                simulation = LandmarkingMonteCarlo(json_file=str(json_path/file.name[:-4]) + ".json",
                                                   nii_file=str(result_landmarks_folder/file.name[:-4]) + ".nii.gz" ,
                                                   npy_file=str(file))
                cur_angles, cur_dists =  simulation.run_simulation()
                arr_mean_angles_ger_pat = np.vstack([arr_mean_angles_ger_pat, cur_angles])
                arr_mean_dists_ger_pat = np.vstack([arr_mean_dists_ger_pat, cur_dists])
            if file.name[0] == "p":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
                                                   nii_file=str(result_landmarks_folder / file.name[:-4]) + ".nii.gz",
                                                   npy_file=str(file))
                cur_angles, cur_dists = simulation.run_simulation()
                arr_mean_angles_slo_pat = np.vstack([arr_mean_angles_slo_pat, cur_angles])
                arr_mean_dists_slo_pat = np.vstack([arr_mean_dists_slo_pat, cur_dists])
            if file.name[0] == "n":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
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


def find_morphometric_parameters(data_path, ds_folder_name):
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    result_folder_path = data_path / "result"
    results_csv_path = result_folder_path / f"morphometric_parameters_{ds_folder_name}.csv"
    files = list(result_landmarks_folder.glob("*.npz"))
    summary_result = []
    for file in files:
        lcc = LandmarkCentersCalculator()
        file_name = file.name[:-4] + ".nii.gz"
        landmarks_centers_of_peak = lcc.extract_landmarks_peak_npz(original_mask_folder / file_name, file)
        landmarks_centers_of_mass = lcc.extract_landmarks_com_npz(original_mask_folder / file_name, file)
        landmarks_centers_of_peaks_topk = lcc.extract_landmarks_topk_peaks_npz(original_mask_folder / file_name,
                                                                               file, top_k=10)
        for metric, sets_landmarks in metric_to_landmarks.items():
            result_cop = controller_metrics(metric, sets_landmarks, landmarks_centers_of_peak)
            result_com = controller_metrics(metric, sets_landmarks, landmarks_centers_of_mass)
            result_mc = controller_metrics(metric, sets_landmarks, landmarks_centers_of_peaks_topk, mc_option=True)
            summary_result.append({
                "file": file_name,
                "metric": metric,
                "peak": result_cop,
                "center_of_mass": result_com,
                "topk_peaks_avg": result_mc
            })

    df = pd.DataFrame.from_records(summary_result)
    result_folder_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv_path, index=False)

