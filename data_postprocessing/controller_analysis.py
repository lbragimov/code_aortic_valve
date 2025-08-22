import os
from ftplib import all_errors

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List
from data_postprocessing.evaluation_analysis import evaluate_segmentation
from data_postprocessing.montecarlo import LandmarkingMonteCarlo
from data_postprocessing.mask_analysis import mask_comparison, LandmarkCentersCalculator
from data_postprocessing.plotting_graphs import summarize_and_plot, plot_group_comparison, plot_table
from data_preprocessing.text_worker import add_info_logging
from models.controller_nnUnet import process_nnunet
from data_postprocessing.metrics_config import metric_to_landmarks
from data_postprocessing.geometric_metrics import controller_metrics


def mask_analysis(data_path, result_path, type_mask, folder_name):
    result_folder_path = os.path.join(result_path, folder_name)
    os.makedirs(result_folder_path, exist_ok=True)
    per_case_csv = os.path.join(str(result_folder_path), "per_case_metrics.csv")

    group_label_map = {
        "all": "All",
        "g": "Ger. path.",
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
                              os.path.join(str(result_folder_path), "aorta_root_comparison"))
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
                               predicting_mod=True, save_probabilities=save_probabilities)
                add_info_logging(f"radius: {radius}, type predicting: {type_map}", "result_logger")
            landmarks_analysis(data_path, ds_folder_name,
                             find_center_mass=find_center_mass,
                             find_monte_carlo=find_monte_carlo,
                             probabilities_map=save_probabilities)
            # data_path = Path(data_path)
            # result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
            # original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
            # json_path = data_path / "nnUNet_folder" / "json_info"


def landmarks_analysis(data_path, dict_all_case,
                       ds_folder_name,
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
        return pred

    def _compute_errors(true, pred, results, file_name, threshold=3.6):
        for number_key, name_key in point_name.items():
            dist = None
            if number_key == 'all':
                continue
            if number_key in pred:
                dist = np.linalg.norm(np.array(true[name_key.upper()][0]) - pred[number_key])  # Евклидово расстояние
                if dist > threshold:
                    dist = None

            results.append({
                "filename": file_name,
                "group": file_name[0],
                "point_id": name_key,
                "error": dist
            })
        return results

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
        "g": "Ger. path.",
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
            ext = "*.npz" if probabilities_map else "*.nii.gz"
            files = list(result_landmarks_folder.glob(ext))

            for file in files:
                landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                if len(landmarks_pred.keys()) < 5 and type_set == "six_landmarks":
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")
                elif len(landmarks_pred.keys()) < 1 and type_set == "gh_landmark":
                    add_info_logging(f"img: {file.name}, not found landmark", "result_logger")

                if file.name.endswith(".npz"):
                    file_name = file.name[:-4]
                elif file.name.endswith(".nii.gz"):
                    file_name = file.name[:-7]

                results = _compute_errors(dict_all_case[file_name], landmarks_pred, results, file_name)

            # Сохраняем подробный CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_csv_path, index=False)
            # add_info_logging("finish analysis", "work_logger")

        mean_error = results_df["error"].mean(numeric_only=True)
        std_error = results_df["error"].std(numeric_only=True)
        mean_error_ger_pat = results_df[results_df['group'] == "g"]["error"].mean(numeric_only=True)
        std_error_ger_pat = results_df[results_df['group'] == "g"]["error"].std(numeric_only=True)
        mean_error_slo_pat = results_df[results_df['group'] == "p"]["error"].mean(numeric_only=True)
        std_error_slo_pat = results_df[results_df['group'] == "p"]["error"].std(numeric_only=True)
        mean_error_slo_norm = results_df[results_df['group'] == "n"]["error"].mean(numeric_only=True)
        std_error_slo_norm = results_df[results_df['group'] == "n"]["error"].std(numeric_only=True)

        if type_set == "six_landmarks":
            num_img = int(len(results_df["error"]) / 6)
            not_found = (results_df["error"].isna().sum() / (num_img * 6)) * 100
            num_img_ger_pat = int(len(results_df[results_df['group'] == "g"]["error"]) / 6)
            not_found_ger_pat = (results_df[results_df['group'] == "g"]["error"].isna().sum() / (
                    num_img_ger_pat * 6)) * 100
            num_img_slo_pat = int(len(results_df[results_df['group'] == "p"]["error"]) / 6)
            not_found_slo_pat = (results_df[results_df['group'] == "p"]["error"].isna().sum() / (
                    num_img_slo_pat * 6)) * 100
            num_img_slo_norm = int(len(results_df[results_df['group'] == "n"]["error"]) / 6)
            not_found_slo_norm = (results_df[results_df['group'] == "n"]["error"].isna().sum() / (
                    num_img_slo_norm * 6)) * 100
        elif type_set == "gh_landmark":
            num_img = int(len(results_df["error"]))
            not_found = (results_df["error"].isna().sum() / num_img) * 100
            num_img_ger_pat = int(len(results_df[results_df['group'] == "g"]["error"]))
            not_found_ger_pat = (results_df[results_df['group'] == "g"]["error"].isna().sum() / num_img_ger_pat) * 100
            num_img_slo_pat = int(len(results_df[results_df['group'] == "p"]["error"]))
            not_found_slo_pat = (results_df[results_df['group'] == "p"]["error"].isna().sum() / num_img_slo_pat) * 100
            num_img_slo_norm = int(len(results_df[results_df['group'] == "n"]["error"]))
            not_found_slo_norm = (results_df[results_df['group'] == "n"]["error"].isna().sum() / num_img_slo_norm) * 100

        data_table = [
            ["All", round(mean_error, 2), round(std_error, 2), round(not_found, 2), num_img],
            ["German\npathology", round(mean_error_ger_pat, 2), round(std_error_ger_pat, 2),
             round(not_found_ger_pat, 2), num_img_ger_pat],
            ["Slovenian\npathology", round(mean_error_slo_pat, 2), round(std_error_slo_pat, 2),
             round(not_found_slo_pat, 2), num_img_slo_pat],
            ["Slovenian\nnormal", round(mean_error_slo_norm, 2), round(std_error_slo_norm, 2),
             round(not_found_slo_norm, 2), num_img_slo_norm],
        ]
        columns = ["Type", "Mean Euclidean\nDistance, mm", "Standard\nDeviation, mm",
                   "Not found, %", "Number of\nimages"]

        results_table_path = result_folder_path / f"landmark_errors_{ds_folder_name}.png"
        plot_table(data_table, columns, results_table_path)

        if type_set == "six_landmarks":
            point_name_dict = {"all": "All", "r": "R", "l": "L", "n": "N", "rlc": "RLC", "rnc": "RNC", "lnc": "LNC"}
            graph_folder = result_folder_path / "six_landmarks_comparsion"
            graph_folder.mkdir(parents=True, exist_ok=True)

            mean_r_error = results_df[results_df['point_id'] == "r"]["error"].mean(numeric_only=True)
            mean_l_error = results_df[results_df['point_id'] == "l"]["error"].mean(numeric_only=True)
            mean_n_error = results_df[results_df['point_id'] == "n"]["error"].mean(numeric_only=True)
            mean_rlc_error = results_df[results_df['point_id'] == "rlc"]["error"].mean(numeric_only=True)
            mean_rnc_error = results_df[results_df['point_id'] == "rnc"]["error"].mean(numeric_only=True)
            mean_lnc_error = results_df[results_df['point_id'] == "lnc"]["error"].mean(numeric_only=True)

        elif type_set == "gh_landmark":
            point_name_dict = {"gh": "Geometric Height"}
            graph_folder = result_folder_path / "gh_landmark_comparsion"
            graph_folder.mkdir(parents=True, exist_ok=True)

        for key, type_name in type_label.items():
            err_std = results_df["error"].std(numeric_only=True)
            err_std_groupby = results_df.groupby("group")["error"].std(numeric_only=True)
            if key == "all":
                data_for_plot = results_df[["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id", "error", point_name_dict, data_for_plot,
                                      str(graph_folder), type_name)
            else:
                data_for_plot = results_df[results_df['group'] == key][["point_id", "error"]].dropna(how='any')
                plot_group_comparison("point_id","error", point_name_dict, data_for_plot,
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

