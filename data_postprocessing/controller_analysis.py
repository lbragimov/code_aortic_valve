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


def mask_analysis(data_path, result_path, type_mask, folder_name):
    per_case_csv = os.path.join(result_path, "per_case_metrics.csv")
    aggregated_csv = os.path.join(result_path, "aggregated_metrics.csv")

    if os.path.exists(per_case_csv) and os.path.exists(aggregated_csv):
        # Загружаем данные из файлов
        add_info_logging("Using cached metrics from CSV files", "work_logger")
        df = pd.read_csv(per_case_csv)
        df_summary = pd.read_csv(aggregated_csv)

        # Восстановим структуру metrics_by_group
        metrics_by_group: Dict[str, Dict[str, list]] = {}
        for _, row in df_summary.iterrows():
            group = row["Group"]
            metric = row["Metric"]
            if group not in metrics_by_group:
                metrics_by_group[group] = {}
            if metric not in metrics_by_group[group]:
                # Найдём все значения этой метрики и группы из df
                values = df[df["group"] == group][metric].dropna().tolist()
                metrics_by_group[group][metric] = values
    else:
        # Пересчитываем метрики
        metrics_by_group, per_case_data = mask_comparison(data_path, type_mask=type_mask, folder_name=folder_name)

        # Сохраняем метрики по кейсам
        df = pd.DataFrame(per_case_data)
        df.to_csv(per_case_csv, index=False)

        # Сохраняем агрегированные метрики
        summary = []
        for group, metrics in metrics_by_group.items():
            for metric_name, values in metrics.items():
                mean = np.nanmean(values)
                std = np.nanstd(values)
                summary.append({"Group": group, "Metric": metric_name, "Mean": mean, "Std": std})
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(aggregated_csv, index=False)

    # Добавим агрегированную группу "all"
    all_metrics: Dict[str, List[float]] = {}
    for group, group_data in metrics_by_group.items():
        for metric, values in group_data.items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].extend(values)
    metrics_by_group["all"] = all_metrics

    # При необходимости — добавим "all" в df_summary
    for metric_name, values in all_metrics.items():
        mean = np.nanmean(values)
        std = np.nanstd(values)
        df_summary.loc[len(df_summary.index)] = ["all", metric_name, mean, std]

    # Пересохраним df_summary с учётом "all"
    df_summary.to_csv(aggregated_csv, index=False)

    # Строим графики
    for group, metrics in metrics_by_group.items():
        save_subdir = os.path.join(result_path, f"group_{group}")
        summarize_and_plot(metrics, save_subdir)

    plot_group_comparison(metrics_by_group, os.path.join(result_path, "group_comparison"))
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
                       probabilities_map=False):

    def process_file(file, original_mask_folder, probabilities_map):
        res_test = LandmarkCentersCalculator()
        if probabilities_map:
            file_name = file.name[:-4] + ".nii.gz"
            pred = res_test.compute_metrics_direct_npz(original_mask_folder / file_name, file)
            true = res_test.compute_metrics_direct_nii(original_mask_folder / file_name)
        else:
            pred = res_test.compute_metrics_direct_nii(file)
            true = res_test.compute_metrics_direct_nii(original_mask_folder / file.name)
        return true, pred

    def compute_errors(true, pred, error_list, r, l, n, rlc, rnc, lnc, results, file_name, group):
        not_found = 0
        for key in true:
            if key in pred:
                dist = np.linalg.norm(true[key] - pred[key]) # Евклидово расстояние
                error_list.append(dist)
                results.append({
                    "filename": file_name,
                    "group": group,
                    "point_id": key,
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

    # add_info_logging("start analysis", "work_logger")
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    json_path = data_path / "nnUNet_folder" / "json_info"
    result_folder_path = data_path / "result"

    results = []  # список словарей
    landmark_errors_grouped = {
        "all": [],
        "r": [],
        "l": [],
        "n": [],
        "rlc": [],
        "rnc": [],
        "lnc": []
    }

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
            first_char = file.name[0]
            if first_char == "H":
                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                num_img_ger_pat += 1
                not_found_ger_pat += compute_errors(landmarks_true, landmarks_pred, errors_ger_pat,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                    results, file.name, "H")
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")
            if first_char == "p":
                # if file.name[1] == "9":
                #     continue
                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                num_img_slo_pat += 1
                not_found_slo_pat += compute_errors(landmarks_true, landmarks_pred, errors_slo_pat,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                    results, file.name, "p")
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")
            if first_char == "n":
                landmarks_true, landmarks_pred = process_file(file, original_mask_folder, probabilities_map)
                num_img_slo_norm += 1
                not_found_slo_norm += compute_errors(landmarks_true, landmarks_pred, errors_slo_norm,
                                                    r_errors, l_errors, n_errors, rlc_errors, rnc_errors, lnc_errors,
                                                    results, file.name, "n")
                if len(landmarks_pred.keys()) < 5:
                    add_info_logging(f"img: {file.name}, not found landmark: {6 - len(landmarks_pred.keys())}",
                                     "result_logger")

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
        add_info_logging(f"Mean Euclidean Distance 'R' point: {mean_r_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'L' point: {mean_l_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'N' point: {mean_n_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'RLC' point: {mean_rlc_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'RNC' point: {mean_rnc_error}", "result_logger")
        add_info_logging(f"Mean Euclidean Distance 'LNC' point: {mean_lnc_error}", "result_logger")

        results_df = pd.DataFrame(results)
        csv_path = result_folder_path / f"landmark_errors_{ds_folder_name}.csv"
        results_df.to_csv(csv_path, index=False)
        # Сохраняем агрегированные ошибки по landmark'ам
        landmark_errors_grouped["all"] = r_errors + l_errors + n_errors + rlc_errors + rnc_errors + lnc_errors
        landmark_errors_grouped_dict = {
            "r": r_errors,
            "l": l_errors,
            "n": n_errors,
            "rlc": rlc_errors,
            "rnc": rnc_errors,
            "lnc": lnc_errors,
            "all": landmark_errors_grouped["all"]
        }
        landmark_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in landmark_errors_grouped_dict.items()]))
        grouped_csv_path = result_folder_path / f"landmark_errors_grouped_{ds_folder_name}.csv"
        landmark_df.to_csv(grouped_csv_path, index=False)
        add_info_logging(f"Saved landmark-wise grouped errors to: {grouped_csv_path}", "work_logger")
        add_info_logging(f"Saved CSV with errors to: {csv_path}", "work_logger")

        plot_group_comparison(landmark_errors_grouped_dict, str(result_folder_path), mode="landmarks")

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
