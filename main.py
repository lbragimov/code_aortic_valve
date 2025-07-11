import os
import logging
import shutil
from pathlib import Path
import platform
import json

import numpy as np
import pandas as pd
from statistics import mode
from datetime import datetime

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

from configurator.equipment_analysis import get_free_cpus
from data_preprocessing.dcm_nii_converter import convert_dcm_to_nii, reader_dcm, check_dcm_info
from data_preprocessing.stl_nii_converter import convert_stl_to_mask_nii, cut_mask_using_points
from data_preprocessing.check_structure import create_directory_structure, collect_file_paths
from data_preprocessing.text_worker import (json_reader, yaml_reader, yaml_save, json_save, txt_json_convert,
                                            add_info_logging, create_new_json, parse_txt_file)
from data_preprocessing.csv_worker import write_csv, read_csv
from data_preprocessing.crop_nii import cropped_image, find_global_size, find_shape, find_shape_2
# from data_postprocessing.evaluation_analysis import landmarking_testing
from data_postprocessing.controller_analysis import (landmarks_analysis, experiment_analysis, mask_analysis,
                                                     find_morphometric_parameters, LandmarkCentersCalculator)
from data_postprocessing.plotting_graphs import summarize_and_plot
from data_postprocessing.coherent_point_drift import create_new_gh_json, find_new_curv
from models.implementation_nnUnet import nnUnet_trainer
from data_visualization.markers import slices_with_markers, process_markers, find_mean_gh_landmark
from models.controller_nnUnet import process_nnunet

from optimization.parallelization import division_processes

from models.implementation_3D_Unet import WrapperUnet

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def controller(data_path, cpus):
    def _find_series_folders(root_folder, types_file, parent=False):
        if isinstance(types_file, str):
            types_file = [types_file]
        series_folders = set()
        for ext in types_file:
            for file_path in Path(root_folder).rglob(f"*.{ext}"):
                if parent:
                    series_folders.add(file_path.parent)
                else:
                    series_folders.add(file_path)
        return list(series_folders)

    def clear_folder(folder_path):
        """Очищает папку, удаляя все файлы и подпапки"""
        folder = Path(folder_path)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            add_info_logging(f"The folder '{folder_path}' did not exist, so it was created.",
                             "work_logger")
            return

        for item in folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # Удаляем файл или символическую ссылку
            elif item.is_dir():
                shutil.rmtree(item)  # Удаляем папку рекурсивно

    def _experiment_training(create_img=False, create_models=False):
        # list_radius = [10, 9, 8, 7, 6, 5, 4]
        # list_radius = [10, 9, 8, 7, 6]
        list_radius = [7, 6]
        if create_img:
            for radius in list_radius:
                cur_mask_markers_visual_path = os.path.join(data_path, f"markers_visual_{radius}")
                for sub_dir in list(dir_structure["nii_resample"]):
                    clear_folder(os.path.join(cur_mask_markers_visual_path, sub_dir))
                    for case in os.listdir(os.path.join(nii_resample_path, sub_dir)):
                        case_name = case[:-7]
                        nii_resample_case_file_path = os.path.join(nii_resample_path, sub_dir, case)
                        mask_markers_img_path = os.path.join(cur_mask_markers_visual_path, sub_dir, f"{case_name}.nii.gz")
                        process_markers(nii_resample_case_file_path,
                                        dict_all_case[case_name],
                                        mask_markers_img_path,
                                        radius)

            # Получаем все пути к изображениям в папке mask_aorta_segment_cut
            all_image_paths = []
            for sub_dir in dir_structure["mask_aorta_segment_cut"]:
                for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                    image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
                    all_image_paths.append(image_path)

            padding = 10
            # Найти общий bounding box для всех изображений
            global_size = find_global_size(all_image_paths, padding)
            add_info_logging(f"global size: {global_size}", "work_logger")

            for radius in list_radius:
                cur_mask_markers_visual_path = os.path.join(data_path, f"markers_visual_{radius}")
                cur_crop_markers_mask_path = os.path.join(data_path, f"crop_markers_mask_{radius}")
                for sub_dir in os.listdir(cur_mask_markers_visual_path):
                    clear_folder(os.path.join(cur_crop_markers_mask_path, sub_dir))
                    for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                        cropped_image(mask_image_path=str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
                                      input_image_path=str(os.path.join(cur_mask_markers_visual_path, sub_dir, case)),
                                      output_image_path=str(os.path.join(cur_crop_markers_mask_path, sub_dir, case)),
                                      size=global_size)

        if create_models:
            # dict_id_case = {10: 491, 9: 499, 8: 498, 7: 497, 6: 496, 5: 495, 4: 494}
            dict_id_case = {10: 481, 9: 489, 8: 488, 7: 487, 6: 486}
            for radius in list_radius:
                cur_crop_markers_mask_path = os.path.join(data_path, f"crop_markers_mask_{radius}")
                dict_dataset = {
                    "channel_names": {0: "CT"},
                    "labels": {
                        "background": 0,
                        "R": 1,
                        "L": 2,
                        "N": 3,
                        "RLC": 4,
                        "RNC": 5,
                        "LNC": 6
                    },
                    "file_ending": ".nii.gz"
                }
                process_nnunet(folder=nnUNet_folder, ds_folder_name=f"Dataset{dict_id_case[radius]}_AortaLandmarks",
                               id_case=dict_id_case[radius], folder_image_path=crop_nii_image_path,
                               folder_mask_path=cur_crop_markers_mask_path, dict_dataset=dict_dataset,
                               num_test=15, test_folder="Homburg pathology", create_ds=True, training_mod=True)

    temp_path = os.path.join(data_path, "temp")
    # temp = np.load(os.path.join(temp_path, "p9.npz"))

    result_folder = os.path.join(data_path, "result")
    dicom_folder = os.path.join(data_path, "dicom")
    image_folder = os.path.join(data_path, "image_nii")
    json_marker_path = os.path.join(data_path, "json_markers_info")
    txt_points_folder = os.path.join(data_path, "txt_points")
    stl_aorta_segment_path = os.path.join(data_path, "stl_aorta_segment")
    mask_aorta_segment_path = os.path.join(data_path, "mask_aorta_segment")
    nii_resample_path = os.path.join(data_path, "nii_resample")
    nii_convert_path = os.path.join(data_path, "nii_convert")
    mask_aorta_segment_cut_path = os.path.join(data_path, "mask_aorta_segment_cut")
    mask_markers_visual_path = os.path.join(data_path, "markers_visual")
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    nnUNet_DS_aorta_path = os.path.join(nnUNet_folder, "nnUNet_raw", "Dataset401_AorticValve")
    nnUNet_DS_landmarks_path = os.path.join(nnUNet_folder, "nnUNet_raw", "Dataset402_AortaLandmarks")
    crop_nii_image_path = os.path.join(data_path, "crop_nii_image")
    crop_markers_mask_path = os.path.join(data_path, "crop_markers_mask")
    UNet_3D_folder = os.path.join(data_path, "3DUNet_folder")
    json_duplication_g_h_path = os.path.join(data_path, "json_duplication_geometric_heights")
    json_land_mask_coord_path = os.path.join(data_path, "json_landmarks_mask_coord")

    controller_path = os.path.join(script_dir, "controller.yaml")
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")
    mask_gh_landmark_folder = os.path.join(data_path, "mask_gh_landmark_cut")

    dict_all_case = {}
    if os.path.isfile(controller_path):
        controller_dump = yaml_reader(controller_path)
    else:
        controller_dump = {}

    if not controller_dump.get("check_metadata_dicom"):
        summary_info = []
        dicom_folders = _find_series_folders(dicom_folder, "dcm", parent=True)

        for folder_path in dicom_folders:
            case_name = folder_path.name
            info = check_dcm_info(folder_path)
            info["case_name"] = case_name
            summary_info.append(info)
        df = pd.DataFrame(summary_info)
        write_csv(df, result_folder, "dicom_metadata_info.csv")

        # 📄 Сохраняем JSON с уникальными значениями по каждому столбцу (кроме case_name)
        unique_info = {
            col: sorted([str(x) for x in df[col].dropna().unique().tolist()])
            for col in df.columns if col != "case_name"
        }

        # Преобразуем к сериализуемым типам
        def convert(o):
            if isinstance(o, (tuple, list)):
                return list(o)
            return str(o)

        json_path = os.path.join(result_folder, "dicom_metadata_info_summary.json")
        with open(json_path, "w") as f:
            json.dump(unique_info, f, indent=4, default=convert)

        controller_dump["check_metadata_dicom"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("create_train_test_lists"):
        df = read_csv(result_folder, "dicom_metadata_info.csv")

        result_rows = []

        h_cases = df[df['case_name'].str.startswith('H')].copy()
        np_cases = df[df['case_name'].str.startswith(('n', 'p'))].copy()

        for idx, row in enumerate(h_cases.itertuples(index=False), start=1):
            result_rows.append({
                'case_name': row.case_name,
                'used_case_name': f'g{idx}',
                'type_series': 'test'
            })

        for letter in ['n', 'p']:
            group = np_cases[np_cases['case_name'].str.startswith(letter)].copy()
            n_total = len(group)
            n_test = int(round(n_total * 0.1))  # 10%

            shuffled = group.sample(frac=1, random_state=42).reset_index(drop=True)
            shuffled['type_series'] = ['test'] * n_test + ['train'] * (n_total - n_test)
            shuffled['used_case_name'] = shuffled['case_name']

            result_rows.extend(shuffled[['case_name', 'used_case_name', 'type_series']].to_dict(orient='records'))

        result_df = pd.DataFrame(result_rows)
        write_csv(result_df, result_folder, "train_test_lists.csv")

    train_test_lists = read_csv(result_folder, "train_test_lists.csv")
    test_cases = train_test_lists[train_test_lists['type_series'] == 'test']['used_case_name'].tolist()
    train_cases = train_test_lists[train_test_lists['type_series'] == 'train']['used_case_name'].tolist()

    if not controller_dump.get("convert_resample_dicom"):
        clear_folder(image_folder)
        dicom_folders = _find_series_folders(dicom_folder, "dcm", parent=True)

        for folder_path in dicom_folders:
            org_folder_name = folder_path.name
            try:
                new_file_name = train_test_lists.loc[
                    train_test_lists['case_name'] == org_folder_name, 'used_case_name'
                ].values[0]
            except IndexError:
                add_info_logging(f"'{org_folder_name}' not found in CSV.")
                continue

            img_convert_path = os.path.join(image_folder, new_file_name)
            convert_dcm_to_nii(str(folder_path), img_convert_path, zip=True)

        controller_dump["convert_resample_dicom"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("train_cases_list"):
        # temp
        ds_folder_name = "Dataset489_AortaLandmarks"
        train_nnUNet_DS_landmarks_folder = Path(nnUNet_folder) / "nnUNet_raw" / ds_folder_name / "imagesTr"
        trains_files = (list(train_nnUNet_DS_landmarks_folder.glob('*.nii')) +
                        list(train_nnUNet_DS_landmarks_folder.glob('*.nii.gz')))
        test_nnUNet_DS_landmarks_folder = Path(nnUNet_folder) / "nnUNet_raw" / ds_folder_name / "imagesTs"
        test_files = (list(test_nnUNet_DS_landmarks_folder.glob('*.nii')) +
                        list(test_nnUNet_DS_landmarks_folder.glob('*.nii.gz')))
        controller_dump["train_cases_list"] = [f.name.split('.')[0][:-5] for f in trains_files]
        controller_dump["test_cases_list"] = [f.name.split('.')[0][:-5] for f in test_files]
        yaml_save(controller_dump, controller_path)
        # temp

    if not dict_all_case:
        if os.path.isfile(dict_all_case_path):
            dict_all_case = json_reader(dict_all_case_path)
        else:
            controller_dump["create_dict_all_case"] = False
            yaml_save(controller_dump, controller_path)
            add_info_logging("There is no data dictionary for all cases. The dictionary will be regenerated.")

    if not controller_dump.get("create_dict_all_case"):
        txt_files = _find_series_folders(txt_points_folder, "txt")
        all_cases_data = {}

        for txt_path in txt_files:
            case_base_name = txt_path.stem[2:]  # Удаляем первые 2 символа

            # Найти в датафрейме
            match = train_test_lists[train_test_lists["case_name"].str.contains(case_base_name, na=False)]
            if match.empty:
                add_info_logging(f" Not found: {case_base_name} in dict cases")
                continue

            used_case_name = match.iloc[0]["used_case_name"]
            type_series = match.iloc[0]["type_series"]

            parsed_data = parse_txt_file(txt_path)
            parsed_data["type_series"] = type_series

            all_cases_data[used_case_name] = parsed_data

        json_save(all_cases_data, dict_all_case_path)
        controller_dump["create_dict_all_case"] = True
        yaml_save(controller_dump, controller_path)

    if not "mask_aorta_segment" in controller_dump.keys() or not controller_dump["mask_aorta_segment"]:
        for sub_dir in list(dir_structure["stl_aorta_segment"]):
            clear_folder(os.path.join(mask_aorta_segment_path, sub_dir))
            for case in os.listdir(os.path.join(stl_aorta_segment_path, sub_dir)):
                stl_aorta_segment_file = os.path.join(stl_aorta_segment_path, sub_dir, case)
                case_name = case[:-4]
                mask_aorta_segment_file = os.path.join(mask_aorta_segment_path, sub_dir, f"{case_name}.nii.gz")
                nii_resample_file = os.path.join(nii_resample_path, sub_dir, f"{case_name}.nii.gz")
                convert_stl_to_mask_nii(stl_aorta_segment_file,
                                        nii_resample_file,
                                        mask_aorta_segment_file)
        controller_dump["mask_aorta_segment"] = True
        yaml_save(controller_dump, controller_path)

    # all_image_paths = []
    # for sub_dir in dir_structure["mask_aorta_segment"]:
    #     for case in os.listdir(os.path.join(mask_aorta_segment_path, sub_dir)):
    #         image_path = os.path.join(mask_aorta_segment_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"mask_aorta_segment {find_shape_2(all_image_paths)}", "work_logger")

    if not "mask_aorta_segment_cut" in controller_dump.keys() or not controller_dump["mask_aorta_segment_cut"]:
        for sub_dir in list(dir_structure["stl_aorta_segment"]):
            clear_folder(os.path.join(mask_aorta_segment_cut_path, sub_dir))
            for case in os.listdir(os.path.join(mask_aorta_segment_path, sub_dir)):
                mask_aorta_segment_file = os.path.join(mask_aorta_segment_path, sub_dir, case)
                mask_aorta_segment_cut_file = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
                case_name = case[:-7]
                top_points = [dict_all_case[case_name]['R'],
                              dict_all_case[case_name]['L'],
                              dict_all_case[case_name]['N']]
                bottom_points = [dict_all_case[case_name]['RLC'],
                                 dict_all_case[case_name]['RNC'],
                                 dict_all_case[case_name]['LNC']]
                cut_mask_using_points(mask_aorta_segment_file,
                                      mask_aorta_segment_cut_file,
                                      top_points, bottom_points, margin=2)
        controller_dump["mask_aorta_segment_cut"] = True
        yaml_save(controller_dump, controller_path)

    if not "mask_markers_create" in controller_dump.keys() or not controller_dump["mask_markers_create"]:
        for sub_dir in list(dir_structure["nii_resample"]):
            clear_folder(os.path.join(mask_markers_visual_path, sub_dir))
            for case in os.listdir(os.path.join(nii_resample_path, sub_dir)):
                case_name = case[:-7]
                radius = 6
                nii_resample_case_file_path = os.path.join(nii_resample_path, sub_dir, case)
                mask_markers_img_path = os.path.join(mask_markers_visual_path, sub_dir, f"{case_name}.nii.gz")
                process_markers(nii_resample_case_file_path,
                                dict_all_case[case_name],
                                mask_markers_img_path,
                                radius)
        controller_dump["mask_markers_create"] = True
        yaml_save(controller_dump, controller_path)

    if not "nnUNet_DS_aorta" in controller_dump.keys() or not controller_dump["nnUNet_DS_aorta"]:
        clear_folder(os.path.join(nnUNet_DS_aorta_path, "imagesTr"))
        clear_folder(os.path.join(nnUNet_DS_aorta_path, "labelsTr"))
        clear_folder(os.path.join(nnUNet_DS_aorta_path, "imagesTs"))
        for sub_dir in list(dir_structure["nii_resample"]):
            file_count = len([f for f in os.listdir(os.path.join(nii_resample_path, sub_dir))])
            n = 0
            for case in os.listdir(os.path.join(nii_resample_path, sub_dir)):
                case_name = case[:-7]
                if int(file_count*0.8) >= n:
                    shutil.copy(str(os.path.join(nii_resample_path, sub_dir, case)),
                                str(os.path.join(nnUNet_DS_aorta_path, "imagesTr", f"{case_name}_0000.nii.gz")))
                    shutil.copy(str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
                                str(os.path.join(nnUNet_DS_aorta_path, "labelsTr", f"{case}.gz")))
                else:
                    shutil.copy(str(os.path.join(nii_resample_path, sub_dir, case)),
                                str(os.path.join(nnUNet_DS_aorta_path, "imagesTs", f"{case_name}_0000.nii.gz")))
                n += 1
        controller_dump["nnUNet_DS_aorta"] = True
        yaml_save(controller_dump, controller_path)

    if os.path.exists(nnUNet_DS_aorta_path):
        if not os.path.isfile(os.path.join(nnUNet_DS_aorta_path, "dataset.json")):
            file_count = len([f for f in os.listdir(os.path.join(nnUNet_DS_aorta_path, "imagesTr"))])
            generate_dataset_json(nnUNet_DS_aorta_path,
                                  channel_names={0: 'CT'},
                                  labels={'background': 0, 'aortic_valve': 1},
                                  num_training_cases=file_count,
                                  file_ending='.nii.gz')
            controller_dump["nnUNet_DS_json_aorta"] = True
            yaml_save(controller_dump, controller_path)
    else:
        add_info_logging("No folder to save to dataset.json", "work_logger")
        return

    # test_case_name = list(dict_all_case.keys())[0]

    # model_nnUnet = nnUnet_trainer(nnUNet_folder)
    # model_nnUnet.train_nnUnet(task_id=401, nnUnet_path=nnUNet_folder)
    # all_image_paths = []
    # for sub_dir in dir_structure["mask_aorta_segment_cut"]:
    #     for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
    #         image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"mask_aorta_segment_cut {find_shape_2(all_image_paths)}", "work_logger")

    if not controller_dump.get("crop_images"):
        if controller_dump.get("crop_img_size"):
            global_size = controller_dump["crop_img_size"]
        else:
            all_image_paths = []
            for sub_dir in dir_structure["mask_aorta_segment_cut"]:
                for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                    image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
                    all_image_paths.append(image_path)

            padding = 10
            # Найти общий bounding box для всех изображений
            global_size = find_global_size(all_image_paths, padding)
            controller_dump["crop_img_size"] = [int(x) for x in global_size]
            yaml_save(controller_dump, controller_path)

        for sub_dir in list(dir_structure["mask_aorta_segment_cut"]):
            clear_folder(os.path.join(crop_nii_image_path, sub_dir))
            clear_folder(os.path.join(crop_markers_mask_path, sub_dir))
            for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                cropped_image(mask_image_path=str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
                              input_image_path=str(os.path.join(nii_resample_path, sub_dir, case)),
                              output_image_path=str(os.path.join(crop_nii_image_path, sub_dir, case)),
                              size=global_size)
                cropped_image(mask_image_path=str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
                              input_image_path=str(os.path.join(mask_markers_visual_path, sub_dir, case)),
                              output_image_path=str(os.path.join(crop_markers_mask_path, sub_dir, case)),
                              size=global_size)
        controller_dump["crop_images"] = True
        yaml_save(controller_dump, controller_path)

    if (not "create_3D_UNet_data_base" in controller_dump.keys()
            or not controller_dump["create_3D_UNet_data_base"]):
        clear_folder(os.path.join(UNet_3D_folder, "data"))
        clear_folder(os.path.join(UNet_3D_folder, "test_data"))
        for sub_dir in list(dir_structure["crop_nii_image"]):
            file_count = len([f for f in os.listdir(os.path.join(crop_nii_image_path, sub_dir))])
            n = 0
            for case in os.listdir(os.path.join(crop_nii_image_path, sub_dir)):
                case_name = case[:-7]
                if int(file_count*0.8) >= n:
                    Path(os.path.join(UNet_3D_folder, "data", case_name)).mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(os.path.join(crop_nii_image_path, sub_dir, case)),
                                str(os.path.join(UNet_3D_folder, "data", case_name, "image.nii.gz")))
                    shutil.copy(str(os.path.join(crop_markers_mask_path, sub_dir, case)),
                                str(os.path.join(UNet_3D_folder, "data", case_name, "mask.nii.gz")))
                else:
                    Path(os.path.join(UNet_3D_folder, "test_data", case_name)).mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(os.path.join(crop_nii_image_path, sub_dir, case)),
                                str(os.path.join(UNet_3D_folder, "test_data", case_name, "image.nii.gz")))
                    shutil.copy(str(os.path.join(crop_markers_mask_path, sub_dir, case)),
                                str(os.path.join(UNet_3D_folder, "test_data", case_name, "mask.nii.gz")))
                n += 1
        controller_dump["create_3D_UNet_data_base"] = True
        yaml_save(controller_dump, controller_path)

    if not "nnUNet_DS_landmarks" in controller_dump.keys() or not controller_dump["nnUNet_DS_landmarks"]:
        clear_folder(os.path.join(nnUNet_DS_landmarks_path, "imagesTr"))
        clear_folder(os.path.join(nnUNet_DS_landmarks_path, "labelsTr"))
        clear_folder(os.path.join(nnUNet_DS_landmarks_path, "imagesTs"))
        for sub_dir in list(dir_structure["crop_nii_image"]):
            file_count = len([f for f in os.listdir(os.path.join(crop_nii_image_path, sub_dir))])
            n = 0
            for case in os.listdir(os.path.join(crop_nii_image_path, sub_dir)):
                case_name = case[:-7]
                if int(file_count*0.8) >= n:
                    shutil.copy(str(os.path.join(crop_nii_image_path, sub_dir, case)),
                                str(os.path.join(nnUNet_DS_landmarks_path, "imagesTr", f"{case_name}_0000.nii.gz")))
                    shutil.copy(str(os.path.join(crop_markers_mask_path, sub_dir, case)),
                                str(os.path.join(nnUNet_DS_landmarks_path, "labelsTr", f"{case}")))
                else:
                    shutil.copy(str(os.path.join(crop_nii_image_path, sub_dir, case)),
                                str(os.path.join(nnUNet_DS_landmarks_path, "imagesTs", f"{case_name}_0000.nii.gz")))
                n += 1
        controller_dump["nnUNet_DS_landmarks"] = True
        yaml_save(controller_dump, controller_path)

        if os.path.exists(nnUNet_DS_aorta_path):
            if not os.path.isfile(os.path.join(nnUNet_DS_landmarks_path, "dataset.json")):
                file_count = len([f for f in os.listdir(os.path.join(nnUNet_DS_landmarks_path, "imagesTr"))])
                generate_dataset_json(nnUNet_DS_landmarks_path,
                                      channel_names={0: 'CT'},
                                      labels={'background': 0, 'R': 1, 'L': 2, 'N': 3, 'RLC': 4, 'RNC': 5, 'LNC': 6},
                                      num_training_cases=file_count,
                                      file_ending='.nii.gz')
                controller_dump["nnUNet_DS_json_landmarks"] = True
                yaml_save(controller_dump, controller_path)
        else:
            add_info_logging("No folder to save to dataset.json", "work_logger")
            return

    if not "nnUNet_lmk_ger_sep" in controller_dump.keys() or not controller_dump["nnUNet_lmk_ger_sep"]:
        dict_dataset = {
            "channel_names": {0: "CT"},
            "labels": {
                "background": 0,
                "R": 1,
                "L": 2,
                "N": 3,
                "RLC": 4,
                "RNC": 5,
                "LNC": 6
            },
            "file_ending": ".nii.gz"
        }
        process_nnunet(folder=nnUNet_folder, ds_folder_name="Dataset403_AortaLandmarks", id_case=403,
                       folder_image_path=crop_nii_image_path, folder_mask_path=crop_markers_mask_path,
                       dict_dataset=dict_dataset, pct_test=0.15, test_folder="Homburg pathology",
                       create_ds=True, training_mod=True)
        controller_dump["nnUNet_lmk_ger_sep"] = True
        yaml_save(controller_dump, controller_path)

    # model_3D_Unet = WrapperUnet()
    # model_3D_Unet.try_unet3d_training(UNet_3D_folder)
    # model_3D_Unet.try_unet3d_testing(UNet_3D_folder)

    # input_folder = os.path.join(nnUNet_folder, "nnUNet_raw", "Dataset402_AortaLandmarks", "imagesTs")
    # output_folder = os.path.join(nnUNet_folder, "nnUNet_test", "Dataset402_AortaLandmarks")
    # model_nnUnet_402 = nnUnet_trainer(nnUNet_folder)
    # model_nnUnet_402.preprocessing(task_id=402)
    # model_nnUnet_402.train(task_id=402, fold="all")
    # # model_nnUnet_402.reassembling_model(nnUnet_path=nnUNet_folder, case_path="Dataset402_AortaLandmarks")
    # model_nnUnet_402.predicting(input_folder=input_folder,
    #                             output_folder=output_folder,
    #                             task_id=402, fold="all")

    # process_nnunet(folder=nnUNet_folder, ds_folder_name="Dataset404_AortaLandmarks", id_case=404,
    #                folder_image_path=None, folder_mask_path=None, dict_dataset=None, pct_test=None,
    #                testing_mod=True, save_probabilities=True)
    #
    # ds_folder_name = "Dataset404_AortaLandmarks"
    # data_path_2 = Path(data_path)
    # process_analysis(data_path=data_path_2, ds_folder_name=ds_folder_name, find_center_mass=True, probabilities_map=True)

    if not controller_dump["experiment"]:
        _experiment_training(create_img=False, create_models=True)
        experiment_analysis(data_path=data_path,
                            dict_case = {10: 481, 9: 489, 8: 488, 7: 487, 6: 486, 5: 485, 4: 484})

    if not controller_dump["aorta_mask_analysis"]:
        mask_analysis(data_path, result_folder, type_mask="aortic_valve", folder_name="Dataset401_AorticValve")

    if not controller_dump["landmarks_analysis"]:
        landmarks_analysis(Path(data_path), ds_folder_name="Dataset489_AortaLandmarks",
                           find_center_mass=True, probabilities_map=True)

    if not controller_dump["calc_morphometric"]:
        find_morphometric_parameters(data_path, ds_folder_name="Dataset499_AortaLandmarks")

    if not controller_dump["duplication_geometric_heights"]:
        ds_folder_name = "Dataset489_AortaLandmarks"
        train_nnUNet_DS_landmarks_path = os.path.join(nnUNet_folder, "nnUNet_raw", ds_folder_name, "imagesTr")
        test_nnUNet_DS_landmarks_path = os.path.join(nnUNet_folder, "nnUNet_raw", ds_folder_name, "imagesTs")
        cases_folders_list = [train_nnUNet_DS_landmarks_path, test_nnUNet_DS_landmarks_path]
        json_dupl_folder = os.path.join(json_duplication_g_h_path, "train")
        for cur_folder in cases_folders_list:
            clear_folder(json_dupl_folder)
            for file in os.listdir(cur_folder):
                file_name = file[:-12]
                first_letter = file[:1]
                if first_letter == "H":
                    json_org_file = os.path.join(json_marker_path, "Homburg pathology", f"{file_name}.json")
                elif first_letter == "n":
                    json_org_file = os.path.join(json_marker_path, "Normal", f"{file_name}.json")
                else:
                    json_org_file = os.path.join(json_marker_path, "Pathology", f"{file_name}.json")
                json_dupl_file = os.path.join(json_dupl_folder, f"{file_name}.json")

                create_new_gh_json(json_org_file, json_dupl_file, n_points=10)
            json_dupl_folder = os.path.join(json_duplication_g_h_path, "test")

        controller_dump["duplication_geometric_heights"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump["json_landmarks_mask_coord"]:
        ds_folder_name_ldms = "Dataset489_AortaLandmarks"
        ds_folder_name_gh = "Dataset479_GeometricHeight"
        cur_nnUNet_folder_path = Path(nnUNet_folder)
        train_mask_ldms_folder_path = cur_nnUNet_folder_path / "nnUNet_raw" / ds_folder_name_ldms / "labelsTr"
        test_mask_ldms_folder_path = cur_nnUNet_folder_path / "nnUNet_test" / ds_folder_name_ldms
        original_mask_ldms_folder_path = cur_nnUNet_folder_path / "original_mask" / ds_folder_name_ldms
        mask_gh_train_folder = cur_nnUNet_folder_path / "nnUNet_raw" / ds_folder_name_gh / "labelsTr"
        mask_gh_org_folder = cur_nnUNet_folder_path / "original_mask" / ds_folder_name_gh
        mask_gh_test_folder = cur_nnUNet_folder_path / "nnUNet_test" / ds_folder_name_gh
        cases_folders_list = [train_mask_ldms_folder_path, test_mask_ldms_folder_path]

        type_set = "train"
        res_test = LandmarkCentersCalculator()
        json_land_mask_org_coord_folder = os.path.join(json_land_mask_coord_path, "original")
        for cur_folder in cases_folders_list:
            if type_set == "train":
                files = list(Path(cur_folder).glob("*.nii.gz"))
                json_land_mask_coord_folder = os.path.join(json_land_mask_coord_path, "train")
            else:
                files = list(Path(cur_folder).glob("*.npz"))
                json_land_mask_coord_folder = os.path.join(json_land_mask_coord_path, "test")
            for file in files:
                json_dupl_file = os.path.join(json_land_mask_coord_folder, f"{file.name.split('.')[0]}.json")
                pred_org = {}
                file_name = file.name.split('.')[0] + ".nii.gz"
                if type_set == "train":
                    mask_gh_org_file = mask_gh_train_folder / file_name
                    labels = {1: "R", 2: "L", 3: "N", 4: "RLC", 5: "RNC", 6: "LNC"}
                    pred = {}
                    for key, name in labels.items():
                        pred[key] = dict_all_case[file.name.split('.')[0]][name]
                    pred_gh_train = res_test.extract_landmarks_com_nii(mask_gh_org_file)
                    pred[7] = pred_gh_train[1]
                else:
                    mask_gh_org_file = mask_gh_org_folder / file_name
                    mask_gh_test_file = mask_gh_test_folder / str(file.name.split('.')[0] + ".npz")
                    pred = res_test.extract_landmarks_com_npz(original_mask_ldms_folder_path / file_name, file)
                    labels = {1: "R", 2: "L", 3: "N", 4: "RLC", 5: "RNC", 6: "LNC"}
                    for key, name in labels.items():
                        pred_org[key] = dict_all_case[file.name.split('.')[0]][name]
                    pred_gh_test = res_test.extract_landmarks_com_npz(original_mask_ldms_folder_path / file_name,
                                                                      mask_gh_test_file)
                    pred[7] = pred_gh_test[1]
                    pred_gh_org = res_test.extract_landmarks_com_nii(mask_gh_org_file)
                    pred_org[7] = pred_gh_org[1]
                create_new_json(json_dupl_file, pred)
                if pred_org:
                    json_dupl_org_file = os.path.join(json_land_mask_org_coord_folder, f"{file.name.split('.')[0]}.json")
                    create_new_json(json_dupl_org_file, pred_org)
            type_set = "test"

        controller_dump["json_landmarks_mask_coord"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump["find_geometric_heights"]:
        test_cases_path = os.path.join(json_land_mask_coord_path, "test")
        train_cases_path = os.path.join(json_land_mask_coord_path, "train")
        org_cases_path = os.path.join(json_land_mask_coord_path, "original")
        list_test_cases = list(Path(test_cases_path).glob("*.json"))
        list_train_cases = list(Path(train_cases_path).glob("*.json"))
        test_curv_cases_path = os.path.join(json_duplication_g_h_path, "test")
        train_curv_cases_path = os.path.join(json_duplication_g_h_path, "train")
        result_curv_cases_path = os.path.join(json_duplication_g_h_path, "result")
        clear_folder(result_curv_cases_path)

        errors = []  # сюда будем собирать результаты по кейсам

        for cur_test_case in list_test_cases:
            find_new_curv(cur_test_case, list_train_cases, train_curv_cases_path, result_curv_cases_path, org_cases_path)

            # Сравнение предсказанных кривых с истинными
            pred_json_path = Path(result_curv_cases_path) / cur_test_case.name
            true_json_path = Path(test_curv_cases_path) / cur_test_case.name

            def _load_curves(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return {k: np.array(v) for k, v in data.items()}

            pred_curves = _load_curves(pred_json_path)
            true_curves = _load_curves(true_json_path)

            combined_pred = []
            combined_true = []

            for name in ["RGH", "NGH", "LGH"]:

                pred = pred_curves[name]
                true = true_curves[name]

                combined_pred.append(pred)
                combined_true.append(true)

            if combined_pred and combined_true:
                all_pred = np.vstack(combined_pred)
                all_gt = np.vstack(combined_true)

                diff = all_pred - all_gt
                mse = np.mean(np.square(diff))
                rmse = np.sqrt(mse)

                # Сохраняем ошибку в список
                errors.append({"case": cur_test_case.name.split('.')[0], "rmse": rmse})

        # После цикла создаем DataFrame
        df_errors = pd.DataFrame(errors)

        # Вычисляем среднее и std
        mean_rmse = df_errors["rmse"].mean()
        std_rmse = df_errors["rmse"].std()

        print(f"Average RMSE across all cases: {mean_rmse:.3f}")
        print(f"Standard Deviation RMSE: {std_rmse:.3f}")

        # Сохраняем в CSV
        csv_path = Path(result_folder) / "rmse_errors_per_case.csv"
        df_errors.to_csv(csv_path, index=False)

    if not controller_dump.get("mask_gh_marker_create"):
        if controller_dump.get("crop_img_size"):
            global_size = controller_dump["crop_img_size"]
        else:
            all_image_paths = []
            for sub_dir in dir_structure["mask_aorta_segment_cut"]:
                for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                    image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
                    all_image_paths.append(image_path)

            padding = 10
            # Найти общий bounding box для всех изображений
            global_size = find_global_size(all_image_paths, padding)
            controller_dump["crop_img_size"] = [int(x) for x in global_size]
            yaml_save(controller_dump, controller_path)
        radius = 9
        train_folder = os.path.join(mask_gh_landmark_folder, "train")
        test_folder = os.path.join(mask_gh_landmark_folder, "test")
        clear_folder(train_folder)
        clear_folder(test_folder)
        type_cases_list = [controller_dump["train_cases_list"], controller_dump["test_cases_list"]]
        for n, case_list in enumerate(type_cases_list):
            for file_name in case_list:
                if file_name.startswith("H"):
                    sub_dir_name = "Homburg pathology"
                elif file_name.startswith("n"):
                    sub_dir_name = "Normal"
                else:
                    sub_dir_name = "Pathology"
                nii_resample_case_file_path = os.path.join(nii_resample_path,
                                                           sub_dir_name,
                                                           f"{file_name}.nii.gz")
                mask_aorta_img_path = os.path.join(mask_aorta_segment_cut_path,
                                                   sub_dir_name,
                                                   f"{file_name}.nii.gz")
                json_file_path = os.path.join(json_marker_path,
                                         sub_dir_name,
                                         f"{file_name}.json")
                with open(json_file_path, 'r') as f:
                    landmarks_coord_data = json.load(f)
                if n == 0:
                    mask_landmark_img_path = os.path.join(train_folder, f"{file_name}.nii.gz")
                else:
                    mask_landmark_img_path = os.path.join(test_folder, f"{file_name}.nii.gz")
                process_markers(nii_resample_case_file_path,
                                {"GH": find_mean_gh_landmark(landmarks_coord_data)},
                                mask_landmark_img_path,
                                radius)
                cropped_image(mask_image_path=mask_aorta_img_path,
                              input_image_path=mask_landmark_img_path,
                              output_image_path=mask_landmark_img_path,
                              size=global_size)
        controller_dump["mask_gh_marker_create"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_DS_gh"):
        ds_folder_name = "Dataset479_GeometricHeight"
        nnUNet_DS_gh_folder = os.path.join(nnUNet_folder, "nnUNet_raw", ds_folder_name)
        img_train_folder = os.path.join(nnUNet_DS_gh_folder, "imagesTr")
        clear_folder(img_train_folder)
        mask_train_folder = os.path.join(nnUNet_DS_gh_folder, "labelsTr")
        clear_folder(mask_train_folder)
        img_test_folder = os.path.join(nnUNet_DS_gh_folder, "imagesTs")
        clear_folder(img_test_folder)
        mask_org_folder = os.path.join(nnUNet_folder, "original_mask", ds_folder_name)
        clear_folder(mask_org_folder)
        mask_gh_train_folder = os.path.join(mask_gh_landmark_folder, "train")
        mask_gh_test_folder = os.path.join(mask_gh_landmark_folder, "test")
        type_cases_list = [controller_dump["train_cases_list"], controller_dump["test_cases_list"]]
        for n, case_list in enumerate(type_cases_list):
            for file_name in case_list:
                if file_name.startswith("H"):
                    sub_dir_name = "Homburg pathology"
                elif file_name.startswith("n"):
                    sub_dir_name = "Normal"
                else:
                    sub_dir_name = "Pathology"

                if n == 0:
                    shutil.copy(str(os.path.join(crop_nii_image_path, sub_dir_name, f"{file_name}.nii.gz")),
                                str(os.path.join(img_train_folder, f"{file_name}_0000.nii.gz")))
                    shutil.copy(str(os.path.join(mask_gh_train_folder, f"{file_name}.nii.gz")),
                                str(os.path.join(mask_train_folder, f"{file_name}.nii.gz")))
                else:
                    shutil.copy(str(os.path.join(crop_nii_image_path, sub_dir_name, f"{file_name}.nii.gz")),
                                str(os.path.join(img_test_folder, f"{file_name}_0000.nii.gz")))
                    shutil.copy(str(os.path.join(mask_gh_test_folder, f"{file_name}.nii.gz")),
                                str(os.path.join(mask_org_folder, f"{file_name}.nii.gz")))
        file_count = len([f for f in os.listdir(img_train_folder)])
        generate_dataset_json(nnUNet_DS_gh_folder,
                              channel_names={0: 'CT'},
                              labels={'background': 0, 'gh': 1},
                              num_training_cases=file_count,
                              file_ending='.nii.gz')

        controller_dump["nnUNet_DS_gh"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("nnUNet_gh_train_test"):
        ds_folder_name = "Dataset479_GeometricHeight"
        process_nnunet(folder=nnUNet_folder, ds_folder_name=ds_folder_name, id_case=479,
                       folder_image_path=None, folder_mask_path=None,
                       dict_dataset=None, pct_test=None, test_folder=None,
                       create_ds=False, training_mod=True, testing_mod=True, save_probabilities=True)
        controller_dump["nnUNet_gh_train_test"] = True
        yaml_save(controller_dump, controller_path)

    if not controller_dump.get("gh_landmark_analysis"):
        landmarks_analysis(Path(data_path), ds_folder_name="Dataset479_GeometricHeight",
                           find_center_mass=True, probabilities_map=True, type_set="gh_landmark")
        controller_dump["gh_landmark_analysis"] = True
        yaml_save(controller_dump, controller_path)

    print('hi')


if __name__ == "__main__":
    current_os = platform.system()
    script_dir = Path(__file__).resolve().parent

    if current_os == "Windows":
        data_path = "C:/Users/Kamil/Aortic_valve/data/"
        # data_path = "C:/Users/Kamil/Aortic_valve/data/temp"
        # data_path = "D:/science/Aortic_valve/data_short"
    elif current_os == "Linux":
        data_path = "/home/kamili/data/data_aortic_valve/"
    else:
        data_path = None

    script_dir = Path(__file__).resolve().parent
    data_structure_path = os.path.join(script_dir, "dir_structure.json")
    dir_structure = json_reader(data_structure_path)
    create_directory_structure(data_path, json_reader(data_structure_path))

    if data_path:
        free_cpus = get_free_cpus()
        current_time = datetime.now()

        # Логгер для хода программы
        log_file_name = current_time.strftime("log_%d-%m-%Y__%H-%M.log")
        work_log_path = os.path.join(data_path, "result/log", log_file_name)
        work_logger = logging.getLogger("work_logger")
        work_logger.setLevel(logging.INFO)
        work_handler = logging.FileHandler(work_log_path, mode='w')
        work_handler.setFormatter(logging.Formatter('%(message)s'))
        work_logger.addHandler(work_handler)

        # Логгер для результатов
        result_file_name = current_time.strftime("result_%d-%m-%Y__%H-%M.log")
        result_log_path = os.path.join(data_path, "result/log", result_file_name)
        result_logger = logging.getLogger("result_logger")
        result_logger.setLevel(logging.INFO)
        result_handler = logging.FileHandler(result_log_path, mode='w')
        result_handler.setFormatter(logging.Formatter('%(message)s'))  # без времени
        result_logger.addHandler(result_handler)

        # # Очистим базовые хендлеры у root-логгера (если есть)
        # for handler in logging.root.handlers[:]:
        #     logging.root.removeHandler(handler)

        controller(data_path, cpus=free_cpus)

