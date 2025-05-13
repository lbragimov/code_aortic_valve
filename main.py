import os
import logging
import shutil
from pathlib import Path
import platform

import numpy as np
from statistics import mode
from datetime import datetime

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

from configurator.equipment_analysis import get_free_cpus
from data_preprocessing.dcm_nii_converter import convert_dcm_to_nii, resample_nii, reader_dcm
from data_preprocessing.stl_nii_converter import convert_stl_to_mask_nii, cut_mask_using_points
from data_preprocessing.check_structure import create_directory_structure, collect_file_paths
from data_preprocessing.text_worker import (json_reader, yaml_reader, yaml_save, json_save, txt_json_convert,
                                            add_info_logging)
from data_preprocessing.crop_nii import cropped_image, find_global_size, find_shape, find_shape_2
# from data_postprocessing.evaluation_analysis import landmarking_testing
from data_postprocessing.controller_analysis import process_analysis, experiment_analysis, mask_analysis
from data_postprocessing.plotting_graphs import summarize_and_plot
from models.implementation_nnUnet import nnUnet_trainer
from data_visualization.markers import slices_with_markers, process_markers
from models.controller_nnUnet import process_nnunet

from optimization.parallelization import division_processes

from models.implementation_3D_Unet import WrapperUnet

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def controller(data_path, cpus):
    def clear_folder(folder_path):
        """Очищает папку, удаляя все файлы и подпапки"""
        folder = Path(folder_path)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            add_info_logging(f"Папка '{folder_path}' не существовала, поэтому была создана.",
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



    script_dir = Path(__file__).resolve().parent

    temp_path = os.path.join(data_path, "temp")
    # temp = np.load(os.path.join(temp_path, "p9.npz"))

    result_path = os.path.join(data_path, "result")
    dicom_path = os.path.join(data_path, "dicom")
    json_marker_path = os.path.join(data_path, "json_markers_info")
    txt_marker_path = os.path.join(data_path, "markers_info")
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

    controller_path = os.path.join(script_dir, "controller.yaml")
    data_structure_path = os.path.join(script_dir, "dir_structure.json")
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")

    dict_all_case = {}
    if os.path.isfile(controller_path):
        controller_dump = yaml_reader(controller_path)
    else:
        controller_dump = {}
    dir_structure = json_reader(data_structure_path)
    # create_directory_structure(data_path, dir_structure)

    if not "convert" in controller_dump.keys() or not controller_dump["convert"]:
        for sub_dir in list(dir_structure['dicom']):
            clear_folder(os.path.join(nii_convert_path, sub_dir))
            for case in os.listdir(os.path.join(dicom_path, sub_dir)):
                dcm_case_path = os.path.join(dicom_path, sub_dir, case)
                if sub_dir == "Homburg pathology":
                    case = case[:-3]
                nii_convert_case_file = os.path.join(nii_convert_path, sub_dir, case)
                img_size, img_origin, img_spacing, img_direction = convert_dcm_to_nii(dcm_case_path,
                                                                                      nii_convert_case_file,
                                                                                      zip=True)
                dict_all_case.setdefault(case, {})
                dict_all_case[case] |= {
                    "img_size": img_size,
                    "img_origin": img_origin,
                    "img_spacing": img_spacing,
                    "img_direction": img_direction
                }
        json_save(dict_all_case, dict_all_case_path)
        controller_dump["convert"] = True
        controller_dump["resample"] = False
        yaml_save(controller_dump, controller_path)

    if not dict_all_case:
        if os.path.isfile(dict_all_case_path):
            dict_all_case = json_reader(dict_all_case_path)
        else:
            for sub_dir in list(dir_structure['dicom']):
                for case in os.listdir(os.path.join(dicom_path, sub_dir)):
                    dcm_case_path = os.path.join(dicom_path, sub_dir, case)
                    if sub_dir == "Homburg pathology":
                        case = case[:-3]
                    img_size, img_origin, img_spacing, img_direction = reader_dcm(dcm_case_path)
                    dict_all_case[case] = {
                        "img_size": img_size,
                        "img_origin": img_origin,
                        "img_spacing": img_spacing,
                        "img_direction": img_direction
                    }
            if os.path.exists(json_marker_path):
                for sub_dir in list(dir_structure["json_markers_info"]):
                    for case in os.listdir(os.path.join(json_marker_path, sub_dir)):
                        json_marker_case_file = os.path.join(json_marker_path, sub_dir, case)
                        case_name = case[:-5]
                        data = json_reader(json_marker_case_file)
                        dict_all_case[case_name] |= {
                            "R": data["R"],
                            "L": data["L"],
                            "N": data["N"],
                            "RLC": data["RLC"],
                            "RNC": data["RNC"],
                            "LNC": data["LNC"]
                        }
            json_save(dict_all_case, dict_all_case_path)

    if not "resample" in controller_dump.keys() or not controller_dump["resample"]:

        if controller_dump["size_or_pixel"]:
            all_shapes = []
            nii_convert_file_paths = collect_file_paths(nii_convert_path, dir_structure["nii_convert"])
            for sub_dir in dir_structure["nii_convert"]:
                for case in os.listdir(os.path.join(nii_convert_path, sub_dir)):
                    image_path = os.path.join(nii_convert_path, sub_dir, case)
                    all_shapes.append(find_shape(image_path, controller_dump["size_or_pixel"]))
            # add_info_logging(f"all_shapes {set(all_shapes)}", "work_logger")

            # Extract the first elements of "img_spacing" and store them in a list
            # img_spac_0 = [case['img_spacing'][0] for case in dict_all_case.values()]
            img_spac_0 = [case[0] for case in all_shapes]
            # Find the minimum value and the average of the first elements
            min_img_spac_0 = min(img_spac_0)
            max_img_spac_0 = max(img_spac_0)
            avg_img_spac_0 = sum(img_spac_0) / len(img_spac_0)
            most_img_spac_0 = float(mode(img_spac_0))

            # Extract the first elements of "img_spacing" and store them in a list
            # img_spac_1 = [case['img_spacing'][1] for case in dict_all_case.values()]
            img_spac_1 = [case[1] for case in all_shapes]
            # Find the minimum value and the average of the first elements
            min_img_spac_1 = min(img_spac_1)
            max_img_spac_1 = max(img_spac_1)
            avg_img_spac_1 = sum(img_spac_1) / len(img_spac_1)
            most_img_spac_1 = float(mode(img_spac_1))

            # Extract the first elements of "img_spacing" and store them in a list
            # img_spac_2 = [case['img_spacing'][2] for case in dict_all_case.values()]
            img_spac_2 = [case[2] for case in all_shapes]
            # Find the minimum value and the average of the first elements
            min_img_spac_2 = min(img_spac_2)
            max_img_spac_2 = max(img_spac_2)
            avg_img_spac_2 = sum(img_spac_2) / len(img_spac_2)
            most_img_spac_2 = float(mode(img_spac_2))

        for sub_dir in list(dir_structure["nii_convert"]):
            clear_folder(os.path.join(nii_resample_path, sub_dir))
            for case in os.listdir(os.path.join(nii_convert_path, sub_dir)):
                nii_convert_case_file_path = os.path.join(nii_convert_path, sub_dir, case)
                nii_resample_case_file_path = os.path.join(nii_resample_path, sub_dir, case)
                if controller_dump["size_or_pixel"]:
                    resample_nii(nii_convert_case_file_path,
                                 nii_resample_case_file_path,
                                 controller_dump["size_or_pixel"],
                                 [most_img_spac_0, most_img_spac_1, most_img_spac_2])
                else:
                    resample_nii(nii_convert_case_file_path,
                                 nii_resample_case_file_path)
        controller_dump["resample"] = True
        yaml_save(controller_dump, controller_path)

    # all_image_paths = []
    # for sub_dir in dir_structure["nii_resample"]:
    #     for case in os.listdir(os.path.join(nii_resample_path, sub_dir)):
    #         image_path = os.path.join(nii_resample_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"nii_resample {find_shape_2(all_image_paths)}", "work_logger")

    if not "markers_info" in controller_dump.keys() or not controller_dump["markers_info"]:
        for sub_dir in list(dir_structure["markers_info"]):
            for case in os.listdir(os.path.join(txt_marker_path, sub_dir)):
                txt_marker_case_file = os.path.join(txt_marker_path, sub_dir, case)
                case = case[2:-4]
                json_marker_case_file = os.path.join(json_marker_path, sub_dir, f"{case}.json")
                data = txt_json_convert(txt_marker_case_file, json_marker_case_file)
                dict_all_case[case] |= {
                    "R": data["R"],
                    "L": data["L"],
                    "N": data["N"],
                    "RLC": data["RLC"],
                    "RNC": data["RNC"],
                    "LNC": data["LNC"]
                }
        controller_dump["markers_info"] = True
        yaml_save(controller_dump, controller_path)
        json_save(dict_all_case, dict_all_case_path)

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

    if not "crop_images" in controller_dump.keys() or not controller_dump["crop_images"]:
        # Получаем все пути к изображениям в папке mask_aorta_segment_cut
        all_image_paths = []
        for sub_dir in dir_structure["mask_aorta_segment_cut"]:
            for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
                all_image_paths.append(image_path)

        padding = 10
        # Найти общий bounding box для всех изображений
        global_size = find_global_size(all_image_paths, padding)

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

    # all_image_paths = []
    # for sub_dir in dir_structure["crop_nii_image"]:
    #     for case in os.listdir(os.path.join(crop_nii_image_path, sub_dir)):
    #         image_path = os.path.join(crop_nii_image_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"crop_nii_image {find_shape_2(all_image_paths)}", "work_logger")
    #
    # all_image_paths = []
    # for sub_dir in dir_structure["crop_markers_mask"]:
    #     for case in os.listdir(os.path.join(crop_markers_mask_path, sub_dir)):
    #         image_path = os.path.join(crop_markers_mask_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"crop_markers_mask {find_shape_2(all_image_paths)}", "work_logger")

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
                            dict_case = {10: 491, 9: 499, 8: 498, 7: 497, 6: 496, 5: 495, 4: 494})

    if not controller_dump["aorta_mask_analysis"]:
        mask_analysis(data_path, result_path, type_mask="aortic_valve")

    # slices_with_markers(
    #     nii_path=data_path + 'nii_resample/' + dir_structure['nii_resample'][0] + '/' + test_case_name + '.nii',
    #     case_info=dict_all_case[test_case_name],
    #     save_path=data_path + 'markers_visual/' + dir_structure['markers_visual'][0] + '/' + test_case_name)
    #
    # totalsegmentator(
    #     input=data_path + 'nii_resample/' + dir_structure['nii_resample'][0] + '/' + test_case_name + '.nii',
    #     output=data_path + 'totalsegmentator_result/' + dir_structure['totalsegmentator_result'][0] + '/' + test_case_name,
    #     task="class_map_part_cardiac")

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

    if data_path:
        free_cpus = get_free_cpus()
        current_time = datetime.now()
        log_file_name = current_time.strftime("log_%H_%M__%d_%m_%Y.log")
        log_path = os.path.join(data_path, log_file_name)
        logging.basicConfig(level=logging.INFO, filename=log_path, filemode="w")
        result_file_name = current_time.strftime("result_%H_%M__%d_%m_%Y.log")
        result_path = os.path.join(data_path, result_file_name)
        logging.basicConfig(level=logging.INFO, filename=result_path, filemode="w")

        # Логгер для хода программы
        log_file_name = current_time.strftime("log_%H_%M__%d_%m_%Y.log")
        work_log_path = os.path.join(data_path, log_file_name)
        work_logger = logging.getLogger("work_logger")
        work_logger.setLevel(logging.INFO)
        work_handler = logging.FileHandler(work_log_path, mode='w')
        work_handler.setFormatter(logging.Formatter('%(message)s'))
        work_logger.addHandler(work_handler)

        # Логгер для результатов
        result_file_name = current_time.strftime("result_%H_%M__%d_%m_%Y.log")
        result_log_path = os.path.join(data_path, result_file_name)
        result_logger = logging.getLogger("result_logger")
        result_logger.setLevel(logging.INFO)
        result_handler = logging.FileHandler(result_log_path, mode='w')
        result_handler.setFormatter(logging.Formatter('%(message)s'))  # без времени
        result_logger.addHandler(result_handler)

        # Очистим базовые хендлеры у root-логгера (если есть)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        controller(data_path, cpus=free_cpus)

