import os
import logging
import shutil
from pathlib import Path
import platform

from statistics import mode
from datetime import datetime

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

from data_preprocessing.dcm_nii_converter import convert_dcm_to_nii, resample_nii, reader_dcm
from data_preprocessing.txt_json_converter import txt_json_convert
from data_preprocessing.stl_nii_converter import convert_stl_to_mask_nii, cut_mask_using_points
from data_preprocessing.check_structure import create_directory_structure
from data_preprocessing.json_worker import json_reader, json_save
from data_preprocessing.log_worker import add_info_logging
from data_preprocessing.crop_nii import cropped_image, find_global_bounds, find_shape, find_shape_2
from models.implementation_nnUnet import nnUnet_trainer
from data_visualization.markers import slices_with_markers, process_markers

from models.implementation_3D_Unet import WrapperUnet

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def controller(data_path, nnUNet_folder_name):
    def clear_folder(folder_path):
        """Очищает папку, удаляя все файлы и подпапки"""
        folder = Path(folder_path)
        if not folder.exists():
            add_info_logging(f"Папка '{folder_path}' не существует.")
            return

        for item in folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # Удаляем файл или символическую ссылку
            elif item.is_dir():
                shutil.rmtree(item)  # Удаляем папку рекурсивно

    script_dir = Path(__file__).resolve().parent

    dicom_path = os.path.join(data_path, "dicom")
    json_marker_path = os.path.join(data_path, "json_markers_info")
    txt_marker_path = os.path.join(data_path, "markers_info")
    stl_aorta_segment_path = os.path.join(data_path, "stl_aorta_segment")
    mask_aorta_segment_path = os.path.join(data_path, "mask_aorta_segment")
    nii_resample_path = os.path.join(data_path, "nii_resample")
    nii_convert_path = os.path.join(data_path, "nii_convert")
    mask_aorta_segment_cut_path = os.path.join(data_path, "mask_aorta_segment_cut")
    mask_markers_visual_path = os.path.join(data_path, "markers_visual")
    nnUNet_folder = os.path.join(data_path, nnUNet_folder_name)
    current_dataset_path = os.path.join(nnUNet_folder, "nnUNet_raw", "Dataset401_AorticValve")
    crop_nii_image_path = os.path.join(data_path, "crop_nii_image")
    crop_markers_mask_path = os.path.join(data_path, "crop_markers_mask")
    UNet_3D_folder = os.path.join(data_path, "3DUNet_folder")

    controller_path = os.path.join(script_dir, "controller.json")
    data_structure_path = os.path.join(script_dir, "dir_structure.json")
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")

    dict_all_case = {}
    if os.path.isfile(controller_path):
        controller_dump = json_reader(controller_path)
    else:
        controller_dump = {}
    dir_structure = json_reader(data_structure_path)
    create_directory_structure(data_path, dir_structure)

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
        json_save(controller_dump, controller_path)

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

        all_shapes = []
        for sub_dir in dir_structure["nii_convert"]:
            for case in os.listdir(os.path.join(nii_convert_path, sub_dir)):
                image_path = os.path.join(nii_convert_path, sub_dir, case)
                all_shapes.append(find_shape(image_path))
        # add_info_logging(f"all_shapes {set(all_shapes)}")

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
                resample_nii(nii_convert_case_file_path,
                             nii_resample_case_file_path,
                             [most_img_spac_0, most_img_spac_1, most_img_spac_2])
        controller_dump["resample"] = True
        json_save(controller_dump, controller_path)

    # all_image_paths = []
    # for sub_dir in dir_structure["nii_resample"]:
    #     for case in os.listdir(os.path.join(nii_resample_path, sub_dir)):
    #         image_path = os.path.join(nii_resample_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"nii_resample {find_shape_2(all_image_paths)}")

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
        json_save(controller_dump, controller_path)
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
        json_save(controller_dump, controller_path)

    # all_image_paths = []
    # for sub_dir in dir_structure["mask_aorta_segment"]:
    #     for case in os.listdir(os.path.join(mask_aorta_segment_path, sub_dir)):
    #         image_path = os.path.join(mask_aorta_segment_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"mask_aorta_segment {find_shape_2(all_image_paths)}")

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
        json_save(controller_dump, controller_path)

    if not "mask_markers_create" in controller_dump.keys() or not controller_dump["mask_markers_create"]:
        for sub_dir in list(dir_structure["nii_resample"]):
            clear_folder(os.path.join(mask_markers_visual_path, sub_dir))
            for case in os.listdir(os.path.join(nii_resample_path, sub_dir)):
                case_name = case[:-7]
                radius = 10
                nii_resample_case_file_path = os.path.join(nii_resample_path, sub_dir, case)
                mask_markers_img_path = os.path.join(mask_markers_visual_path, sub_dir, f"{case_name}.nii.gz")
                process_markers(nii_resample_case_file_path,
                                dict_all_case[case_name],
                                mask_markers_img_path,
                                radius)
        controller_dump["mask_markers_create"] = True
        json_save(controller_dump, controller_path)

    if (not "create_nnU_Net_data_base" in controller_dump.keys()
            or not controller_dump["create_nnU_Net_data_base"]):
        clear_folder(os.path.join(current_dataset_path, "imagesTr"))
        clear_folder(os.path.join(current_dataset_path, "labelsTr"))
        clear_folder(os.path.join(current_dataset_path, "imagesTs"))
        for sub_dir in list(dir_structure["nii_resample"]):
            file_count = len([f for f in os.listdir(os.path.join(nii_resample_path, sub_dir))])
            n = 0
            for case in os.listdir(os.path.join(nii_resample_path, sub_dir)):
                case_name = case[:-7]
                if int(file_count*0.8) >= n:
                    shutil.copy(str(os.path.join(nii_resample_path, sub_dir, case)),
                                str(os.path.join(current_dataset_path, "imagesTr", f"{case_name}_0000.nii.gz")))
                    shutil.copy(str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
                                str(os.path.join(current_dataset_path, "labelsTr", f"{case}.gz")))
                else:
                    shutil.copy(str(os.path.join(nii_resample_path, sub_dir, case)),
                                str(os.path.join(current_dataset_path, "imagesTs", f"{case_name}_0000.nii.gz")))
                n += 1
        controller_dump["create_nnU_Net_data_base"] = True
        json_save(controller_dump, controller_path)

    if os.path.exists(current_dataset_path):
        if not os.path.isfile(os.path.join(current_dataset_path, "dataset.json")):
            file_count = len([f for f in os.listdir(os.path.join(current_dataset_path, "imagesTr"))])
            generate_dataset_json(current_dataset_path,
                                  channel_names={0: 'CT'},
                                  labels={'background': 0, 'aortic_valve': 1},
                                  num_training_cases=file_count,
                                  file_ending='.nii.gz')
            controller_dump["create_nnU_Net_dataset_json"] = True
            json_save(controller_dump, controller_path)
    else:
        add_info_logging("No folder to save to dataset.json")
        return

    # test_case_name = list(dict_all_case.keys())[0]

    # model_nnUnet = nnUnet_trainer(nnUNet_folder)
    # model_nnUnet.train_nnUnet(task_id=401, nnUnet_path=nnUNet_folder)
    # all_image_paths = []
    # for sub_dir in dir_structure["mask_aorta_segment_cut"]:
    #     for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
    #         image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
    #         all_image_paths.append(image_path)
    # add_info_logging(f"mask_aorta_segment_cut {find_shape_2(all_image_paths)}")

    if not "crop_images" in controller_dump.keys() or not controller_dump["crop_images"]:
        # Получаем все пути к изображениям в папке mask_aorta_segment_cut
        all_image_paths = []
        for sub_dir in dir_structure["mask_aorta_segment_cut"]:
            for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
                all_image_paths.append(image_path)

        padding = 16
        # 🔹 1. Найти общий bounding box для всех изображений
        global_bounds = find_global_bounds(all_image_paths, padding)

        for sub_dir in list(dir_structure["mask_aorta_segment_cut"]):
            clear_folder(os.path.join(crop_nii_image_path, sub_dir))
            clear_folder(os.path.join(crop_markers_mask_path, sub_dir))
            for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
                cropped_image(input_image_path=str(os.path.join(nii_resample_path, sub_dir, case)),
                              output_image_path=str(os.path.join(crop_nii_image_path, sub_dir, case)),
                              bounds=global_bounds)
                cropped_image(input_image_path=str(os.path.join(mask_markers_visual_path, sub_dir, case)),
                              output_image_path=str(os.path.join(crop_markers_mask_path, sub_dir, case)),
                              bounds=global_bounds)
        controller_dump["crop_images"] = True
        json_save(controller_dump, controller_path)

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
        json_save(controller_dump, controller_path)

    model_3D_Unet = WrapperUnet()
    model_3D_Unet.try_unet3d_training(UNet_3D_folder)
    model_3D_Unet.try_unet3d_testing(UNet_3D_folder)

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
        current_time = datetime.now()
        filename = current_time.strftime("log_%Y_%m_%d_%H_%M.log")
        log_path = os.path.join(data_path, filename)
        logging.basicConfig(level=logging.INFO, filename=log_path, filemode="w")
        nnUNet_folder = "nnUNet_folder"
        # os.environ["nnUNet_raw"] = nnUNet_folder + "nnUNet_raw/"
        # os.environ["nnUNet_preprocessed"] = nnUNet_folder + "nnUNet_preprocessed/"
        # os.environ["nnUNet_results"] = nnUNet_folder + "nnUNet_results/"
        controller(data_path, nnUNet_folder)

