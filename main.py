import json
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
from models.implementation_nnUnet import nnUnet_trainer
from data_visualization.markers import slices_with_markers

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def controller(data_path, nnUNet_folder):

    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")
    dict_all_case = {}

    controller_path = os.path.join(data_path, "controller.json")
    if os.path.isfile(controller_path):
        with open(controller_path, 'r') as read_file:
            controller_dump = json.load(read_file)
    else:
        controller_dump = {}

    data_structure_path = os.path.join(data_path, "dir_structure.json")

    with open(data_structure_path, 'r') as read_file:
        dir_structure = json.load(read_file)

    create_directory_structure(data_path, dir_structure)


    if not "convert" in controller_dump.keys() or not controller_dump["convert"]:
        dicom_path = os.path.join(data_path, "dicom")
        for sub_dir in list(dir_structure['dicom']):
            for case in os.listdir(dicom_path + sub_dir):
                dcm_case_path = os.path.join(data_path, "dicom", sub_dir, case)
                if sub_dir == "Homburg pathology":
                    case = case[:-3]
                nii_convert_case_file = os.path.join(data_path, "nii_convert", sub_dir, case)
                img_size, img_origin, img_spacing, img_direction = convert_dcm_to_nii(dcm_case_path,
                                                                                      nii_convert_case_file)
                dict_all_case[case] = {
                    "img_size": img_size,
                    "img_origin": img_origin,
                    "img_spacing": img_spacing,
                    "img_direction": img_direction
                }

        with open(dict_all_case_path, 'w') as json_file:
            json.dump(dict_all_case, json_file)

        controller_dump["convert"] = True
        controller_dump["resample"] = False
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    if not dict_all_case:
        if os.path.isfile(dict_all_case_path):
            with open(dict_all_case_path, 'r') as read_file:
                dict_all_case = json.load(read_file)
        else:
            dicom_path = os.path.join(data_path, "dicom")
            for sub_dir in list(dir_structure['dicom']):
                for case in os.listdir(dicom_path + sub_dir):
                    dcm_case_path = os.path.join(data_path, "dicom", sub_dir, case)
                    if sub_dir == "Homburg pathology":
                        case = case[:-3]
                    img_size, img_origin, img_spacing, img_direction = reader_dcm(dcm_case_path)

                    dict_all_case[case] = {
                        "img_size": img_size,
                        "img_origin": img_origin,
                        "img_spacing": img_spacing,
                        "img_direction": img_direction
                    }

            json_marker_path = os.path.join(data_path, "json_markers_info")
            if os.path.exists(json_marker_path):
                for sub_dir in list(dir_structure["json_markers_info"]):
                    for case in os.listdir(json_marker_path + sub_dir):
                        json_marker_case_file = os.path.join(json_marker_path, sub_dir, case)
                        case_name = case[:-5]
                        with open(json_marker_case_file, "r") as file:
                            data = json.load(file)

                        dict_all_case[case_name] |= {
                            "R": data["R"],
                            "L": data["L"],
                            "N": data["N"],
                            "RLC": data["RLC"],
                            "RNC": data["RNC"],
                            "LNC": data["LNC"]
                        }

            with open(dict_all_case_path, 'w') as json_file:
                json.dump(dict_all_case, json_file)

    if not "resample" in controller_dump.keys() or not controller_dump["resample"]:

        # Extract the first elements of "img_spacing" and store them in a list
        img_spac_0 = [case['img_spacing'][0] for case in dict_all_case.values()]
        # Find the minimum value and the average of the first elements
        min_img_spac_0 = min(img_spac_0)
        max_img_spac_0 = max(img_spac_0)
        avg_img_spac_0 = sum(img_spac_0) / len(img_spac_0)
        most_img_spac_0 = float(mode(img_spac_0))

        # Extract the first elements of "img_spacing" and store them in a list
        img_spac_1 = [case['img_spacing'][1] for case in dict_all_case.values()]
        # Find the minimum value and the average of the first elements
        min_img_spac_1 = min(img_spac_1)
        max_img_spac_1 = max(img_spac_1)
        avg_img_spac_1 = sum(img_spac_1) / len(img_spac_1)
        most_img_spac_1 = float(mode(img_spac_1))

        # Extract the first elements of "img_spacing" and store them in a list
        img_spac_2 = [case['img_spacing'][2] for case in dict_all_case.values()]
        # Find the minimum value and the average of the first elements
        min_img_spac_2 = min(img_spac_2)
        max_img_spac_2 = max(img_spac_2)
        avg_img_spac_2 = sum(img_spac_2) / len(img_spac_2)
        most_img_spac_2 = float(mode(img_spac_2))

        nii_convert_path = data_path + "nii_convert/"
        for sub_dir in list(dir_structure["nii_convert"]):
            for case in os.listdir(nii_convert_path + sub_dir):
                nii_convert_case_file_path = os.path.join(data_path, "nii_convert", sub_dir, case)
                nii_resample_case_file_path = os.path.join(data_path, "nii_resample", sub_dir, case)
                resample_nii(nii_convert_case_file_path,
                             nii_resample_case_file_path,
                             [most_img_spac_0, most_img_spac_1, most_img_spac_2])
        controller_dump["resample"] = True
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    if not "markers_info" in controller_dump.keys() or not controller_dump["markers_info"]:
        txt_marker_path = os.path.join(data_path, "markers_info")
        json_marker_path = os.path.join(data_path, "json_markers_info")
        for sub_dir in list(dir_structure["markers_info"]):
            for case in os.listdir(txt_marker_path + sub_dir):
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
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)
        with open(dict_all_case_path, 'w') as json_file:
            json.dump(dict_all_case, json_file)

    # if (not "stl_aorta_segment_resample" in controller_dump.keys()
    #         or not controller_dump["stl_aorta_segment_resample"]):
    #     stl_aorta_segment_path = data_path + "stl_aorta_segment/"
    #     stl_aorta_segment_resample_path = data_path + "stl_aorta_segment_resample/"
    #     for sub_dir in list(dir_structure["stl_aorta_segment_resample"]):
    #         for case in os.listdir(stl_aorta_segment_path + sub_dir):
    #             stl_aorta_segment_file =stl_aorta_segment_path + sub_dir + "/" + case
    #             stl_aorta_segment_resample_file = stl_aorta_segment_resample_path + sub_dir + "/" + case
    #
    #             stl_resample(stl_aorta_segment_file,
    #                          stl_aorta_segment_resample_file,
    #                          5000)
    #
    #
    #     controller_dump["stl_aorta_segment_resample"] = True
    #     with open(controller_path, 'w') as json_file:
    #         json.dump(controller_dump, json_file)

    if not "mask_aorta_segment" in controller_dump.keys() or not controller_dump["mask_aorta_segment"]:
        # stl_aorta_segment_path = data_path + "stl_aorta_segment_resample/"
        stl_aorta_segment_path = os.path.join(data_path, "stl_aorta_segment")
        mask_aorta_segment_path = os.path.join(data_path, "mask_aorta_segment")
        nii_resample_path = os.path.join(data_path, "nii_resample")
        for sub_dir in list(dir_structure["stl_aorta_segment"]):
            for case in os.listdir(stl_aorta_segment_path + sub_dir):
                stl_aorta_segment_file = os.path.join(stl_aorta_segment_path, sub_dir, case)
                case_name = case[:-4]
                mask_aorta_segment_file = os.path.join(mask_aorta_segment_path, sub_dir, f"{case_name}.nii")
                nii_resample_file = os.path.join(nii_resample_path, sub_dir, f"{case_name}.nii")

                convert_stl_to_mask_nii(stl_aorta_segment_file,
                                        nii_resample_file,
                                        mask_aorta_segment_file)

        controller_dump["mask_aorta_segment"] = True
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    if not "mask_aorta_segment_cut" in controller_dump.keys() or not controller_dump["mask_aorta_segment_cut"]:
        mask_aorta_segment_path = os.path.join(data_path, "mask_aorta_segment")
        mask_aorta_segment_cut_path = os.path.join(data_path, "mask_aorta_segment_cut")
        for sub_dir in list(dir_structure["stl_aorta_segment"]):
            for case in os.listdir(mask_aorta_segment_path + sub_dir):
                logging.info(f"{sub_dir} + {case}")
                mask_aorta_segment_file = os.path.join(mask_aorta_segment_path, sub_dir, case)
                mask_aorta_segment_cut_file = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
                case_name = case[:-4]
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
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    nnUNet_folder = os.path.join(data_path, dir_structure['nnUNet_folder'])
    if (not "create_nnU_Net_data_base" in controller_dump.keys()
            or not controller_dump["create_nnU_Net_data_base"]):

        mask_aorta_segment_cut_path = os.path.join(data_path, "mask_aorta_segment_cut")
        nii_resample_path = os.path.join(data_path, "nii_resample")

        for sub_dir in list(dir_structure["nii_resample"]):
            file_count = len([f for f in os.listdir(nii_resample_path + sub_dir)])
            n = 0
            for case in os.listdir(nii_resample_path + sub_dir):
                case_name = case[:-4]
                if int(file_count*0.8) >= n:
                    shutil.copy(str(os.path.join(nii_resample_path, sub_dir, case)),
                                str(os.path.join(nnUNet_folder, "imagesTr", f"{case_name}_0000.nii.gz")))
                    shutil.copy(str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
                                str(os.path.join(nnUNet_folder, "labelsTr", f"{case}.gz")))
                else:
                    shutil.copy(str(os.path.join(nii_resample_path, sub_dir, case)),
                                str(os.path.join(nnUNet_folder, "imagesTs", f"{case_name}_0000.nii.gz")))
                n += 1

        controller_dump["create_nnU_Net_data_base"] = True
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    current_dataset_path = os.path.join(nnUNet_folder, "nnUNet_raw", "Dataset401_AorticValve")
    if os.path.exists(current_dataset_path):
        if not os.path.isfile(os.path.join(current_dataset_path, "dataset.json")):

            file_count = len([f for f in os.listdir(os.path.join(current_dataset_path, "imagesTr"))])

            generate_dataset_json(current_dataset_path,
                                  channel_names={0: 'CT'},
                                  labels={'background': 0, 'aortic_valve': 1},
                                  num_training_cases=file_count,
                                  file_ending='.nii.gz')

            controller_dump["create_nnU_Net_dataset_json"] = True
            with open(controller_path, 'w') as json_file:
                json.dump(controller_dump, json_file)
    else:
        current_time = datetime.now()
        str_time = current_time.strftime("%d:%H:%M")
        logging.info(f"time:  {str_time} No folder to save to dataset.json")
        return

    # test_case_name = list(dict_all_case.keys())[0]

    model_nnUnet = nnUnet_trainer(data_path + nnUNet_folder)
    model_nnUnet.train_nnUnet(task_id=401, nnUnet_path=data_path + nnUNet_folder)

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

    if current_os == "Windows":
        # data_path = "C:/Users/Kamil/Aortic_valve/data/"
        data_path = "C:/Users/Kamil/Aortic_valve/data/temp"
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

