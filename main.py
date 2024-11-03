import json
import os

from statistics import mode

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

from data_preprocessing.dcm_nii_converter import convert_dcm_to_nii, resample_nii, reader_dcm
from data_preprocessing.txt_json_converter import txt_json_convert
from data_preprocessing.stl_nii_converter import convert_stl_to_nii, stl_to_mask, stl_resample
from data_visualization.markers import slices_with_markers


def controller(data_path):

    dict_all_case_path = data_path + "dict_all_case.json"
    dict_all_case = {}

    controller_path = data_path + "controller.json"
    if os.path.isfile(controller_path):
        with open(controller_path, 'r') as read_file:
            controller_dump = json.load(read_file)
    else:
        controller_dump = {}

    data_structure_path = data_path + "dir_structure.json"

    with open(data_structure_path, 'r') as read_file:
        dir_structure = json.load(read_file)

    if not "convert" in controller_dump.keys() or not controller_dump["convert"]:
        dicom_path = data_path + "dicom/"
        for sub_dir in list(dir_structure['dicom']):
            for case in os.listdir(dicom_path + sub_dir):
                dcm_case_path = data_path + "dicom/" + sub_dir + "/" + case
                if sub_dir == "Homburg pathology":
                    case = case[:-3]
                nii_convert_case_file = data_path + "nii_convert/" + sub_dir + "/" + case
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
            dicom_path = data_path + "dicom/"
            for sub_dir in list(dir_structure['dicom']):
                for case in os.listdir(dicom_path + sub_dir):
                    dcm_case_path = data_path + "dicom/" + sub_dir + "/" + case
                    if sub_dir == "Homburg pathology":
                        case = case[:-3]
                    img_size, img_origin, img_spacing, img_direction = reader_dcm(dcm_case_path)

                    dict_all_case[case] = {
                        "img_size": img_size,
                        "img_origin": img_origin,
                        "img_spacing": img_spacing,
                        "img_direction": img_direction
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
                nii_convert_case_file = data_path + "nii_convert/" + sub_dir + "/" + case
                nii_resample_case_file = data_path + "nii_resample/" + sub_dir + "/" + case
                resample_nii(nii_convert_case_file,
                             nii_resample_case_file,
                             [most_img_spac_0, most_img_spac_1, most_img_spac_2])
        controller_dump["resample"] = True
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    if not "markers_info" in controller_dump.keys() or not controller_dump["markers_info"]:
        txt_marker_path = data_path + "markers_info/"
        json_marker_path = data_path + "json_markers_info/"
        for sub_dir in list(dir_structure["markers_info"]):
            for case in os.listdir(txt_marker_path + sub_dir):
                txt_marker_case_file = txt_marker_path + sub_dir + "/" + case
                case = case[2:-4]
                json_marker_case_file = json_marker_path + sub_dir + "/" + case + ".json"
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

    if (not "stl_aorta_segment_resample" in controller_dump.keys()
            or not controller_dump["stl_aorta_segment_resample"]):
        stl_aorta_segment_path = data_path + "stl_aorta_segment/"
        stl_aorta_segment_resample_path = data_path + "stl_aorta_segment_resample/"
        for sub_dir in list(dir_structure["stl_aorta_segment_resample"]):
            for case in os.listdir(stl_aorta_segment_path + sub_dir):
                stl_aorta_segment_file =stl_aorta_segment_path + sub_dir + "/" + case
                stl_aorta_segment_resample_file = stl_aorta_segment_resample_path + sub_dir + "/" + case

                stl_resample(stl_aorta_segment_file,
                             stl_aorta_segment_resample_file,
                             3000)


        controller_dump["stl_aorta_segment_resample"] = True
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)


    if not "mask_aorta_segment" in controller_dump.keys() or not controller_dump["mask_aorta_segment"]:
        stl_aorta_segment_path = data_path + "stl_aorta_segment_resample/"
        mask_aorta_segment_path = data_path + "mask_aorta_segment/"
        nii_resample_path = data_path + "nii_resample/"
        for sub_dir in list(dir_structure["stl_aorta_segment_resample"]):
            for case in os.listdir(stl_aorta_segment_path + sub_dir):
                stl_aorta_segment_file =stl_aorta_segment_path + sub_dir + "/" + case
                case_name = case[:-4]
                mask_aorta_segment_file = mask_aorta_segment_path + sub_dir + "/" + case_name + ".nii"
                nii_resample_file = nii_resample_path + sub_dir + "/" + case_name + ".nii"

                # convert_stl_to_nii(stl_aorta_segment_file,
                #                    mask_aorta_segment_file,
                #                    tuple(dict_all_case[case_name]['img_size']))

                stl_to_mask(stl_aorta_segment_file,
                            nii_resample_file,
                            mask_aorta_segment_file)


        controller_dump["mask_aorta_segment"] = True
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    test_case_name = list(dict_all_case.keys())[0]

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
    # data_path = "C:/Users/Kamil/Aortic_valve/data/"
    data_path = "D:/science/Aortic_valve/data_short/"
    controller(data_path)

