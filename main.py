import json
import os

from data_preprocessing.dcm_nii_converter import convert_dcm_to_nii


def controller(data_path, dir_structure: dict):
    dict_all_case_json = data_path + "/dict_all_case.json"
    if os.path.isfile(dict_all_case_json):
        with open(dict_all_case_json, 'r') as read_file:
            dict_all_case = json.load(read_file)
    else:
        dict_all_case = {}
        dicom_path = data_path + "dicom/"
        for sub_dir in list(dir_structure['dicom']):
            for case in os.listdir(dicom_path + sub_dir):
                dcm_case_path = data_path + "dicom/" + sub_dir + "/" + case
                nii_convert_case_file = data_path + "nii_convert/" + sub_dir + "/" + case
                img_size, img_origin, img_spacing, img_direction = convert_dcm_to_nii(dcm_case_path,
                                                                                      nii_convert_case_file)
                dict_all_case[case] = {
                    "img_size": img_size,
                    "img_origin": img_origin,
                    "img_spacing": img_spacing,
                    "img_direction": img_direction
                }

        with open(dict_all_case_json, 'w') as json_file:
            json.dump(dict_all_case, json_file)
    print('hi')


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    data_structure = data_path + "dir_structure.json"

    with open(data_structure, 'r') as read_file:
        dir_structure = json.load(read_file)

    controller(data_path, dir_structure)

