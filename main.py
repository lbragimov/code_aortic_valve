import json
import os

from statistics import mode

from data_preprocessing.dcm_nii_converter import convert_dcm_to_nii, resample_nii


def controller(data_path):

    dict_all_case_path = data_path + "/dict_all_case.json"

    controller_path = data_path + "controller.json"
    if os.path.isfile(controller_path):
        with open(controller_path, 'r') as read_file:
            controller_dump = json.load(read_file)
    else:
        controller_dump = {}

    data_structure_path = data_path + "dir_structure.json"

    with open(data_structure_path, 'r') as read_file:
        dir_structure = json.load(read_file)

    if os.path.isfile(dict_all_case_path):
        with open(dict_all_case_path, 'r') as read_file:
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

        with open(dict_all_case_path, 'w') as json_file:
            json.dump(dict_all_case, json_file)

        controller_dump["resample"] = False
        with open(controller_path, 'w') as json_file:
            json.dump(controller_dump, json_file)

    # Extract the first elements of "img_spacing" and store them in a list
    img_spac_0 = [case['img_spacing'][0] for case in dict_all_case.values()]
    # Find the minimum value and the average of the first elements
    min_img_spac_0 = min(img_spac_0)
    max_img_spac_0 = max(img_spac_0)
    avg_img_spac_0 = sum(img_spac_0) / len(img_spac_0)
    most_img_spac_0 = mode(img_spac_0)

    # Extract the first elements of "img_spacing" and store them in a list
    img_spac_1 = [case['img_spacing'][1] for case in dict_all_case.values()]
    # Find the minimum value and the average of the first elements
    min_img_spac_1 = min(img_spac_1)
    max_img_spac_1 = max(img_spac_1)
    avg_img_spac_1 = sum(img_spac_1) / len(img_spac_1)
    most_img_spac_1 = mode(img_spac_1)

    # Extract the first elements of "img_spacing" and store them in a list
    img_spac_2 = [case['img_spacing'][2] for case in dict_all_case.values()]
    # Find the minimum value and the average of the first elements
    min_img_spac_2 = min(img_spac_2)
    max_img_spac_2 = max(img_spac_2)
    avg_img_spac_2 = sum(img_spac_2) / len(img_spac_2)
    most_img_spac_2 = mode(img_spac_2)

    if not "resample" in controller_dump.keys() or not controller_dump["resample"]:
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
    print('hi')


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)

