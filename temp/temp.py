from data_preprocessing.text_worker import add_info_logging
import os
from pathlib import Path
import json
import numpy as np
from pycpd import AffineRegistration



def controller(data_path):
    ds_folder_name = "Dataset499_AortaLandmarks"

    # if controller_dump.get("crop_img_size"):
    #     global_size = controller_dump["crop_img_size"]
    # else:
    #     all_image_paths = []
    #     for sub_dir in dir_structure["mask_aorta_segment_cut"]:
    #         for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
    #             image_path = os.path.join(mask_aorta_segment_cut_path, sub_dir, case)
    #             all_image_paths.append(image_path)
    #
    #     padding = 10
    #     # Найти общий bounding box для всех изображений
    #     global_size = find_global_size(all_image_paths, padding)
    #     controller_dump["crop_img_size"] = [int(x) for x in global_size]
    #     yaml_save(controller_dump, controller_path)
    #
    # for sub_dir in list(dir_structure["mask_aorta_segment_cut"]):
    #     clear_folder(os.path.join(crop_nii_image_path, sub_dir))
    #     clear_folder(os.path.join(crop_markers_mask_path, sub_dir))
    #     for case in os.listdir(os.path.join(mask_aorta_segment_cut_path, sub_dir)):
    #         cropped_image(mask_image_path=str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
    #                       input_image_path=str(os.path.join(nii_resample_path, sub_dir, case)),
    #                       output_image_path=str(os.path.join(crop_nii_image_path, sub_dir, case)),
    #                       size=global_size)
    #         cropped_image(mask_image_path=str(os.path.join(mask_aorta_segment_cut_path, sub_dir, case)),
    #                       input_image_path=str(os.path.join(mask_markers_visual_path, sub_dir, case)),
    #                       output_image_path=str(os.path.join(crop_markers_mask_path, sub_dir, case)),
    #                       size=global_size)
    # controller_dump["crop_images"] = True
    # yaml_save(controller_dump, controller_path)

    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)