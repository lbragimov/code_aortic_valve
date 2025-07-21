import os


def experiment_training(create_img=False, create_models=False):
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