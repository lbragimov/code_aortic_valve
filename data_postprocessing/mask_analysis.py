import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from data_postprocessing.evaluation_analysis import  evaluate_segmentation
from data_preprocessing.text_worker import add_info_logging


def mask_comparison(data_path, type_mask, folder_name):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    if type_mask == "aortic_valve":
        result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
        original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)
    elif type_mask == "aoric_landmarks":
        result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
        original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)

    dice_scores = []
    iou_scores = []

    for case in os.listdir(result_mask_folder):
        if not case.endswith(".nii.gz"):
            continue
        case_name = case[:-7]
        result_mask_path = os.path.join(result_mask_folder, case)
        original_mask_path = os.path.join(original_mask_folder, f"{case_name}.nii.gz")

        if not os.path.exists(original_mask_path):
            add_info_logging(f" Пропущен {case_name} — нет оригинальной маски", "work_logger")
            continue

        result_mask = nib.load(result_mask_path).get_fdata()
        mask_img = nib.load(original_mask_path).get_fdata()

        if result_mask.shape != mask_img.shape:
            add_info_logging(f" Размеры не совпадают в кейсе: {case_name}", "work_logger")
            add_info_logging(f" result_mask shape:  {result_mask.shape}", "work_logger")
            add_info_logging(f" original_mask shape:{mask_img.shape}", "work_logger")
            continue  # пропустить

        try:
            metrics = evaluate_segmentation(mask_img, result_mask)
            dice_scores.append(metrics["Dice"])
            iou_scores.append(metrics["IoU"])
        except Exception as e:
            add_info_logging(f" Ошибка при сравнении {case_name}: {str(e)}", "work_logger")

    return {
        "Dice": dice_scores,
        "IoU": iou_scores
    }


class LandmarkCentersCalculator:

    def compute_center_of_mass(self, binary_mask, spacing, origin, direction):
        # Function to compute center of mass in world coordinates
        indices = np.argwhere(binary_mask)  # Get voxel indices of the mask
        if len(indices) == 0:
            return None  # No center of mass if mask is empty

        # Compute the mean position in voxel space
        center_voxel = np.mean(indices, axis=0)[::-1]  # Reverse order (Z, Y, X) -> (X, Y, Z)

        # Convert to world coordinates using the corrected direction matrix
        center_world = np.dot(direction, center_voxel * spacing) + origin
        return center_world

    def compute_metrics_direct_nii(self, mask_nii):
        mask_image = sitk.ReadImage(mask_nii)
        mask_array = sitk.GetArrayFromImage(mask_image)  # Convert to NumPy array

        # Get image metadata
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z) voxel size
        origin = np.array(mask_image.GetOrigin())  # World coordinate of (0,0,0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)  # Reshape to 3x3 matrix

        # Find unique labels (excluding background 0)
        labels = np.unique(mask_array)
        labels = labels[labels != 0]  # Remove background if label 0 exists

        # Compute center of mass for each label
        centers_of_mass = {}
        for label in labels:
            binary_mask = (mask_array == label)  # Create binary mask for current label
            center_world = self.compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        # R_land, L_land, N_land, RLC_land, RNC_land, LNC_land
        # measurerer = landmarking_computeMeasurements(centers_of_mass[1], centers_of_mass[2], centers_of_mass[3],
        #                                              centers_of_mass[4], centers_of_mass[5], centers_of_mass[6])
        # metrics = measurerer.compute_metrics()
        return centers_of_mass

    def compute_metrics_direct_npz(self, mask_nii, mask_npz):
        # Get image metadata
        mask_image = sitk.ReadImage(mask_nii)
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z) voxel size
        origin = np.array(mask_image.GetOrigin())  # World coordinate of (0,0,0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)  # Reshape to 3x3 matrix

        prob_map_all = np.load(mask_npz)
        labels = len(prob_map_all["probabilities"])

        # Compute center of mass for each label
        centers_of_mass = {}
        for label in range(1, labels):
            binary_mask = prob_map_all["probabilities"][label]  # Create binary mask for current label
            binary_mask[binary_mask < np.max(binary_mask)*0.2] = 0
            center_world = self.compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        # R_land, L_land, N_land, RLC_land, RNC_land, LNC_land
        # measurerer = landmarking_computeMeasurements(centers_of_mass[1], centers_of_mass[2], centers_of_mass[3],
        #                                              centers_of_mass[4], centers_of_mass[5], centers_of_mass[6])
        # metrics = measurerer.compute_metrics()
        return centers_of_mass
