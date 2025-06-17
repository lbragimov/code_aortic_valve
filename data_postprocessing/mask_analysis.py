import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from data_postprocessing.evaluation_analysis import  evaluate_segmentation
from data_preprocessing.text_worker import add_info_logging


def mask_comparison(data_path, type_mask, folder_name):
    nnUNet_folder = os.path.join(data_path, "nnUNet_folder")
    result_mask_folder = os.path.join(nnUNet_folder, "nnUNet_test", folder_name)
    original_mask_folder = os.path.join(nnUNet_folder, "original_mask", folder_name)

    per_case_data = []

    for case in os.listdir(result_mask_folder):
        if not case.endswith(".nii.gz"):
            continue
        case_name = case[:-7]
        first_char = case_name[0]

        result_mask_path = os.path.join(result_mask_folder, case)
        original_mask_path = os.path.join(original_mask_folder, f"{case_name}.nii.gz")

        if not os.path.exists(original_mask_path):
            add_info_logging(f"Missing {case_name} - no original mask", "work_logger")
            continue

        result_mask = nib.load(result_mask_path).get_fdata()
        mask_img = nib.load(original_mask_path).get_fdata()

        if result_mask.shape != mask_img.shape:
            add_info_logging(f"The dimensions do not match for the case: {case_name}", "work_logger")
            add_info_logging(f"result_mask shape:  {result_mask.shape}", "work_logger")
            add_info_logging(f"original_mask shape:{mask_img.shape}", "work_logger")
            continue  # пропустить

        try:
            metrics = evaluate_segmentation(mask_img, result_mask)
            # Сохраняем по кейсу
            metrics["case"] = case_name
            metrics["group"] = first_char
            per_case_data.append(metrics)
        except Exception as e:
            add_info_logging(f"Error while comparing {case_name}: {str(e)}", "work_logger")

    return per_case_data


class LandmarkCentersCalculator:

    @staticmethod
    def _compute_center_of_mass(binary_mask, spacing, origin, direction):
        # Function to compute center of mass in world coordinates
        indices = np.argwhere(binary_mask)  # Get voxel indices of the mask
        if len(indices) == 0:
            return None  # No center of mass if mask is empty

        # Compute the mean position in voxel space
        center_voxel = np.mean(indices, axis=0)[::-1]  # Reverse order (Z, Y, X) -> (X, Y, Z)

        # Convert to world coordinates using the corrected direction matrix
        center_world = np.dot(direction, center_voxel * spacing) + origin
        return center_world

    def extract_landmarks_com_nii(self, mask_nii):
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
            center_world = self._compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        return centers_of_mass

    def extract_landmarks_com_npz(self, mask_nii, mask_npz):
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
            center_world = self._compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        return centers_of_mass

    @staticmethod
    def extract_landmarks_peak_npz(mask_nii, mask_npz):
        # Get image metadata
        mask_image = sitk.ReadImage(mask_nii)
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z)
        origin = np.array(mask_image.GetOrigin())  # (x0, y0, z0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)

        prob_map_all = np.load(mask_npz)
        labels = len(prob_map_all["probabilities"])

        # Compute peak voxel for each label
        peaks_world = {}
        for label in range(1, labels):
            prob_map = prob_map_all["probabilities"][label]
            if np.max(prob_map) == 0:
                continue  # Skip empty maps

            # Find voxel index of the maximum probability
            peak_index = np.unravel_index(np.argmax(prob_map), prob_map.shape)  # (z, y, x)

            # Convert index to physical coordinates
            peak_voxel = np.array(peak_index)[::-1]  # Convert to (x, y, z) from (z, y, x)
            peak_world = np.dot(direction, peak_voxel * spacing) + origin

            peaks_world[label] = peak_world

        return peaks_world

    @staticmethod
    def extract_landmarks_topk_peaks_npz(mask_nii, mask_npz, top_k=3):
        # Get image metadata
        mask_image = sitk.ReadImage(mask_nii)
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z)
        origin = np.array(mask_image.GetOrigin())  # (x0, y0, z0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)

        prob_map_all = np.load(mask_npz)
        labels = len(prob_map_all["probabilities"])

        # Compute top-k peak voxels for each label
        peaks_world = {}
        for label in range(1, labels):
            prob_map = prob_map_all["probabilities"][label]
            flat = prob_map.flatten()
            if np.max(flat) == 0:
                continue  # Skip empty maps

            # Get indices of top_k highest probabilities
            topk_indices_flat = np.argpartition(-flat, range(min(top_k, flat.size)))[:top_k]
            topk_indices_sorted = topk_indices_flat[np.argsort(-flat[topk_indices_flat])]

            peak_world_coords = []
            for idx in topk_indices_sorted:
                peak_index = np.unravel_index(idx, prob_map.shape)  # (z, y, x)
                peak_voxel = np.array(peak_index)[::-1]  # to (x, y, z)
                peak_world = np.dot(direction, peak_voxel * spacing) + origin
                peak_world_coords.append(peak_world)

            peaks_world[label] = peak_world_coords

        return peaks_world  # Dict[label] = [coord1, coord2, ..., coordK]
