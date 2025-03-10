import os
import numpy as np
import SimpleITK as sitk


def probability_map_separator(file_original_nii, file_probability_map, output_directory):
    # Load probability maps
    data = np.load(file_probability_map, allow_pickle=True)
    prob_maps = data["probabilities"]  # Shape: (6, H, W, D) for 6 classes

    # Load reference image for metadata (affine, spacing, origin)
    ref_img = sitk.ReadImage(file_original_nii)
    spacing = ref_img.GetSpacing()
    origin = ref_img.GetOrigin()
    direction = ref_img.GetDirection()

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save each probability map as a separate NIfTI file
    for class_idx in range(prob_maps.shape[0]):
        prob_map = prob_maps[class_idx]  # Extract probability map for this class
        sitk_img = sitk.GetImageFromArray(prob_map)  # Convert numpy to SimpleITK image

        # Assign metadata from reference NIfTI
        sitk_img.SetSpacing(spacing)
        sitk_img.SetOrigin(origin)
        sitk_img.SetDirection(direction)

        # Save as NIfTI
        output_path = os.path.join(output_directory, f"class_{class_idx}_prob.nii.gz")
        sitk.WriteImage(sitk_img, output_path)
