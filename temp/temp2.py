import os
import numpy as np
import nibabel as nib

def split_mask_into_labels(image_file_name, folder_path):
    # Load mask
    case_name = image_file_name.split(".")[0]
    mask_img = nib.load(folder_path + image_file_name)
    mask_data = mask_img.get_fdata()
    affine = mask_img.affine
    header = mask_img.header

    # # Create output folder
    # os.makedirs(output_folder, exist_ok=True)

    # Find unique labels except background 0
    labels = np.unique(mask_data)
    labels = labels[labels != 0]

    print(f"Found labels: {labels}")

    for label in labels:
        # Create binary mask for this label
        label_mask = (mask_data == label).astype(np.uint8)

        # Save output
        out_path = os.path.join(folder_path, f"{case_name}_{int(label)}.nii.gz")
        out_img = nib.Nifti1Image(label_mask, affine, header)
        nib.save(out_img, out_path)

        print(f"Saved: {out_path}")

    print("Done!")


if __name__ == "__main__":
    folder_path = "C:/Users/Kamil/Aortic_valve/meet 4/Seg3D/g11/"
    img_name = "pred_g11.nii.gz"
    split_mask_into_labels(image_file_name=img_name, folder_path=folder_path)