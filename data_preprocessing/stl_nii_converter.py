import trimesh
import nibabel as nib
import numpy as np


def convert_stl_to_nii(input_stl_file: str, output_nii_file: str, volume_shape: tuple[int]):
    # Load STL file
    mesh = trimesh.load_mesh(input_stl_file)

    # Create empty volume (e.g. 128x128x128)
    volume = np.zeros(volume_shape)

    # Convert STL coordinates to voxels
    for vertex in mesh.vertices:
        x, y, z = (vertex / mesh.bounds[1] * np.array(volume_shape)).astype(int)
        volume[x, y, z] = 1  # Mark voxel as "occupied"

    # Save in NIfTI format
    nii_img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nii_img, output_nii_file)