import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path


def _load_image(image_path):
    return sitk.ReadImage(image_path)


def _load_landmarks(dict_landmarks):
    landmarks = []
    list_landmarks_name = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']
    for current_landmarks in list_landmarks_name:
        landmarks.append(tuple(dict_landmarks[current_landmarks]))
    return landmarks


def _world_to_voxel_coords(image, world_coords):
    return [image.TransformPhysicalPointToIndex(point) for point in world_coords]


def _extract_and_plot_slices(image, voxel_coords):
    for idx, (x, y, z) in enumerate(voxel_coords):
        # Extracting the slices
        axial_slice = sitk.GetArrayFromImage(image[:, :, z])
        coronal_slice = sitk.GetArrayFromImage(image[:, y, :])
        sagittal_slice = sitk.GetArrayFromImage(image[x, :, :])

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(axial_slice, cmap='gray')
        axes[0].plot(x, y, 'ro')  # Plotting the landmark
        axes[0].set_title('Axial Slice')

        axes[1].imshow(coronal_slice, cmap='gray')
        axes[1].plot(x, z, 'ro')  # Plotting the landmark
        axes[1].set_title('Coronal Slice')

        axes[2].imshow(sagittal_slice, cmap='gray')
        axes[2].plot(y, z, 'ro')  # Plotting the landmark
        axes[2].set_title('Sagittal Slice')

        plt.suptitle(f'Landmark {idx + 1}')
        plt.show()
        print('hi')


def _extract_and_save_slices(image, voxel_coords, save_path):
    for idx, (x, y, z) in enumerate(voxel_coords):
        # Extracting the slices
        axial_slice = sitk.GetArrayFromImage(image[:, :, z])
        coronal_slice = sitk.GetArrayFromImage(image[:, y, :])
        sagittal_slice = sitk.GetArrayFromImage(image[x, :, :])

        # Prepare figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(axial_slice, cmap='gray')
        axes[0].plot(x, y, 'ro')  # Plotting the landmark
        axes[0].set_title('Axial Slice')

        axes[1].imshow(coronal_slice, cmap='gray')
        axes[1].plot(x, z, 'ro')  # Plotting the landmark
        axes[1].set_title('Coronal Slice')

        axes[2].imshow(sagittal_slice, cmap='gray')
        axes[2].plot(y, z, 'ro')  # Plotting the landmark
        axes[2].set_title('Sagittal Slice')

        # Save the figure
        plt.suptitle(f'Landmark {idx + 1}')

        folder_path = Path(save_path)
        if folder_path.is_dir():
            plt.savefig(save_path + f'/Landmark_{idx + 1}_slices.png')
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path + f'Landmark_{idx + 1}_slices.png')
        plt.close()


def slices_with_markers(nii_path: str, case_info: dict, save_path):
    # Path to your NIFTI file
    # Path to your landmarks file
    # Load the image and landmarks
    image = _load_image(nii_path)
    landmarks = _load_landmarks(case_info)
    # Convert landmarks to voxel coordinates
    voxel_landmarks = _world_to_voxel_coords(image, landmarks)
    # Extract slices and plot
    # _extract_and_plot_slices(image, voxel_landmarks)
    _extract_and_save_slices(image, voxel_landmarks, save_path)

