import SimpleITK as sitk
import matplotlib.pyplot as plt


def _load_image(image_path):
    return sitk.ReadImage(image_path)


def _load_landmarks(file_path):
    landmarks = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            landmarks.append((x, y, z))
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


def _extract_and_save_slices(image, voxel_coords):
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
        plt.savefig(f'Landmark_{idx + 1}_slices.png')
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
    _extract_and_plot_slices(image, voxel_landmarks, save_path)

