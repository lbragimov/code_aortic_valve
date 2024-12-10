import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path


def _load_image(image_path):
    return sitk.ReadImage(image_path)


def _save_image(image, output_path):
    """Saves the image to a file."""
    sitk.WriteImage(image, output_path)


def _create_sphere_mask(shape, center, radius, spacing):
    """
    Creates a sphere at the given point with the given radius.
    :param shape: Mask dimensions (voxels).
    :param center: Center coordinates (voxels).
    :param radius: Sphere radius (in millimeters).
    :param spacing: Voxel dimensions (image spacing).
    :return: SimpleITK image with mask.
    """
    mask = sitk.Image(shape, sitk.sitkUInt8)
    mask.SetSpacing(spacing)

    # Creates a sphere
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                distance = np.sqrt(
                    ((z - center[2]) * spacing[2]) ** 2 +
                    ((y - center[1]) * spacing[1]) ** 2 +
                    ((x - center[0]) * spacing[0]) ** 2
                )
                if distance <= radius:
                    mask[x, y, z] = 1
    return mask


def process_pair(image_path, json_path, output_path, radius):
    """
    Обрабатывает одну пару (изображение и таблица координат).
    Создаёт маску и сохраняет её.
    """
    # Загрузка изображения
    image = _load_image(image_path)
    spacing = image.GetSpacing()
    shape = image.GetSize()

    # Загрузка координат из JSON
    with open(json_path, 'r') as json_file:
        coords = json.load(json_file)

    # Создаём пустую маску
    mask = sitk.Image(shape, sitk.sitkUInt8)
    mask.SetSpacing(spacing)

    # Обрабатываем первые шесть точек
    for i, (key, coord) in enumerate(coords.items()):
        if i >= 6:  # Только первые 6 точек
            break

        # Создаём сферу
        sphere = _create_sphere_mask(shape, coord, radius, spacing)

        # Объединяем сферу с текущей маской
        mask += sphere

    # Сохраняем маску
    _save_image(mask, output_path)


def _load_landmarks(dict_landmarks):
    landmarks = []
    keys_to_keep = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']
    filtered_dict = dict(filter(lambda item: item[0] in keys_to_keep, dict_landmarks.items()))
    # for current_landmarks in keys_to_keep:
    #     landmarks.append(tuple(dict_landmarks[current_landmarks]))
    return filtered_dict


def _world_to_voxel_coords(image, world_coords):
    for name_point, point in world_coords.items():
        world_coords[name_point] = image.TransformPhysicalPointToIndex(point)
    # return [image.TransformPhysicalPointToIndex(point) for point in world_coords]
    return world_coords


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
    for name_point, point in voxel_coords.items():
        x = point[0]
        y = point[1]
        z = point[2]
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
        plt.suptitle(f'Landmark {name_point}')

        folder_path = Path(save_path)
        if folder_path.is_dir():
            plt.savefig(save_path + f'/Landmark_{name_point}_slices.png')
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path + f'/Landmark_{name_point}_slices.png')
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

