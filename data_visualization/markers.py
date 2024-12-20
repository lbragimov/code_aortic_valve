import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path


def _load_image(image_path):
    return sitk.ReadImage(image_path)


def _save_image(image, output_path):
    """Saves the image to a file."""
    sitk.WriteImage(image, output_path)


def _create_sphere_mask(shape, center, radius):#, spacing):
    """
    Creates a sphere at the given point with the given radius.
    :param shape: Mask dimensions (voxels).
    :param center: Center coordinates (voxels).
    :param radius: Sphere radius (in millimeters).
    :param spacing: Voxel dimensions (image spacing).
    :return: SimpleITK image with mask.
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]  # Z, Y, X
    distance = np.sqrt((z - center[2]) ** 2 + (y - center[1]) ** 2 + (x - center[0]) ** 2)
    sphere = distance <= radius
    return sphere.astype(np.uint8)


def _world_to_voxel(coord, image):
    """
    Converts physical coordinates (world) to voxel indices.
    :param coord: Coordinates in physical units (millimeters).
    :param image: SimpleITK image object.
    :return: Coordinates in voxel indices.
    """
    origin = np.array(image.GetOrigin())  # Точка отсчёта
    spacing = np.array(image.GetSpacing())  # Размер вокселя
    direction = np.array(image.GetDirection()).reshape(3, 3)  # Ориентация

    # Преобразование
    voxel_coord = np.linalg.inv(direction).dot(np.array(coord) - origin) / spacing
    return np.round(voxel_coord).astype(int)


def process_markers(image_path, dict_case, output_path, radius):
    """
    Processes one pair (image and coordinate table).
    Creates a mask and saves it.
    """
    image = _load_image(image_path)
    shape = image.GetSize()

    # Создаём пустую маску
    mask = np.zeros(shape[::-1], dtype=np.uint8)  # Меняем порядок: Z, Y, X

    keys_to_need = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']
    for key, coord in dict_case.items():
        if not key in keys_to_need:
            continue
        voxel_coord = _world_to_voxel(coord, image)

        # Проверка на выход за границы
        if not all(0 <= voxel_coord[d] < shape[d] for d in range(3)):
            print(f"Точка {coord} (воксельные координаты {voxel_coord}) вне объёма изображения.")
            continue

        # Создаём сферу и добавляем её в маску
        sphere = _create_sphere_mask(mask.shape, voxel_coord, radius)
        mask = np.maximum(mask, sphere)

    # Преобразуем маску в SimpleITK-объект
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetSpacing(image.GetSpacing())
    mask_image.SetOrigin(image.GetOrigin())
    mask_image.SetDirection(image.GetDirection())

    # Сохраняем маску
    _save_image(mask_image, output_path)


# все ниже для отрисовки маркеров на 2д картинке
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

