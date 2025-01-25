import numpy as np
import SimpleITK as sitk


def calculate_new_bounds(image_path, padding):
    """
    Calculates the bounding box for the mask with padding.
    :param mask: NumPy array representing the mask.
    :param padding: Number of voxels to add to the bounding box.
    :return: Boundings in the format [(z_min, z_max), (y_min, y_max), (x_min, x_max)].
    """
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # Find the indices of nonzero voxels in the mask
    nonzero_indices = np.argwhere(image_array > 0)

    # Determine the minimum and maximum indices for each axis
    z_min, y_min, x_min = nonzero_indices.min(axis=0)
    z_max, y_max, x_max = nonzero_indices.max(axis=0)

    # Добавляем запас
    z_min = max(0, z_min - padding)
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    z_max = min(image_array.shape[0], z_max + padding)
    y_max = min(image_array.shape[1], y_max + padding)
    x_max = min(image_array.shape[2], x_max + padding)

    return [(z_min, z_max), (y_min, y_max), (x_min, x_max)]


def _crop_image(image, bounds):
    """
    Обрезает изображение по указанным границам.
    :param image: NumPy массив, представляющий изображение.
    :param bounds: Границы [(z_min, z_max), (y_min, y_max), (x_min, x_max)].
    :return: Обрезанное изображение.
    """
    z_min, z_max = bounds[0]
    y_min, y_max = bounds[1]
    x_min, x_max = bounds[2]
    return image[z_min:z_max, y_min:y_max, x_min:x_max]


def cropped_image(input_image_path, output_image_path, bounds):
    mask_image = sitk.ReadImage(input_image_path)
    mask_array = sitk.GetArrayFromImage(mask_image)

    # Обрезаем изображение (например, саму маску или другое изображение из кейса)
    cropped_mask = _crop_image(mask_array, bounds)

    # Преобразование обрезанной маски обратно в SimpleITK формат
    cropped_mask_image = sitk.GetImageFromArray(cropped_mask)
    cropped_mask_image.SetSpacing(mask_image.GetSpacing())
    cropped_mask_image.SetOrigin(mask_image.GetOrigin())
    cropped_mask_image.SetDirection(mask_image.GetDirection())

    sitk.WriteImage(cropped_mask_image, output_image_path)
