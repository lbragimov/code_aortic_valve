import math
import numpy as np
import SimpleITK as sitk


def find_shape(image_path, size_or_pixel):
    # shapes = []
    # for image_path in image_paths:
    image = sitk.ReadImage(image_path)
    if size_or_pixel == "size":
        variable = image.GetSize()
    else:
        variable = image.GetSpacing()
    return variable


def find_shape_2(image_paths):
    shapes = []
    for image_path in image_paths:
        image = sitk.ReadImage(image_path)
        shapes.append(image.GetSize())
    return set(shapes)


def find_global_size(image_paths, padding=16):
    """
    Находит общий size box для всех изображений.
    :param image_paths: Список путей к изображениям в формате NIfTI (.nii)
    :param padding: Дополнительный запас вокселей вокруг найденной маски
    :return: [(z_min, z_max), (y_min, y_max), (x_min, x_max)]
    """
    global_size_z, global_size_y, global_size_x = 0, 0, 0

    for image_path in image_paths:
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)

        nonzero_indices = np.argwhere(image_array > 0)  # Координаты всех ненулевых пикселей

        if nonzero_indices.size == 0:
            continue  # Если маска пустая, пропускаем

        # Определяем min/max координаты
        z_min, y_min, x_min = nonzero_indices.min(axis=0)
        z_max, y_max, x_max = nonzero_indices.max(axis=0)

        global_size_z = max(global_size_z, z_max - z_min)
        global_size_y = max(global_size_y, y_max - y_min)
        global_size_x = max(global_size_x, x_max - x_min)

    global_size_z = global_size_z + (2*padding)
    global_size_y = global_size_y + (2*padding)
    global_size_x = global_size_x + (2*padding)

    return [global_size_z, global_size_y, global_size_x]



def _calculate_new_bounds(mask, size):
    """
    Calculates the bounding box for the mask with padding.
    :param mask: NumPy array representing the mask.
    :param padding: Number of voxels to add to the bounding box.
    :return: Boundings in the format [(z_min, z_max), (y_min, y_max), (x_min, x_max)].
    """
    # Find the indices of nonzero voxels in the mask
    nonzero_indices = np.argwhere(mask > 0)

    # Determine the minimum and maximum indices for each axis
    z_min, y_min, x_min = nonzero_indices.min(axis=0)
    z_max, y_max, x_max = nonzero_indices.max(axis=0)
    padding_z_st = (size[0] - (z_max - z_min)) / 2
    padding_z_fin = padding_z_st
    padding_y_st = (size[1] - (y_max - y_min)) / 2
    padding_y_fin = padding_y_st
    padding_x_st = (size[2] - (x_max - x_min)) / 2
    padding_x_fin = padding_x_st
    if math.ceil(padding_z_st) > z_min:
        padding_z_fin = math.floor(padding_z_fin) + math.ceil(padding_z_st) - z_min
        padding_z_st = z_min
    elif math.floor(padding_z_fin) + z_max > mask.shape[0]:
        padding_z_st = math.floor(padding_z_st) + math.ceil(padding_z_fin) - (mask.shape[0]-z_max)
        padding_z_fin = mask.shape[0]-z_max
    if math.ceil(padding_y_st) > y_min:
        padding_y_fin = math.floor(padding_y_fin) + math.ceil(padding_y_st) - y_min
        padding_y_st = y_min
    elif math.floor(padding_y_fin) + y_max > mask.shape[1]:
        padding_y_st = math.floor(padding_y_st) + math.ceil(padding_y_fin) - (mask.shape[1]-y_max)
        padding_y_fin = mask.shape[1]-y_max
    if math.ceil(padding_x_st) > x_min:
        padding_x_fin = math.floor(padding_x_fin) + math.ceil(padding_x_st) - x_min
        padding_x_st = x_min
    elif math.floor(padding_x_fin) + x_max > mask.shape[2]:
        padding_x_st = math.floor(padding_x_st) + math.ceil(padding_x_fin) - (mask.shape[2]-x_max)
        padding_x_fin = mask.shape[2]-x_max

    # Добавляем запас
    z_min = z_min - math.ceil(padding_z_st)
    y_min = y_min - math.ceil(padding_y_st)
    x_min = x_min - math.ceil(padding_x_st)
    z_max = z_max + math.floor(padding_z_fin)
    y_max = y_max + math.floor(padding_y_fin)
    x_max = x_max + math.floor(padding_x_fin)

    return [(z_min, z_max), (y_min, y_max), (x_min, x_max)]


def _crop_image(mask, image, size):
    """
    Обрезает изображение по указанным границам.
    :param mask: NumPy массив, представляющий маску.
    :param image: NumPy массив, представляющий изображение.
    :param size: Размеры обрезаемого участка [z, y, x].
    :return: Обрезанное изображение.
    """
    bounds = _calculate_new_bounds(mask, size)
    z_min, z_max = bounds[0]
    y_min, y_max = bounds[1]
    x_min, x_max = bounds[2]

    cropped = image[z_min:z_max, y_min:y_max, x_min:x_max]

    return cropped, (z_min, y_min, x_min)


def cropped_image(mask_image_path, input_image_path, output_image_path, size):
    mask_image = sitk.ReadImage(mask_image_path)
    mask_array = sitk.GetArrayFromImage(mask_image)

    input_image = sitk.ReadImage(input_image_path)
    input_array = sitk.GetArrayFromImage(input_image)

    # Обрезаем изображение и получаем смещение
    cropped_array_image, start_index = _crop_image(mask_array, input_array, size)

    cropped_image = sitk.GetImageFromArray(cropped_array_image)
    cropped_image.SetSpacing(input_image.GetSpacing())
    cropped_image.SetDirection(input_image.GetDirection())

    old_origin = input_image.GetOrigin()
    spacing = input_image.GetSpacing()

    # Важно! Индексы массива идут как (z, y, x), origin — в порядке (x, y, z)
    new_origin = [
        old_origin[0] + start_index[2] * spacing[0],  # x
        old_origin[1] + start_index[1] * spacing[1],  # y
        old_origin[2] + start_index[0] * spacing[2]  # z
    ]

    cropped_image.SetOrigin(new_origin)

    sitk.WriteImage(cropped_image, output_image_path)


def fix_origin_cropped_image(cropped_image_path, mask_image_path, original_image_path, size, output_image_path):
    """
    Корректирует origin у уже обрезанного изображения, используя маску и предполагаемый размер обрезки.

    :param cropped_image_path: Путь к уже обрезанному изображению (но с origin как у полного).
    :param mask_image_path: Путь к маске, по которой делалась обрезка.
    :param original_image_path: Путь к полному исходному изображению.
    :param size: Размер обрезаемого участка [z, y, x].
    :param output_image_path: Путь для сохранения изображения с исправленным origin.
    """
    mask_image = sitk.ReadImage(mask_image_path)
    mask_array = sitk.GetArrayFromImage(mask_image)

    original_image = sitk.ReadImage(original_image_path)
    old_origin = original_image.GetOrigin()
    spacing = original_image.GetSpacing()

    # Вычисляем границы и смещение, как если бы делали обрезку
    _, start_index = _crop_image(mask_array, sitk.GetArrayFromImage(original_image), size)

    # Загружаем уже обрезанное изображение
    cropped_image = sitk.ReadImage(cropped_image_path)

    # Считаем новое origin
    new_origin = [
        old_origin[0] + start_index[2] * spacing[0],  # x
        old_origin[1] + start_index[1] * spacing[1],  # y
        old_origin[2] + start_index[0] * spacing[2]  # z
    ]

    cropped_image.SetOrigin(new_origin)

    sitk.WriteImage(cropped_image, output_image_path)
