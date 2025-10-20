import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from data_preprocessing.text_worker import add_info_logging
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_dilation


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
    :param radius: Sphere radius (in voxels).
    :param spacing: Voxel dimensions (image spacing).
    :return: SimpleITK image with mask.
    """
    # center is expected as (X, Y, Z) — so we access center[2]=Z, [1]=Y, [0]=X
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

    coord = np.array(coord).reshape(-1)  # гарантируем форму (3,)
    # Преобразование
    voxel_coord = np.linalg.inv(direction).dot(np.array(coord) - origin) / spacing
    return np.round(voxel_coord).astype(int)


def process_mask_gh_lines(image_path, dict_case, output_path, radius, keys_to_need=None, step_vox=0.5):
    """
    Creates a mask with multiple curves, where each curve has its own mask value.

    :param image_path: path to the original .nii file
    :param dict_case: dictionary with point coordinates (in world coordinates)
    :param output_path: path to save the mask
    :param radius: tube radius (in pixels)
    :param keys_to_need: dictionary { 'RGH':1, 'LGH':2, ... }
    :param step_vox: step along the curve (in pixels)
    """
    # --- загрузка изображения ---
    image = _load_image(image_path)
    shape = image.GetSize()  # (x, y, z)
    mask = np.zeros(shape[::-1], dtype=np.uint8)  # (z, y, x)

    # --- структурный элемент для утолщения ---
    rad = int(np.ceil(radius))
    zz, yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1, -rad:rad + 1]
    struct = xx ** 2 + yy ** 2 + zz ** 2 <= radius ** 2

    # --- проход по кривым ---
    for key, point_list in dict_case.items():
        if not keys_to_need or key not in keys_to_need:
            continue
        if len(point_list) < 2:
            continue

        label_value = keys_to_need[key]  # значение, которое запишем в маску

        # --- перевод точек в воксели ---
        vox_points = np.array([_world_to_voxel(p, image) for p in point_list])
        if vox_points.shape[0] < 2:
            continue

        # --- построение сплайна ---
        try:
            tck, _ = splprep(vox_points.T, s=0)
        except Exception as e:
            print(f"Ошибка сплайна для {key}: {e}")
            continue

        # --- вычисляем длину кривой и число точек ---
        u_vals = np.linspace(0, 1, 100)
        pts = np.array(splev(u_vals, tck)).T
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_len = np.sum(dists)

        n_points = max(2, int(total_len / step_vox))
        u_new = np.linspace(0, 1, n_points)
        interp_points = np.array(splev(u_new, tck)).T

        # --- временная маска для одной кривой ---
        curve_mask = np.zeros_like(mask, dtype=bool)

        # --- рисуем тонкую линию ---
        for p in interp_points:
            x, y, z = np.round(p).astype(int)
            if (0 <= x < shape[0]) and (0 <= y < shape[1]) and (0 <= z < shape[2]):
                curve_mask[z, y, x] = True

        # --- утолщаем до трубки ---
        curve_mask = binary_dilation(curve_mask, structure=struct)

        # --- записываем в итоговую маску ---
        mask[curve_mask] = label_value

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(image)
    _save_image(mask_image, output_path)


def process_markers(image_path, dict_case, output_path, radius, keys_to_need=None):
    """
    Processes one pair (image and coordinate table).
    Creates a mask and saves it.
    """
    image = _load_image(image_path)
    shape = image.GetSize()

    # Создаём пустую маску
    mask = np.zeros(shape[::-1], dtype=np.uint8)  # Меняем порядок: Z, Y, X

    # keys_to_need = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']
    # keys_to_need = {
    #     'R': 1, 'L': 2, 'N': 3,
    #     'RLC': 4, 'RNC': 5, 'LNC': 6
    # }
    # keys_to_need = {'GH': 1}
    for key, coord in dict_case.items():
        if key not in keys_to_need:
            continue
        voxel_coord = _world_to_voxel(coord, image)

        # Проверка на выход за границы
        if not all(0 <= voxel_coord[d] < shape[d] for d in range(3)):
            add_info_logging(f"Point {coord} (voxel coordinates {voxel_coord}) is outside the image volume.",
                             "work_logger")
            continue

        # Создаём сферу и добавляем её в маску
        sphere = _create_sphere_mask(mask.shape, voxel_coord, radius)
        # Заполняем маску уникальным числом
        mask[sphere > 0] = keys_to_need[key]
        # mask = np.maximum(mask, sphere)

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


def find_mean_gh_landmark(dict_points):
    # Функция для подсчёта суммарного расстояния между тремя точками
    def _sum_distances(p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        return np.linalg.norm(p1 - p2) + np.linalg.norm(p1 - p3) + np.linalg.norm(p2 - p3)

    # Берём первые и последние точки
    first_points = (dict_points["RGH"][0], dict_points["LGH"][0], dict_points["NGH"][0])
    last_points = (dict_points["RGH"][-1], dict_points["LGH"][-1], dict_points["NGH"][-1])

    # Суммарные расстояния
    sum_first = _sum_distances(*first_points)
    sum_last = _sum_distances(*last_points)

    # Выбор набора точек
    if sum_first < sum_last:
        chosen_points = first_points
    else:
        chosen_points = last_points

    # Средняя точка
    center_point = np.mean(np.array(chosen_points), axis=0)
    center_point = [round(x, 2) for x in center_point.tolist()]
    return center_point

