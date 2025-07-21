import numpy as np
from scipy.ndimage import center_of_mass
import SimpleITK as sitk
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from data_preprocessing.text_worker import add_info_logging
from scipy.ndimage import label


def convert_stl_to_mask_nii(stl_file, reference_image_file):
    """Convert an STL file to a binary mask that matches the reference image grid."""

    def vtk_image_to_sitk(vtk_image):
        """Convert vtkImageData to SimpleITK Image."""
        # Step 1: Extract VTK image dimensions, spacing, and origin
        dimensions = vtk_image.GetDimensions()
        spacing = vtk_image.GetSpacing()
        origin = vtk_image.GetOrigin()
        np.bool = np.bool_
        # Step 2: Get VTK image data as a NumPy array
        vtk_array = vtk_image.GetPointData().GetScalars()  # Get the VTK data array
        np_array = vtk_to_numpy(vtk_array).astype(np.uint8)  # Convert VTK array to NumPy array
        np_array = np_array.reshape(dimensions[::-1])  # Reshape to match SimpleITK's z, y, x order

        # Step 3: Convert NumPy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(np_array)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)

        # Step 4: Set Direction (assuming identity if not available)
        # VTK doesn't provide direction, so we often assume identity, or you can define it as needed.
        sitk_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

        return sitk_image

    # Step 1: Load the STL file using VTK
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    poly_data = reader.GetOutput()

    # Step 2: Get the reference image properties
    reference_image = sitk.ReadImage(reference_image_file)
    ref_spacing = reference_image.GetSpacing()
    ref_origin = reference_image.GetOrigin()
    ref_direction = reference_image.GetDirection()
    ref_size = reference_image.GetSize()

    # Step 3: Set up VTK to match the reference image grid
    # Define VTK image with the same dimensions and spacing
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(ref_size)
    vtk_image.SetSpacing(ref_spacing)
    vtk_image.SetOrigin(ref_origin)

    # Step 4: Use VTK's PolyDataToImageStencil to convert STL to a mask
    # Create a stencil from the STL poly data
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly_data)
    stencil.SetOutputOrigin(vtk_image.GetOrigin())
    stencil.SetOutputSpacing(vtk_image.GetSpacing())
    stencil.SetOutputWholeExtent(vtk_image.GetExtent())
    stencil.Update()

    # Step 5: Convert stencil to binary mask in VTK
    # Initialize an empty binary mask in VTK and apply stencil to it
    stencil_to_image = vtk.vtkImageStencilToImage()
    stencil_to_image.SetInputConnection(stencil.GetOutputPort())
    stencil_to_image.SetInsideValue(1)  # Set voxel value inside the STL surface
    stencil_to_image.SetOutsideValue(0)  # Set voxel value outside the STL surface
    stencil_to_image.Update()

    return vtk_image_to_sitk(stencil_to_image.GetOutput())


def cut_mask_using_points(sitk_image, updated_mask_path, points_1, points_2, margin):
    def find_plane_coefficients(p1, p2, p3):
        """Calculate the coefficients (a, b, c, d) of the plane passing through points p1, p2, p3."""
        # Create vectors from the points
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)

        # Compute the cross product to find the normal vector
        normal = np.cross(v1, v2)
        a, b, c = normal

        # Calculate d using point p1
        d = -np.dot(normal, p1)

        return a, b, c, d

    def shifted_planes(p1, p2, p3, offset_mm):
        """
        Построить две параллельные плоскости, смещённые на ±offset_mm (в миллиметрах)
        от базовой плоскости через точки p1, p2, p3.
        Возвращает два набора коэффициентов (a, b, c, d1) и (a, b, c, d2)
        """
        a, b, c, d = find_plane_coefficients(p1, p2, p3)
        normal = np.array([a, b, c])
        norm_length = np.linalg.norm(normal)
        unit_normal = normal / norm_length

        # d0 = -dot(unit_normal, p1)
        d0 = -np.dot(unit_normal, p1)

        # Смещаем вдоль нормали (в миллиметрах)
        d1 = d0 - offset_mm  # одна сторона
        d2 = d0 + offset_mm  # другая сторона

        a_u, b_u, c_u = unit_normal

        return (a_u, b_u, c_u, d1), (a_u, b_u, c_u, d2)

    def split_mask_by_plane_largest_components(nii_mask, plane_coefficients):
        """
        Разделяет маску на две части по плоскости и возвращает два массива,
        соответствующих самым крупным связным компонентам с каждой стороны.

        Parameters:
            nii_mask (SimpleITK.Image): бинарная маска.
            plane_coefficients (tuple): (a, b, c, d) — уравнение плоскости.

        Returns:
            largest_comp_1 (np.ndarray), largest_comp_2 (np.ndarray): два массива,
            содержащих только самые крупные компоненты по разные стороны плоскости.
        """
        a, b, c, d = plane_coefficients

        spacing = nii_mask.GetSpacing()
        origin = nii_mask.GetOrigin()
        mask_np = sitk.GetArrayFromImage(nii_mask)  # shape: [Z, Y, X]

        # Координаты ненулевых пикселей
        zyx = np.argwhere(mask_np > 0)

        # Разделим по плоскости
        side1 = np.zeros_like(mask_np, dtype=bool)
        side2 = np.zeros_like(mask_np, dtype=bool)

        for z, y, x in zyx:
            world_x = origin[0] + x * spacing[0]
            world_y = origin[1] + y * spacing[1]
            world_z = origin[2] + z * spacing[2]
            val = a * world_x + b * world_y + c * world_z + d
            if val < 0:
                side1[z, y, x] = True
            else:
                side2[z, y, x] = True

        # Найдём крупнейшую связную компоненту в каждой стороне
        def largest_connected_component(mask):
            labeled, num_features = label(mask)
            if num_features == 0:
                return np.zeros_like(mask)
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # фон
            max_label = sizes.argmax()
            return (labeled == max_label).astype(np.uint8)

        largest_comp_1 = largest_connected_component(side1)
        largest_comp_2 = largest_connected_component(side2)

        return largest_comp_1, largest_comp_2

    def get_mask_intersection(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Возвращает бинарную маску пересечения двух масок (AND по элементам).

        Parameters:
            mask1 (np.ndarray): первая бинарная маска
            mask2 (np.ndarray): вторая бинарная маска

        Returns:
            np.ndarray: маска пересечения
        """
        # Приведение к bool, если вдруг маски целочисленные
        intersection = np.logical_and(mask1.astype(bool), mask2.astype(bool))
        return intersection.astype(np.uint8)  # или оставить bool, если не критично

    def find_unique_max_intersection(mask_set1, mask_set2):
        """
        Находит уникальную пару масок с наибольшим объёмом пересечения.
        Если пересечений с одинаковым объёмом несколько — исключает их и ищет следующее по величине.

        Parameters:
            mask_set1 (List[np.ndarray])
            mask_set2 (List[np.ndarray])

        Returns:
            np.ndarray or None: маска пересечения, если найдена уникальная по объёму, иначе None
        """
        from collections import defaultdict

        # Собираем пересечения по объёму
        intersections_by_volume = defaultdict(list)

        for i, m1 in enumerate(mask_set1):
            for j, m2 in enumerate(mask_set2):
                inter = get_mask_intersection(m1, m2)
                inter_voxels = np.sum(inter)
                if inter_voxels > 0:
                    intersections_by_volume[inter_voxels].append(inter)

        if not intersections_by_volume:
            return None

        # Сортируем объёмы по убыванию
        sorted_volumes = sorted(intersections_by_volume.keys(), reverse=True)

        for volume in sorted_volumes:
            candidates = intersections_by_volume[volume]
            if len(candidates) == 1:
                return candidates[0]  # Единственное максимальное пересечение найдено
            # иначе переходим к следующему по убыванию объёму

        return None  # Ни одно пересечение не является уникальным максимальным

    def replace_mask_with_intersection(original_sitk_image: sitk.Image, intersection_mask: np.ndarray) -> sitk.Image:
        """
        Заменяет содержимое маски в исходном sitk.Image на данные из intersection_mask.

        Parameters:
            original_sitk_image (sitk.Image): оригинальное NIfTI изображение маски
            intersection_mask (np.ndarray): бинарная маска-пересечение (тип np.uint8 или bool)

        Returns:
            sitk.Image: новое изображение с той же геометрией, но с обновлённой маской
        """

        # Преобразуем numpy-маску обратно в SimpleITK изображение
        new_image = sitk.GetImageFromArray(intersection_mask.astype(np.uint8))
        new_image.CopyInformation(original_sitk_image)

        return new_image

    plane_1_1, plane_1_2 = shifted_planes(points_1[0], points_1[1], points_1[2], margin)
    plane_2_1, plane_2_2 = shifted_planes(points_2[0], points_2[1], points_2[2], margin)

    mask_1 = [*split_mask_by_plane_largest_components(sitk_image, plane_1_1),
              *split_mask_by_plane_largest_components(sitk_image, plane_1_2)]

    mask_2 = [*split_mask_by_plane_largest_components(sitk_image, plane_2_1),
              *split_mask_by_plane_largest_components(sitk_image, plane_2_2)]

    new_mask_array = find_unique_max_intersection(mask_1, mask_2)

    cut_mask = replace_mask_with_intersection(sitk_image, new_mask_array)

    sitk.WriteImage(cut_mask, updated_mask_path)


def convert_and_cut_stl(stl_path, image_nii_path, mask_nii_path,  points1, points2, margin=2):
    convert_mask = convert_stl_to_mask_nii(stl_path, image_nii_path)
    cut_mask_using_points(convert_mask, mask_nii_path, points1, points2, margin)
