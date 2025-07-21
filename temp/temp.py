from pathlib import Path
import logging
import numpy as np
import SimpleITK as sitk
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import os
from data_preprocessing.text_worker import (json_reader, yaml_reader, read_csv, clear_folder,
                                            add_info_logging)


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

    # Step 6: Convert VTK image to SimpleITK image
    # Get the numpy array from the VTK image
    # vtk_to_numpy = vtk.util.numpy_support.vtk_to_numpy
    # mask_array = vtk_to_numpy(stencil_to_image.GetOutput().GetPointData().GetScalars())
    # mask_array = mask_array.reshape(ref_size[::-1])  # Reshape to match SimpleITK

    # Convert numpy array to SimpleITK image
    # mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    # mask_image.SetSpacing(ref_spacing)
    # mask_image.SetOrigin(ref_origin)
    # mask_image.SetDirection(ref_direction)
    return vtk_image_to_sitk(stencil_to_image.GetOutput())


def cut_mask_using_points(sitk_image, updated_mask_path, top_points, bottom_points, margin=5):
    def compute_mask_volume(plane_coeff):
        """Подсчитывает объём маски"""
        a, b, c, d = plane_coeff
        non_zero_indices = np.transpose(np.nonzero(mask_array))
        volume = 0

        for zyx in non_zero_indices:
            z, y, x = zyx
            world = [
                origin[0] + x * spacing[0],
                origin[1] + y * spacing[1],
                origin[2] + z * spacing[2]
            ]
            value = a * world[0] + b * world[1] + c * world[2] + d
            if value <= 0:
                volume += 1

        return volume

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

        return a, b, c, d, normal

    def add_safety_margin_to_plane(d, normal, change_direction=False):
        """Add a safety margin to the plane by shifting it by `margin` along the normal direction."""

        def get_all_bounding_box_corners(size, spacing, origin):
            """Return all 8 corners of the image bounding box in world coordinates."""
            corners = []
            for xi in [0, size[0]]:
                for yi in [0, size[1]]:
                    for zi in [0, size[2]]:
                        world_x = origin[0] + xi * spacing[0]
                        world_y = origin[1] + yi * spacing[1]
                        world_z = origin[2] + zi * spacing[2]
                        corners.append(np.array([world_x, world_y, world_z]))
            return corners

        def calculate_max_margin():
            """Calculate the maximum allowed margin based on image bounding box."""
            corners = get_all_bounding_box_corners(size, spacing, origin)
            normal_norm = np.linalg.norm(normal)
            distances = [
                abs(np.dot(normal, corner) + d) / normal_norm
                for corner in corners
            ]
            return min(distances)

        # Normalize the normal vector
        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            logging.info("The normal vector is zero; the points may be collinear.")
            # raise ValueError("The normal vector is zero; the points may be collinear.")

        max_margin = calculate_max_margin()
        effective_margin = min(margin, max_margin)  # Use the smaller of the desired or maximum possible margin
        if margin > effective_margin:
            logging.info(f"Using effective margin: {effective_margin} (requested: {margin}, max possible: {max_margin})")
            # log_file.write(f"Using effective margin: {effective_margin} (requested: {margin}, max possible: {max_margin})")

        # Adjust d to move the plane by `margin` along the normal direction
        if change_direction:
            d_with_margin = d - (margin * normal_length)
        else:
            d_with_margin = d + (margin * normal_length)
        return d_with_margin

    def cut_mask_above_planes(lower_plane, upper_plane):
        """Set voxels above the plane ax + by + cz + d > 0 to zero in the binary mask."""
        a1, b1, c1, d1 = upper_plane
        a2, b2, c2, d2 = lower_plane

        # Получаем индексы ненулевых вокселей
        non_zero_indices = np.argwhere(mask_array > 0)

        for z, y, x in non_zero_indices:
            # Преобразуем индекс в мировые координаты
            world = np.array([
                origin[0] + x * spacing[0],
                origin[1] + y * spacing[1],
                origin[2] + z * spacing[2]
            ])

            # Проверка: находится ли воксель вне объема между плоскостями
            val1 = a1 * world[0] + b1 * world[1] + c1 * world[2] + d1
            val2 = a2 * world[0] + b2 * world[1] + c2 * world[2] + d2

            if val1 > 0 or val2 < 0:
                mask_array[z, y, x] = 0

        cut_mask = sitk.GetImageFromArray(mask_array)
        cut_mask.CopyInformation(sitk_image)
        return cut_mask

    # Get the bounding box of the mask in world coordinates
    size = sitk_image.GetSize()
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    mask_array = sitk.GetArrayFromImage(sitk_image)

    a1, b1, c1, d_upper, normal1 = find_plane_coefficients(top_points[0], top_points[1], top_points[2])
    vol_upper = compute_mask_volume([a1, b1, c1, d_upper])
    # logging.info("top")
    d1 = add_safety_margin_to_plane(d_upper, normal1)
    vol_upper_cut = compute_mask_volume([a1, b1, c1, d1])
    if vol_upper > vol_upper_cut:
        d1 = add_safety_margin_to_plane(d_upper, normal1, change_direction=True)

    upper_plane = [a1, b1, c1, d1]

    a2, b2, c2, d_lower, normal2 = find_plane_coefficients(bottom_points[0], bottom_points[1], bottom_points[2])
    vol_lower = compute_mask_volume([a2, b2, c2, d_lower])
    # logging.info("bottom")
    d2 = add_safety_margin_to_plane(d_lower, normal2)
    vol_lower_cut = compute_mask_volume([a2, b2, c2, d2])
    if vol_lower > vol_lower_cut:
        d2 = add_safety_margin_to_plane(d_lower, normal2, change_direction=True)

    lower_plane = [a2, b2, c2, d2]

    # Find the plane coefficients

    # Cut the mask above the plane
    cut_mask = cut_mask_above_planes(lower_plane, upper_plane)

    # Save the modified mask
    sitk.WriteImage(cut_mask, updated_mask_path)


def convert_and_cut_stl(stl_path, image_nii_path, mask_nii_path,  top_points, bottom_points, margin=2):
    convert_mask = convert_stl_to_mask_nii(stl_path, image_nii_path)
    cut_mask_using_points(convert_mask, mask_nii_path, top_points, bottom_points, margin)


def _find_series_folders(root_folder, types_file, parent=False):
    if isinstance(types_file, str):
        types_file = [types_file]
    series_folders = set()
    for ext in types_file:
        for file_path in Path(root_folder).rglob(f"*.{ext}"):
            if parent:
                series_folders.add(file_path.parent)
            else:
                series_folders.add(file_path)
    return list(series_folders)


def controller():
    stl_aorta_segment_folder = os.path.join(data_path, "stl_aorta_segment")
    stl_files = _find_series_folders(stl_aorta_segment_folder, "stl")
    result_folder = os.path.join(data_path, "result")
    train_test_lists = read_csv(result_folder, "train_test_lists.csv")
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")
    dict_all_case = json_reader(dict_all_case_path)
    mask_aorta_segment_folder = os.path.join(data_path, "mask_aorta_segment")
    mask_aorta_segment_cut_folder = os.path.join(data_path, "mask_aorta_segment_cut")
    image_folder = os.path.join(data_path, "image_nii")

    clear_folder(mask_aorta_segment_folder)
    for stl_path in stl_files:
        case_base_name = stl_path.stem
        # Найти в датафрейме
        match = train_test_lists[train_test_lists["case_name"].str.contains(case_base_name, na=False)]
        if match.empty:
            add_info_logging(f" Not found: {case_base_name} in dict cases")
            continue

        used_case_name = match.iloc[0]["used_case_name"]

        mask_aorta_segment_file = os.path.join(mask_aorta_segment_folder, f"{used_case_name}.nii.gz")
        image_file = os.path.join(image_folder, f"{used_case_name}.nii.gz")
        top_points = [dict_all_case[used_case_name]['R'],
                      dict_all_case[used_case_name]['L'],
                      dict_all_case[used_case_name]['N']]
        bottom_points = [dict_all_case[used_case_name]['RLC'],
                         dict_all_case[used_case_name]['RNC'],
                         dict_all_case[used_case_name]['LNC']]
        convert_and_cut_stl(stl_path,
                            image_file,
                            mask_aorta_segment_file,
                            top_points, bottom_points, margin=2)

    # add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller_path = "C:/Users/Kamil/Aortic_valve/code_aortic_valve/controller.yaml"
    controller_dump = yaml_reader(controller_path)
    controller()