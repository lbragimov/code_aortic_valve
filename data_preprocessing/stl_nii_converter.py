import logging
import trimesh
import nibabel as nib
import numpy as np
import SimpleITK as sitk
# from stl import mesh
from stl import mesh
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import os
import glob
import random
from data_preprocessing.text_worker import add_info_logging



# def convert_stl_to_nii(input_stl_file: str, output_nii_file: str, volume_shape: tuple[int]):
#     # Load STL file
#     mesh = trimesh.load_mesh(input_stl_file)
#
#     # Create empty volume (e.g. 128x128x128)
#     volume = np.zeros(volume_shape)
#
#     # Convert STL coordinates to voxels
#     for vertex in mesh.vertices:
#         x, y, z = (
#                 (vertex - mesh.bounds[0]) / (mesh.bounds[1] - mesh.bounds[0]) * np.array(volume_shape)
#         ).astype(int)
#         volume[x-1, y-1, z-1] = 1  # Mark voxel as "occupied"
#
#     # Save in NIfTI format
#     nii_img = nib.Nifti1Image(volume, affine=np.eye(4))
#     nib.save(nii_img, output_nii_file)



# def create_binary_mask_from_stl(stl_file, grid_size):
#     # Load STL file
#     stl_mesh = mesh.Mesh.from_file(stl_file)
#
#     # Get the bounds of the mesh
#     x_min, x_max = np.min(stl_mesh.points[:, 0]), np.max(stl_mesh.points[:, 0])
#     y_min, y_max = np.min(stl_mesh.points[:, 1]), np.max(stl_mesh.points[:, 1])
#     z_min, z_max = np.min(stl_mesh.points[:, 2]), np.max(stl_mesh.points[:, 2])
#
#     # Create a SimpleITK image
#     spacing = [(x_max - x_min) / grid_size[0], (y_max - y_min) / grid_size[1], (z_max - z_min) / grid_size[2]]
#     size = grid_size
#     binary_image = sitk.Image(size, sitk.sitkUInt8)
#     binary_image.SetSpacing(spacing)
#     binary_image.SetOrigin((x_min, y_min, z_min))
#
#     # Initialize a numpy array to hold voxel values
#     voxel_array = sitk.GetArrayFromImage(binary_image)
#
#     # Rasterize the STL into the voxel array
#     for triangle in stl_mesh.vectors:
#         # Create a mask for the current triangle
#         mask = create_triangle_mask(triangle, grid_size, (x_min, y_min, z_min), spacing)
#         voxel_array |= mask
#
#     # Fill holes in the binary mask
#     filled_mask = sitk.BinaryFillhole(sitk.GetImageFromArray(voxel_array))
#
#     return filled_mask
#
# def create_triangle_mask(triangle, grid_size, origin, spacing):
#     # Create a blank mask
#     mask = np.zeros(grid_size, dtype=np.uint8)
#
#     # Convert triangle vertices to voxel indices
#     for point in triangle:
#         # Calculate voxel indices based on point coordinates
#         index = ((point - np.array(origin)) / spacing).astype(int)
#         if all(0 <= idx < size for idx, size in zip(index, grid_size)):
#             mask[tuple(index)] = 1
#
#     return mask
#
# # Example usage
# stl_file_path = 'path/to/your/file.stl'
# grid_size = (100, 100, 100)  # Define the resolution of the binary mask
# binary_mask = create_binary_mask_from_stl(stl_file_path, grid_size)
#
# # You can save or visualize the binary mask
# sitk.WriteImage(binary_mask, 'binary_mask.nrrd')
#
#
#
# def create_solid_mask_from_stl(stl_file, grid_size):
#     # Load the STL file
#     stl_mesh = mesh.Mesh.from_file(stl_file)
#
#     # Check if the mesh is a closed surface
#     if not is_closed_surface(stl_mesh):
#         raise ValueError("The STL file does not represent a closed surface.")
#
#     # Get the bounds of the mesh
#     x_min, x_max = np.min(stl_mesh.points[:, 0]), np.max(stl_mesh.points[:, 0])
#     y_min, y_max = np.min(stl_mesh.points[:, 1]), np.max(stl_mesh.points[:, 1])
#     z_min, z_max = np.min(stl_mesh.points[:, 2]), np.max(stl_mesh.points[:, 2])
#
#     # Create a SimpleITK image
#     spacing = [(x_max - x_min) / grid_size[0], (y_max - y_min) / grid_size[1], (z_max - z_min) / grid_size[2]]
#     size = grid_size
#     binary_image = sitk.Image(size, sitk.sitkUInt8)
#     binary_image.SetSpacing(spacing)
#     binary_image.SetOrigin((x_min, y_min, z_min))
#
#     # Initialize a numpy array to hold voxel values
#     voxel_array = sitk.GetArrayFromImage(binary_image)
#
#     # Rasterize the STL into the voxel array
#     for triangle in stl_mesh.vectors:
#         mask = create_triangle_mask(triangle, grid_size, (x_min, y_min, z_min), spacing)
#         voxel_array |= mask
#
#     # Fill holes in the binary mask to create a solid
#     filled_mask = sitk.BinaryFillhole(sitk.GetImageFromArray(voxel_array))
#
#     return filled_mask
#
# def is_closed_surface(stl_mesh):
#     # A simple check for closed surface:
#     # Ensure all triangles share edges properly (more complex checks are needed for robustness)
#     # This is a placeholder; you may need a more rigorous approach for complex models.
#     return len(stl_mesh.vectors) > 0
#
# def create_triangle_mask(triangle, grid_size, origin, spacing):
#     # Create a blank mask
#     mask = np.zeros(grid_size, dtype=np.uint8)
#
#     # Define a method to fill the triangle in the 3D grid
#     # Placeholder for a simple implementation (requires proper rasterization logic)
#     # For now, just mark the triangle points
#     for point in triangle:
#         index = ((point - np.array(origin)) / spacing).astype(int)
#         if all(0 <= idx < size for idx, size in zip(index, grid_size)):
#             mask[tuple(index)] = 1
#
#     return mask
#
# # Example usage
# stl_file_path = 'path/to/your/file.stl'
# grid_size = (100, 100, 100)  # Define the resolution of the binary mask
# try:
#     solid_mask = create_solid_mask_from_stl(stl_file_path, grid_size)
#     # Save the solid mask
#     sitk.WriteImage(solid_mask, 'solid_mask.nrrd')
# except ValueError as e:
#     print(e)



# def define_plane(point1, point2, point3):
#     # Create vectors from the points
#     vector1 = point2 - point1
#     vector2 = point3 - point1
#     # Calculate the normal vector to the plane
#     normal = np.cross(vector1, vector2)
#     d = -np.dot(normal, point1)  # d in the plane equation Ax + By + Cz + D = 0
#     return normal, d


# def cut_mask_with_plane(mask, point1, point2, point3):
#     # Get the shape of the mask
#     mask_array = sitk.GetArrayFromImage(mask)
#     z, y, x = mask_array.shape
#
#     # Define the plane using the three points
#     normal, d = define_plane(point1, point2, point3)
#
#     # Create a new mask for the cut
#     cut_mask_array = np.copy(mask_array)
#
#     # Loop through each voxel in the mask
#     for i in range(z):
#         for j in range(y):
#             for k in range(x):
#                 # Get the coordinates of the voxel
#                 voxel_coord = np.array([k, j, i])
#
#                 # Calculate the distance from the voxel to the plane
#                 distance = np.dot(normal, voxel_coord) + d
#
#                 # Modify the mask based on the distance
#                 if distance > 0:  # Voxel is above the plane
#                     cut_mask_array[i, j, k] = 0  # Set to 0 (cutting away)
#
#     # Create a new SimpleITK image from the modified array
#     cut_mask = sitk.GetImageFromArray(cut_mask_array)
#     cut_mask.CopyInformation(mask)  # Retain original mask info (spacing, origin, etc.)
#
#     return cut_mask
#
#
# # Example usage
# # Load your binary mask (example path)
# binary_mask_path = 'path/to/your/binary_mask.nrrd'
# binary_mask = sitk.ReadImage(binary_mask_path)
#
# # Define the points in 3D space (make sure they are in the correct coordinate system)
# point1 = np.array([x1, y1, z1])
# point2 = np.array([x2, y2, z2])
# point3 = np.array([x3, y3, z3])
#
# # Cut the mask
# cut_mask = cut_mask_with_plane(binary_mask, point1, point2, point3)
#
# # Save the cut mask
# sitk.WriteImage(cut_mask, 'cut_mask.nrrd')



# def stl_to_mask(stl_path, reference_nii_path, output_nii_path):
#     # Загрузка исходного nii файла для получения параметров изображения
#     reference_image = sitk.ReadImage(reference_nii_path)
#     image_size = reference_image.GetSize()           # Размеры изображения
#     image_spacing = reference_image.GetSpacing()     # Размеры пикселей (вокселей)
#     image_origin = reference_image.GetOrigin()       # Начало координат
#     image_direction = reference_image.GetDirection() # Ориентация
#
#     # Создаем пустое изображение для маски с такими же параметрами, как у исходного файла
#     mask_image = sitk.Image(image_size, sitk.sitkUInt8)
#     mask_image.SetSpacing(image_spacing)
#     mask_image.SetOrigin(image_origin)
#     mask_image.SetDirection(image_direction)
#
#     # Загрузка STL файла
#     mesh = trimesh.load_mesh(stl_path)
#
#     # Создаем сетку координат для размещения маски в пространстве исходного изображения
#     x = np.arange(0, image_size[0]) * image_spacing[0] + image_origin[0]
#     y = np.arange(0, image_size[1]) * image_spacing[1] + image_origin[1]
#     z = np.arange(0, image_size[2]) * image_spacing[2] + image_origin[2]
#     xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
#     coords = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
#
#     # Определяем, какие из этих точек находятся внутри STL-модели
#     points_inside = mesh.contains(coords).reshape(image_size)
#
#     # Преобразуем массив точек в изображение маски
#     mask_image_np = sitk.GetArrayFromImage(mask_image)
#     mask_image_np[points_inside] = 1
#
#     # Конвертируем массив обратно в SimpleITK изображение и сохраняем как NIfTI
#     mask_image = sitk.GetImageFromArray(mask_image_np.astype(np.uint8))
#     mask_image.SetSpacing(image_spacing)
#     mask_image.SetOrigin(image_origin)
#     mask_image.SetDirection(image_direction)
#     sitk.WriteImage(mask_image, output_nii_path)



# def stl_resample(input_stl_path, output_stl_path, target_vertex_count):
#     def load_stl(file_path):
#         return mesh.Mesh.from_file(file_path)
#
#
#     def reduce_vertices(original_mesh, target_vertex_count):
#         # Flatten the vertex array
#         vertices = original_mesh.vectors.reshape(-1, 3)
#
#         # Use k-NN to cluster vertices and reduce
#         nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(vertices)
#         distances, indices = nbrs.kneighbors(vertices)
#
#         # Simplify by averaging neighbors
#         new_vertices = []
#         unique_indices = set()
#
#         for idx in range(len(vertices)):
#             if idx in unique_indices:
#                 continue
#             neighbor_indices = indices[idx]
#             neighbor_indices = [i for i in neighbor_indices if i not in unique_indices]
#
#             # Calculate the average position
#             new_vertex = np.mean(vertices[neighbor_indices], axis=0)
#             new_vertices.append(new_vertex)
#
#             unique_indices.update(neighbor_indices)
#
#         new_vertices = np.array(new_vertices)
#
#         # Create new faces based on the reduced vertices
#         vertex_map = {tuple(v): i for i, v in enumerate(new_vertices)}
#         new_faces = []
#
#         for face in original_mesh.vectors:
#             new_face = [vertex_map[tuple(vertex)] for vertex in face]
#             new_faces.append(new_face)
#
#         new_faces = np.array(new_faces)
#
#         # Create a new mesh
#         new_mesh = mesh.Mesh(np.zeros(new_faces.shape[0], dtype=mesh.Mesh.dtype))
#         for i, face in enumerate(new_faces):
#             for j in range(3):
#                 new_mesh.vectors[i][j] = new_vertices[face[j]]
#
#         return new_mesh
#
#
#     def save_stl(mesh, file_path):
#         mesh.save(file_path)
#
#     original_mesh = load_stl(input_stl_path)
#     reduced_mesh = reduce_vertices(original_mesh, target_vertex_count)
#     save_stl(reduced_mesh, output_stl_path)


# def stl_resample(input_stl_path, output_stl_path, target_vertex_count):
#     # Загрузить STL файл
#     mesh = o3d.io.read_triangle_mesh(input_stl_path)
#
#     # Упрощение модели до нужного количества вершин
#     simplified_mesh = mesh.simplify_vertex_clustering(
#         voxel_size=mesh.get_max_bound().max() / (target_vertex_count ** (1/3))
#     )
#
#     # Убедимся, что мы достигли нужного количества вершин
#     while len(simplified_mesh.vertices) > target_vertex_count:
#         simplified_mesh = simplified_mesh.simplify_quadric_decimation(target_vertex_count)
#
#     # Сохранить упрощенную модель
#     o3d.io.write_triangle_mesh(output_stl_path, simplified_mesh)


def stl_resample(input_stl_path, output_stl_path, target_vertex_count):
    # Загрузка STL файла
    mesh = o3d.io.read_triangle_mesh(input_stl_path)

    # Проверка успешной загрузки
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        add_info_logging("Error: The file is empty or does not contain valid data.", "work_logger")
        return

    # Начальный размер вокселя - меньшая доля от общего размера модели
    initial_voxel_size = (mesh.get_max_bound() - mesh.get_min_bound()).max() / 1000
    voxel_size = initial_voxel_size  # начнем с малого значения
    add_info_logging(f"Initial voxel size: {voxel_size}", "work_logger")

    # Итеративное упрощение с кластеризацией вершин
    simplified_mesh = mesh
    while len(simplified_mesh.vertices) > target_vertex_count and voxel_size < (
            mesh.get_max_bound() - mesh.get_min_bound()).max():
        simplified_mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size)
        voxel_size *= 1.1  # поэтапно увеличиваем размер вокселя

    add_info_logging(f"Final voxel size: {voxel_size}", "work_logger")

    # Проверка после кластеризации
    if len(simplified_mesh.vertices) == 0 or len(simplified_mesh.triangles) == 0:
        add_info_logging("Error: The simplified model is empty after clustering.", "work_logger")
        return

    # Если все еще больше целевого количества, применяем квадратичное упрощение
    if len(simplified_mesh.vertices) > target_vertex_count:
        simplified_mesh = simplified_mesh.simplify_quadric_decimation(target_vertex_count)

    # Вычисление нормалей перед сохранением в STL
    simplified_mesh.compute_triangle_normals()
    simplified_mesh.compute_vertex_normals()

    # Сохранение упрощенной модели
    success = o3d.io.write_triangle_mesh(output_stl_path, simplified_mesh)
    if success:
        add_info_logging(f"File saved successfully: {output_stl_path}", "work_logger")
    else:
        add_info_logging("Error saving STL file.", "work_logger")


# def __vtk_image_to_sitk(vtk_image):
#     """Convert vtkImageData to SimpleITK Image."""
#     # Step 1: Extract VTK image dimensions, spacing, and origin
#     dimensions = vtk_image.GetDimensions()
#     spacing = vtk_image.GetSpacing()
#     origin = vtk_image.GetOrigin()
#     np.bool = np.bool_
#     # Step 2: Get VTK image data as a NumPy array
#     vtk_array = vtk_image.GetPointData().GetScalars()  # Get the VTK data array
#     np_array = vtk_to_numpy(vtk_array).astype(np.uint8)  # Convert VTK array to NumPy array
#     np_array = np_array.reshape(dimensions[::-1])  # Reshape to match SimpleITK's z, y, x order
#
#     # Step 3: Convert NumPy array to SimpleITK image
#     sitk_image = sitk.GetImageFromArray(np_array)
#     sitk_image.SetSpacing(spacing)
#     sitk_image.SetOrigin(origin)
#
#     # Step 4: Set Direction (assuming identity if not available)
#     # VTK doesn't provide direction, so we often assume identity, or you can define it as needed.
#     sitk_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
#
#     return sitk_image
#
# def convert_STL_to_mask(stl_file, reference_image_file, destination_mask_file):
#     """Convert an STL file to a binary mask that matches the reference image grid."""
#     # Step 1: Load the STL file using VTK
#     reader = vtk.vtkSTLReader()
#     reader.SetFileName(stl_file)
#     reader.Update()
#     poly_data = reader.GetOutput()
#
#     # Step 2: Get the reference image properties
#     reference_image = sitk.ReadImage(reference_image_file)
#     ref_spacing = reference_image.GetSpacing()
#     ref_origin = reference_image.GetOrigin()
#     ref_direction = reference_image.GetDirection()
#     ref_size = reference_image.GetSize()
#
#     # Step 3: Set up VTK to match the reference image grid
#     # Define VTK image with the same dimensions and spacing
#     vtk_image = vtk.vtkImageData()
#     vtk_image.SetDimensions(ref_size)
#     vtk_image.SetSpacing(ref_spacing)
#     vtk_image.SetOrigin(ref_origin)
#
#     # Step 4: Use VTK's PolyDataToImageStencil to convert STL to a mask
#     # Create a stencil from the STL poly data
#     stencil = vtk.vtkPolyDataToImageStencil()
#     stencil.SetInputData(poly_data)
#     stencil.SetOutputOrigin(vtk_image.GetOrigin())
#     stencil.SetOutputSpacing(vtk_image.GetSpacing())
#     stencil.SetOutputWholeExtent(vtk_image.GetExtent())
#     stencil.Update()
#
#     # Step 5: Convert stencil to binary mask in VTK
#     # Initialize an empty binary mask in VTK and apply stencil to it
#     stencil_to_image = vtk.vtkImageStencilToImage()
#     stencil_to_image.SetInputConnection(stencil.GetOutputPort())
#     stencil_to_image.SetInsideValue(1)  # Set voxel value inside the STL surface
#     stencil_to_image.SetOutsideValue(0)  # Set voxel value outside the STL surface
#     stencil_to_image.Update()
#
#     # Step 6: Convert VTK image to SimpleITK image
#     # Get the numpy array from the VTK image
#     #vtk_to_numpy = vtk.util.numpy_support.vtk_to_numpy
#     #mask_array = vtk_to_numpy(stencil_to_image.GetOutput().GetPointData().GetScalars())
#     #mask_array = mask_array.reshape(ref_size[::-1])  # Reshape to match SimpleITK
#
#     # Convert numpy array to SimpleITK image
#     #mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
#     #mask_image.SetSpacing(ref_spacing)
#     #mask_image.SetOrigin(ref_origin)
#     #mask_image.SetDirection(ref_direction)
#     res_image = DataProcessing_VS.__vtk_image_to_sitk(stencil_to_image.GetOutput())
#     sitk.WriteImage(res_image, destination_mask_file)
#     pass


def convert_stl_to_mask_nii(stl_file, reference_image_file, destination_mask_file):
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
    res_image = vtk_image_to_sitk(stencil_to_image.GetOutput())
    sitk.WriteImage(res_image, destination_mask_file)


def cut_mask_using_points(source_mask_file, updated_mask_file, top_points, bottom_points, margin=5):
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

    def add_safety_margin_to_plane(a, b, c, d, normal, margin):
        """Add a safety margin to the plane by shifting it by `margin` along the normal direction."""
        def calculate_max_margin(normal, d, binary_mask):
            # Get the bounding box of the mask in world coordinates
            size = binary_mask.GetSize()
            spacing = binary_mask.GetSpacing()
            origin = binary_mask.GetOrigin()

            # Compute the max and min corners of the bounding box
            max_corner = np.array(origin) + np.array(size) * np.array(spacing)
            min_corner = np.array(origin)

            # Project corners onto the normal to find the maximum possible margin
            distances = []
            for corner in [min_corner, max_corner]:
                distance = abs(np.dot(normal, corner) + d) / np.linalg.norm(normal)
                distances.append(distance)

            return min(distances)

        # Normalize the normal vector
        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            logging.info("The normal vector is zero; the points may be collinear.")
            # raise ValueError("The normal vector is zero; the points may be collinear.")

        max_margin = calculate_max_margin(normal, d, binary_mask)
        effective_margin = min(margin, max_margin)  # Use the smaller of the desired or maximum possible margin
        if margin > effective_margin:
            logging.info(f"Using effective margin: {effective_margin} (requested: {margin}, max possible: {max_margin})")
            # log_file.write(f"Using effective margin: {effective_margin} (requested: {margin}, max possible: {max_margin})")

        # Adjust d to move the plane by `margin` along the normal direction
        d_with_margin = d + (margin * normal_length)
        return a, b, c, d_with_margin

    def cut_mask_above_planes(binary_mask, lower_plane, upper_plane):
        """Set voxels above the plane ax + by + cz + d > 0 to zero in the binary mask."""
        # Get the size and spacing of the mask
        size = binary_mask.GetSize()
        spacing = binary_mask.GetSpacing()
        origin = binary_mask.GetOrigin()
        direction = binary_mask.GetDirection()

        # Convert the SimpleITK image to a NumPy array for easy manipulation
        mask_array = sitk.GetArrayFromImage(binary_mask)

        non_zero_indices = np.nonzero(mask_array)
        # Combine indices into a list of (z, y, x) coordinates
        non_zero_coordinates = list(zip(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]))

        # Iterate over each voxel and calculate its world coordinates
        for t in range(len(non_zero_coordinates)):
            x, y, z = non_zero_coordinates[t][2], non_zero_coordinates[t][1], non_zero_coordinates[t][0]
            # Compute the world coordinates of the voxel
            world_x = origin[0] + x * spacing[0]
            world_y = origin[1] + y * spacing[1]
            world_z = origin[2] + z * spacing[2]

            # Check if the point is above the plane
            if lower_plane[0] * world_x + lower_plane[1] * world_y + lower_plane[2] * world_z + lower_plane[3] < 5:
                mask_array[z, y, x] = 0  # Set the voxel to 0
            if upper_plane[0] * world_x + upper_plane[1] * world_y + upper_plane[2] * world_z + upper_plane[3] < 5:
                mask_array[z, y, x] = 0  # Set the voxel to 0

        # Convert the modified array back to a SimpleITK image
        cut_mask = sitk.GetImageFromArray(mask_array)
        cut_mask.CopyInformation(binary_mask)
        return cut_mask

    def check_points_within_mask(points, binary_mask):
        """Check if all given points are within the binary mask."""
        spacing = binary_mask.GetSpacing()
        origin = binary_mask.GetOrigin()
        size = binary_mask.GetSize()

        mask_array = sitk.GetArrayFromImage(binary_mask)

        for point in points:
            # Convert world coordinates to index coordinates
            index = [
                int(round((point[0] - origin[0]) / spacing[0])),
                int(round((point[1] - origin[1]) / spacing[1])),
                int(round((point[2] - origin[2]) / spacing[2]))
            ]

            # Check if index is within bounds
            if not (0 <= index[0] < size[0] and 0 <= index[1] < size[1] and 0 <= index[2] < size[2]):
                logging.info(f"Point {point} is out of the bounds of the mask.")
                # log_file.write(f"Point {point} is out of the bounds of the mask.")
                # return False

            # Check if the point is inside the mask (non-zero in mask array)
            if mask_array[index[2], index[1], index[0]] == 0:
                logging.info(f"Point {point} is outside the mask region (zero in mask).")
                # log_file.write(f"Point {point} is outside the mask region (zero in mask).")
                # return False

        # return True

    binary_mask = sitk.ReadImage(source_mask_file)

    # Checking that all points are inside the mask
    # logging.info("top")
    # check_points_within_mask(top_points, binary_mask)
    # logging.info("bottom")
    # check_points_within_mask(bottom_points, binary_mask)
        # logging.info("")
        # logging.info("Some points are outside the bounds of the mask. Please adjust the points.")
        # raise ValueError("Some points are outside the bounds of the mask. Please adjust the points.")

    a1, b1, c1, d1, normal1 = find_plane_coefficients(top_points[0], top_points[1], top_points[2])
    # logging.info("top")
    a1, b1, c1, d1 = add_safety_margin_to_plane(a1, b1, c1, d1, normal1, margin)

    lower_plane = [a1, b1, c1, d1]

    a2, b2, c2, d2, normal2 = find_plane_coefficients(bottom_points[0], bottom_points[1], bottom_points[2])
    # logging.info("bottom")
    a2, b2, c2, d2 = add_safety_margin_to_plane(a2, b2, c2, d2, normal2, margin)
    upper_plane = [a2, b2, c2, d2]

    # Find the plane coefficients

    # Cut the mask above the plane
    cut_mask = cut_mask_above_planes(binary_mask, lower_plane, upper_plane)

    # Save the modified mask
    sitk.WriteImage(cut_mask, updated_mask_file)
