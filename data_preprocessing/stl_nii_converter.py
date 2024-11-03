import trimesh
import nibabel as nib
import numpy as np
import SimpleITK as sitk
# from stl import mesh
from stl import mesh
from sklearn.neighbors import NearestNeighbors



def convert_stl_to_nii(input_stl_file: str, output_nii_file: str, volume_shape: tuple[int]):
    # Load STL file
    mesh = trimesh.load_mesh(input_stl_file)

    # Create empty volume (e.g. 128x128x128)
    volume = np.zeros(volume_shape)

    # Convert STL coordinates to voxels
    for vertex in mesh.vertices:
        x, y, z = (
                (vertex - mesh.bounds[0]) / (mesh.bounds[1] - mesh.bounds[0]) * np.array(volume_shape)
        ).astype(int)
        volume[x-1, y-1, z-1] = 1  # Mark voxel as "occupied"

    # Save in NIfTI format
    nii_img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nii_img, output_nii_file)



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



def define_plane(point1, point2, point3):
    # Create vectors from the points
    vector1 = point2 - point1
    vector2 = point3 - point1
    # Calculate the normal vector to the plane
    normal = np.cross(vector1, vector2)
    d = -np.dot(normal, point1)  # d in the plane equation Ax + By + Cz + D = 0
    return normal, d


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



def stl_to_mask(stl_path, reference_nii_path, output_nii_path):
    # Загрузка исходного nii файла для получения параметров изображения
    reference_image = sitk.ReadImage(reference_nii_path)
    image_size = reference_image.GetSize()           # Размеры изображения
    image_spacing = reference_image.GetSpacing()     # Размеры пикселей (вокселей)
    image_origin = reference_image.GetOrigin()       # Начало координат
    image_direction = reference_image.GetDirection() # Ориентация

    # Создаем пустое изображение для маски с такими же параметрами, как у исходного файла
    mask_image = sitk.Image(image_size, sitk.sitkUInt8)
    mask_image.SetSpacing(image_spacing)
    mask_image.SetOrigin(image_origin)
    mask_image.SetDirection(image_direction)

    # Загрузка STL файла
    mesh = trimesh.load_mesh(stl_path)

    # Создаем сетку координат для размещения маски в пространстве исходного изображения
    x = np.arange(0, image_size[0]) * image_spacing[0] + image_origin[0]
    y = np.arange(0, image_size[1]) * image_spacing[1] + image_origin[1]
    z = np.arange(0, image_size[2]) * image_spacing[2] + image_origin[2]
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    coords = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T

    # Определяем, какие из этих точек находятся внутри STL-модели
    points_inside = mesh.contains(coords).reshape(image_size)

    # Преобразуем массив точек в изображение маски
    mask_image_np = sitk.GetArrayFromImage(mask_image)
    mask_image_np[points_inside] = 1

    # Конвертируем массив обратно в SimpleITK изображение и сохраняем как NIfTI
    mask_image = sitk.GetImageFromArray(mask_image_np.astype(np.uint8))
    mask_image.SetSpacing(image_spacing)
    mask_image.SetOrigin(image_origin)
    mask_image.SetDirection(image_direction)
    sitk.WriteImage(mask_image, output_nii_path)



def stl_resample(input_stl_path, output_stl_path, target_vertex_count):
    def load_stl(file_path):
        return mesh.Mesh.from_file(file_path)


    def reduce_vertices(original_mesh, target_vertex_count):
        # Flatten the vertex array
        vertices = original_mesh.vectors.reshape(-1, 3)

        # Use k-NN to cluster vertices and reduce
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(vertices)
        distances, indices = nbrs.kneighbors(vertices)

        # Simplify by averaging neighbors
        new_vertices = []
        unique_indices = set()

        for idx in range(len(vertices)):
            if idx in unique_indices:
                continue
            neighbor_indices = indices[idx]
            neighbor_indices = [i for i in neighbor_indices if i not in unique_indices]

            # Calculate the average position
            new_vertex = np.mean(vertices[neighbor_indices], axis=0)
            new_vertices.append(new_vertex)

            unique_indices.update(neighbor_indices)

        new_vertices = np.array(new_vertices)

        # Create new faces based on the reduced vertices
        vertex_map = {tuple(v): i for i, v in enumerate(new_vertices)}
        new_faces = []

        for face in original_mesh.vectors:
            new_face = [vertex_map[tuple(vertex)] for vertex in face]
            new_faces.append(new_face)

        new_faces = np.array(new_faces)

        # Create a new mesh
        new_mesh = mesh.Mesh(np.zeros(new_faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(new_faces):
            for j in range(3):
                new_mesh.vectors[i][j] = new_vertices[face[j]]

        return new_mesh


    def save_stl(mesh, file_path):
        mesh.save(file_path)

    original_mesh = load_stl(input_stl_path)
    reduced_mesh = reduce_vertices(original_mesh, target_vertex_count)
    save_stl(reduced_mesh, output_stl_path)
