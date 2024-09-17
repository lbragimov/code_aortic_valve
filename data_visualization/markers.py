import SimpleITK as sitk
import matplotlib.pyplot as plt

def load_image(image_path):
    return sitk.ReadImage(image_path)

def load_landmarks(file_path):
    landmarks = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            landmarks.append((x, y, z))
    return landmarks

def world_to_voxel_coords(image, world_coords):
    return [image.TransformPhysicalPointToIndex(point) for point in world_coords]

def extract_and_plot_slices(image, voxel_coords):
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

# Path to your NIFTI file
nii_path = 'path_to_your_nii_file.nii.gz'
# Path to your landmarks file
landmarks_path = 'path_to_your_landmarks_file.txt'

# Load the image and landmarks
image = load_image(nii_path)
landmarks = load_landmarks(landmarks_path)

# Convert landmarks to voxel coordinates
voxel_landmarks = world_to_voxel_coords(image, landmarks)

# Extract slices and plot
extract_and_plot_slices(image, voxel_landmarks)



def extract_and_save_slices(image, voxel_coords):
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

