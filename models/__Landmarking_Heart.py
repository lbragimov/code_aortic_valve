import SimpleITK as sitk
import numpy as np
import json
from pathlib import Path
from subprocess import call
from scipy.spatial import distance
import os
import subprocess


class LH_processing:

    def __dist_landmarks():
        lanmdark_names = {'R', 'L', 'N', 'RLC', 'RNC', 'LNC'}
        radius_mm = 5
        return lanmdark_names, radius_mm

    def __is_voxel_inside_image(mask, voxel_coord):

        size = mask.GetSize()
        return all(0 <= voxel_coord[i] < size[i] for i in range(3))

    def __add_sphere(mask, center, value, radius_mm):
        """
        Adds a sphere to the mask at a given center voxel, considering non-uniform spacing.
        """
        xc, yc, zc = center  # Center voxel coordinates

        # Convert radius from mm to voxels in each dimension
        spacing = mask.GetSpacing()
        radius_vox = [int(np.ceil(radius_mm / spacing[0])), int(np.ceil(radius_mm / spacing[1])), int(np.ceil(radius_mm / spacing[2]))]

        for i in range (xc - radius_vox[0], xc + radius_vox[0] + 1):
            for j in range (yc - radius_vox[1], yc + radius_vox[1] + 1):
                for k in range (zc - radius_vox[2], zc + radius_vox[2] + 1):
                    voxel_coord = [i, j, k]
                    if np.sqrt(((i - xc) * spacing[0])**2 + ((j - yc) * spacing[0])**2 + ((k - zc) * spacing[0])**2) >= radius_mm:
                        continue
                    if LH_processing.__is_voxel_inside_image(mask, voxel_coord):
                        mask[voxel_coord] = value

    def __generate_landmark_case(json_file, nii_image_file, destination_file):

        lanmdark_names, radius_mm = LH_processing.__dist_landmarks()
        # Load the NIfTI image
        nii_image = sitk.ReadImage(nii_image_file)
        image_size = nii_image.GetSize()

        # Create an empty binary mask of the same size as the NIfTI image
        

        # Load the JSON file with landmark coordinates
        with open(json_file, 'r') as f:
            landmarks = json.load(f)

        # Function to convert physical coordinates to image indices
        def physical_to_index(image, physical_coords):
            return image.TransformPhysicalPointToIndex(physical_coords)


        binary_mask = sitk.Image(image_size, sitk.sitkUInt8)
        binary_mask.SetOrigin(nii_image.GetOrigin())
        binary_mask.SetSpacing(nii_image.GetSpacing())
        binary_mask.SetDirection(nii_image.GetDirection())

        #spacing = np.array(nii_image.GetSpacing())
        #binary_mask = np.zeros_like(sitk.GetArrayFromImage(nii_image), dtype=np.uint8)
        #mask_array = np.zeros_like(image_array, dtype=np.uint8)
        # Iterate over the landmarks and set the corresponding voxels in the binary mask
        land_ind = 1
        for landmark, coords in landmarks.items():
            if landmark in lanmdark_names:
                index = physical_to_index(nii_image, coords)
                #binary_mask[index] = 1
                LH_processing.__add_sphere(binary_mask, index, land_ind, radius_mm)
                #LH_processing.__add_sphere(binary_mask, index, land_ind, radius_mm, spacing)
                land_ind += 1

        #mask_image = sitk.GetImageFromArray(binary_mask)
        #mask_image.CopyInformation(nii_image)
        #sitk.WriteImage(mask_image, destination_file)
        sitk.WriteImage(binary_mask, destination_file)
        pass

    def generate_landmarks(json_folder, nii_image_folder, destination_folder):
        folder_path = Path(json_folder)
        num_files = sum(1 for file in folder_path.rglob("*") if file.is_file())

        for i in range(1, num_files + 1):
            LH_processing.__generate_landmark_case(json_folder + '/n' + str(i) + '.json', 
                                                   nii_image_folder + '/n' + str(i) + '.nii.gz', 
                                                   destination_folder + '/n' + str(i) + '.nii.gz')
            pass

    def rename_images_for_nnUnet(directory):
        return
        for filename in os.listdir(directory):
            if filename.endswith(".nii.gz"):
                old_path = os.path.join(directory, filename)
        
                # Rename file by inserting "000" before .nii.gz
                new_filename = filename.replace(".nii.gz", "_0000.nii.gz")
                new_path = os.path.join(directory, new_filename)
        
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
        pass

class nnUnet_Landmarking:

    def set_enviroment(folder):
        os.environ["nnUNet_raw"] = folder + "/database_raw"
        os.environ["nnUNet_preprocessed"] = folder + "/database_preprocessing"
        os.environ["nnUNet_results"] = folder + "/nnUnet_results"

        #folders = os.listdir(folder + "/database_raw" + '/Dataset001/imagesTr')
        pass

    def preprocessing():
        dataset_id = "001"  # Change to match your dataset
        command = [
            "nnUNetv2_plan_and_preprocess",
            '-d ' + dataset_id,
            "--verify_dataset_integrity",  # Optional: Save softmax predictions
        ]

        # Execute preprocessing
        try:
            print("Starting nnU-Net preprocessing...")
            call(command)
            print("Preprocessing completed successfully.")
        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
        pass

    def train():
        dataset_id = "001"  # Change to match your dataset
        command = [
            "nnUNetv2_train",
            dataset_id,
            "3d_fullres",
            "all"
            #str(0),
            #"--npz",  # Optional: Save softmax predictions
        ]

        # Execute the training
        try:
            print("Starting nnU-Net training...")
            call(command)
            print("Training completed successfully.")
        except Exception as e:
            print(f"An error occurred during training: {e}")
        pass

    def test(image_file, output_folder, model_folder):
        dataset_id = "001"  # Change to match your dataset
        command = [
            "nnUNetv2_predict",
            "-i", image_file,
            "-o", output_folder,
            "-d", dataset_id,
            "-c", "3d_fullres",
            "-f", "all"
        ]
        # Execute the training
        try:
            print("Starting nnU-Net testing...")
            call(command)
            print("Testing completed successfully.")
        except Exception as e:
            print(f"An error occurred during testing: {e}")
        pass

class landmarking_computeMeasurements:

    def __init__(self, R_land, L_land, N_land, RLC_land, RNC_land, LNC_land):
        self.landmarks = {}
        self.landmarks['R_land'] = R_land
        self.landmarks['L_land'] = L_land
        self.landmarks['N_land'] = N_land
        self.landmarks['RLC_land'] = RLC_land
        self.landmarks['RNC_land'] = RNC_land
        self.landmarks['LNC_land'] = LNC_land

    def __euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def __compute_angle(self, p1, p2, p3):
        ba = np.array(p1) - np.array(p2)
        bc = np.array(p3) - np.array(p2)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    # Function to compute the aortic valve orifice area using the Shoelace theorem
    def __compute_area(self, landmarks, points):
        coords = np.array([landmarks[p] for p in points])
        x, y = coords[:, 0], coords[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def compute_metrics(self):
        # Compute commissural distances
        commissural_widths = {
            "Right-Left Commissural Width (RLC-LNC)": self.__euclidean_dist(self.landmarks["RLC_land"], self.landmarks["LNC_land"]),
            "Right-Non Commissural Width (RNC-RLC)":  self.__euclidean_dist(self.landmarks["RNC_land"], self.landmarks["RLC_land"]),
            "Left-Non Commissural Width (LNC-RNC)":   self.__euclidean_dist(self.landmarks["LNC_land"], self.landmarks["RNC_land"]),
        }

        # Compute leaflet distances
        leaflet_distances = {
            "R-L Distance": self.__euclidean_dist(self.landmarks["R_land"], self.landmarks["L_land"]),
            "R-N Distance": self.__euclidean_dist(self.landmarks["R_land"], self.landmarks["N_land"]),
            "L-N Distance": self.__euclidean_dist(self.landmarks["L_land"], self.landmarks["N_land"]),
        }

        # Compute commissural angles
        commissural_angles = {
            "Right-Left Commissural Angle": self.__compute_angle(self.landmarks["RLC_land"], self.landmarks["LNC_land"], self.landmarks["N_land"]),
            "Right-Non Commissural Angle":  self.__compute_angle(self.landmarks["RNC_land"], self.landmarks["RLC_land"], self.landmarks["LNC_land"]),
            "Left-Non Commissural Angle":   self.__compute_angle(self.landmarks["LNC_land"], self.landmarks["RNC_land"], self.landmarks["RLC_land"]),
        }

        # Compute aortic valve orifice area
        aortic_valve_area = self.__compute_area(self.landmarks, ["R_land", "L_land", "N_land"])

        # Compute ellipticity ratio (ratio of max to min axis)
        landmark_coords = np.array(list(self.landmarks.values()))
        cov_matrix = np.cov(landmark_coords.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        ellipticity_ratio = max(eigenvalues) / min(eigenvalues)

        measurements = {}
        measurements['commissural_widths'] = commissural_widths
        measurements['leaflet_distances']  = leaflet_distances
        measurements['commissural_angles'] = commissural_angles
        measurements['aortic_valve_area']  = aortic_valve_area
        measurements['ellipticity_ratio']  = ellipticity_ratio
        return measurements

class landmarking_testSimualtion:

    def __init__(self, file_original_nii, file_probability_map, output_directory):
        # Load probability maps
        data = np.load(file_probability_map, allow_pickle = True)
        prob_maps = data["probabilities"]  # Shape: (6, H, W, D) for 6 classes

        # Load reference image for metadata (affine, spacing, origin)
        ref_img = sitk.ReadImage(file_original_nii)
        spacing = ref_img.GetSpacing()
        origin = ref_img.GetOrigin()
        direction = ref_img.GetDirection()

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Save each probability map as a separate NIfTI file
        for class_idx in range(prob_maps.shape[0]):
            prob_map = prob_maps[class_idx]  # Extract probability map for this class
            sitk_img = sitk.GetImageFromArray(prob_map)  # Convert numpy to SimpleITK image
    
            # Assign metadata from reference NIfTI
            sitk_img.SetSpacing(spacing)
            sitk_img.SetOrigin(origin)
            sitk_img.SetDirection(direction)

            # Save as NIfTI
            output_path = os.path.join(output_directory, f"class_{class_idx}_prob.nii.gz")
            sitk.WriteImage(sitk_img, output_path)
            print(f"Saved: {output_path}")

        print("All probability maps saved as NIfTI files.")

class landmarking_testing:

    def compute_metrics_direct(mask_nii):
        mask_image = sitk.ReadImage(mask_nii)
        mask_array = sitk.GetArrayFromImage(mask_image)  # Convert to NumPy array

        # Get image metadata
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z) voxel size
        origin = np.array(mask_image.GetOrigin())    # World coordinate of (0,0,0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)  # Reshape to 3x3 matrix

        # Find unique labels (excluding background 0)
        labels = np.unique(mask_array)
        labels = labels[labels != 0]  # Remove background if label 0 exists

        # Function to compute center of mass in world coordinates
        def compute_center_of_mass(binary_mask, spacing, origin, direction):
            indices = np.argwhere(binary_mask)  # Get voxel indices of the mask
            if len(indices) == 0:
                return None  # No center of mass if mask is empty

            # Compute the mean position in voxel space
            center_voxel = np.mean(indices, axis=0)[::-1]  # Reverse order (Z, Y, X) -> (X, Y, Z)

            # Convert to world coordinates using the corrected direction matrix
            center_world = np.dot(direction, center_voxel * spacing) + origin
            return center_world

        # Compute center of mass for each label
        centers_of_mass = {}
        for label in labels:
            binary_mask = (mask_array == label)  # Create binary mask for current label
            center_world = compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        #R_land, L_land, N_land, RLC_land, RNC_land, LNC_land
        measurerer = landmarking_computeMeasurements(centers_of_mass[1], centers_of_mass[2], centers_of_mass[3],
                                                     centers_of_mass[4], centers_of_mass[5], centers_of_mass[6])
        metrics = measurerer.compute_metrics()
        pass

class landmarking_locked:

    def __init__(self, probability_maps):
        self.probability_maps = probability_maps

    def __prepare_probability_lists(self):
        # Step 1: Extract nonzero probability locations for each landmark
        nonzero_locs = {}  # Store voxel locations where probability > 0
        probs = {}         # Store corresponding probabilities

        for key, prob_map in self.probability_maps.items():
            indices = np.argwhere(prob_map > 0)  # Get all non-zero locations
            probabilities = prob_map[prob_map > 0]  # Get probability values

            # Normalize probabilities to sum to 1 (so they form a proper distribution)
            probabilities /= np.sum(probabilities)

            # Store valid locations and their probabilities
            nonzero_locs[key] = indices
            probs[key] = probabilities

        return nonzero_locs, probs

    def __measure_metrics(self, landmarks):
        pass

    def simulation_MC(self, num_simulations):
        # Step 2: Monte Carlo Sampling
        results = []  # Store weighted measurements

        nonzero_locs, probs = self.__prepare_probability_lists()

        for _ in range(num_simulations):
            sampled_landmarks = {}

            # Step 3: Randomly select a voxel location for each landmark
            for key in self.probability_maps.keys():
                sampled_index = np.random.choice(len(probs[key]), p = probs[key])  # Weighted random selection
                sampled_landmarks[key] = nonzero_locs[key][sampled_index]  # Get the selected voxel

            # Step 4: Compute measurement (Replace with actual function)
            measurement = self.__measure_metrics(sampled_landmarks)

            # Step 5: Store weighted measurement (measurement * probability product)
            weight = np.prod([probs[key][sampled_index] for key in probability_maps.keys()])
            results.append(measurement * weight)

        return results


#razhnie funkcii dlya vipolneniya raznih chastej programmi
#LH_processing.generate_landmarks(heart_data + '/json_markers_info/Normal', heart_data + '/nii_convert/Normal', heart_data + '/landmark_masks/Normal')
#LH_processing.rename_images_for_nnUnet(heart_nnUnet + '/Landmarking/temp/test_image')#'/Landmarking/database_raw/Dataset001/imagesTs')

#return
#nnUnet_Landmarking.set_enviroment(heart_nnUnet + '/Landmarking')
#nnUnet_Landmarking.preprocessing()
#return
#nnUnet_Landmarking.train()
#nnUnet_Landmarking.test(heart_nnUnet + '/Landmarking/temp/test_image/',heart_nnUnet + '/Landmarking/temp/result',
#                        heart_nnUnet + '/Landmarking/nnUnet_results/Dataset001/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/')