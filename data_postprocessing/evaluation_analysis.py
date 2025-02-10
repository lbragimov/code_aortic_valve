import os
import numpy as np
import SimpleITK as sitk


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
            "Right-Left Commissural Width (RLC-LNC)": self.__euclidean_dist(self.landmarks["RLC_land"],
                                                                            self.landmarks["LNC_land"]),
            "Right-Non Commissural Width (RNC-RLC)": self.__euclidean_dist(self.landmarks["RNC_land"],
                                                                           self.landmarks["RLC_land"]),
            "Left-Non Commissural Width (LNC-RNC)": self.__euclidean_dist(self.landmarks["LNC_land"],
                                                                          self.landmarks["RNC_land"]),
        }

        # Compute leaflet distances
        leaflet_distances = {
            "R-L Distance": self.__euclidean_dist(self.landmarks["R_land"], self.landmarks["L_land"]),
            "R-N Distance": self.__euclidean_dist(self.landmarks["R_land"], self.landmarks["N_land"]),
            "L-N Distance": self.__euclidean_dist(self.landmarks["L_land"], self.landmarks["N_land"]),
        }

        # Compute commissural angles
        commissural_angles = {
            "Right-Left Commissural Angle": self.__compute_angle(self.landmarks["RLC_land"], self.landmarks["LNC_land"],
                                                                 self.landmarks["N_land"]),
            "Right-Non Commissural Angle": self.__compute_angle(self.landmarks["RNC_land"], self.landmarks["RLC_land"],
                                                                self.landmarks["LNC_land"]),
            "Left-Non Commissural Angle": self.__compute_angle(self.landmarks["LNC_land"], self.landmarks["RNC_land"],
                                                               self.landmarks["RLC_land"]),
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
        measurements['leaflet_distances'] = leaflet_distances
        measurements['commissural_angles'] = commissural_angles
        measurements['aortic_valve_area'] = aortic_valve_area
        measurements['ellipticity_ratio'] = ellipticity_ratio
        return measurements


class landmarking_testSimualtion:

    def __init__(self, file_original_nii, file_probability_map, output_directory):
        # Load probability maps
        data = np.load(file_probability_map, allow_pickle=True)
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
        origin = np.array(mask_image.GetOrigin())  # World coordinate of (0,0,0)
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

        # R_land, L_land, N_land, RLC_land, RNC_land, LNC_land
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
        probs = {}  # Store corresponding probabilities

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
                sampled_index = np.random.choice(len(probs[key]), p=probs[key])  # Weighted random selection
                sampled_landmarks[key] = nonzero_locs[key][sampled_index]  # Get the selected voxel

            # Step 4: Compute measurement (Replace with actual function)
            measurement = self.__measure_metrics(sampled_landmarks)

            # Step 5: Store weighted measurement (measurement * probability product)
            weight = np.prod([probs[key][sampled_index] for key in self.probability_maps.keys()])
            results.append(measurement * weight)

        return results