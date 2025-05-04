import os
import numpy as np
import SimpleITK as sitk
import json
from scipy.ndimage import center_of_mass
from sklearn.metrics import jaccard_score, f1_score
from typing import Literal, Tuple
from data_preprocessing.text_worker import add_info_logging


class landmarking_MonteCarlo:

    def __init__(self, json_file, nii_file, npy_file):
        # Load the JSON file with landmark coordinates
        self.N = 10  # Total candidate points per landmark
        self.N_land = 6  # Total landmarks
        self.X = 50  # Percentage for selecting the center of mass (in %)
        self.num_simulations = 1000  # Number of Monte Carlo simulations
        self.lanmdark_names = {0: 'R', 1: 'L', 2: 'N', 3: 'RLC', 4: 'RNC', 5: 'LNC'}
        with open(json_file, 'r') as f:
            self.landmarks_ref = json.load(f)

        self.landmarks = []
        for i in range(1, self.N_land + 1):
            temp, spacing, origin, direction = self.__load_probability_map(nii_file)
            prob_map_all = np.load(npy_file)
            prob_map = prob_map_all["probabilities"][i]
            candidate_points, probabilities = self.__get_candidate_points(prob_map, spacing, origin, direction)
            self.landmarks.append({'candidate_points': candidate_points, 'probabilities': probabilities})

    # Load probability map
    def __load_probability_map(self, filepath):
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)  # Convert to numpy array
        spacing = np.array(image.GetSpacing())  # Get voxel spacing
        origin = np.array(image.GetOrigin())  # Get image origin
        direction = np.array(image.GetDirection()).reshape(3, 3)  # Convert to 3x3 matrix
        return array, spacing, origin, direction

    # Convert voxel coordinates to world coordinates
    def __voxel_to_world(self, voxel_coords, spacing, origin, direction):
        world_coords = np.dot(direction, voxel_coords * spacing) + origin
        return world_coords

    # Get candidate points from probability map
    def __get_candidate_points(self, prob_map, spacing, origin, direction):
        # Find Center of Mass (CoM)
        com_voxel = np.array(center_of_mass(prob_map)).astype(float)  # Voxel coordinates

        # Get top N-1 highest probability locations
        prob_flat = prob_map.ravel()
        top_indices = np.argpartition(prob_flat, -self.N)[-self.N:]  # Get indices of top N values
        top_indices = top_indices[np.argsort(prob_flat[top_indices])][::-1]  # Sort descending

        # Convert flattened indices back to 3D coordinates
        top_voxel_coords = np.array(np.unravel_index(top_indices, prob_map.shape)).T  # Convert to voxel coords

        # Ensure CoM is the first candidate
        candidate_voxels = np.vstack([com_voxel, top_voxel_coords.astype(float)])

        # Remove duplicates (if CoM is already in top points)
        candidate_voxels = candidate_voxels[:self.N]  # Keep only N points
        candidate_voxels = candidate_voxels[:, [2, 1, 0]]  # Swap to (x, y, z)

        selected_world_coords = np.array(
            [self.__voxel_to_world(p, spacing, origin, direction) for p in candidate_voxels])

        p_com = self.X / 100
        p_others = (100 - self.X) / (self.N - 1) / 100
        probabilities = [p_com] + [p_others] * (self.N - 1)
        return selected_world_coords, probabilities

    def __compute_3d_angle(self, A, B, C):
        """
        Compute the angle between three points A, B, and C in 3D space.
        B is the vertex (middle point).

        Parameters:
            A, B, C: np.array([x, y, z]) - 3D coordinates of the points

        Returns:
            angle_radians: Angle in radians
            angle_degrees: Angle in degrees
        """
        # Compute vectors BA and BC
        BA = A - B
        BC = C - B

        # Compute dot product and magnitudes
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)

        # Compute cosine of the angle
        cos_theta = np.clip(dot_product / (magnitude_BA * magnitude_BC), -1.0, 1.0)

        # Compute angle in radians
        angle_radians = np.arccos(cos_theta)

        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def run_simulation(self):
        statistics_dists = []
        statistics_angles = []

        for t in range(0, self.N_land):
            for k in range(t + 1, self.N_land):
                for r in range(k + 1, self.N_land):
                    measurements = []
                    for _ in range(self.num_simulations):
                        selected_points = []
                        for i in range(0, self.N_land):
                            selected_index = np.random.choice(len(self.landmarks[i]['candidate_points']),
                                                              p=self.landmarks[i]['probabilities'])
                            selected_voxel = self.landmarks[i]['candidate_points'][selected_index]
                            selected_points.append(selected_voxel)
                        # measurements.append(np.linalg.norm(selected_points[t] - selected_points[k]))
                        measurements.append(
                            self.__compute_3d_angle(selected_points[t], selected_points[k], selected_points[r]))

                    ref_measure = self.__compute_3d_angle(np.asarray(self.landmarks_ref[self.lanmdark_names[t]]),
                                                          np.asarray(self.landmarks_ref[self.lanmdark_names[k]]),
                                                          np.asarray(self.landmarks_ref[self.lanmdark_names[r]]))
                    mean1 = np.mean(measurements)
                    std1 = np.std(measurements)
                    statistics_angles.append([np.abs(mean1 - ref_measure), np.abs(measurements[0] - ref_measure), std1])
            pass

        for t in range(0, self.N_land):
            for k in range(t + 1, self.N_land):
                measurements = []
                for _ in range(self.num_simulations):
                    selected_points = []
                    for i in range(0, self.N_land):
                        selected_index = np.random.choice(len(self.landmarks[i]['candidate_points']),
                                                          p=self.landmarks[i]['probabilities'])
                        selected_voxel = self.landmarks[i]['candidate_points'][selected_index]
                        selected_points.append(selected_voxel)
                    # measurements.append(np.linalg.norm(selected_points[t] - selected_points[k]))
                    measurements.append(np.linalg.norm(selected_points[t] - selected_points[k]))

                ref_measure = np.linalg.norm(np.asarray(self.landmarks_ref[self.lanmdark_names[t]]) - np.asarray(
                    self.landmarks_ref[self.lanmdark_names[k]]))
                mean1 = np.mean(measurements)
                std1 = np.std(measurements)
                statistics_dists.append([np.abs(mean1 - ref_measure), np.abs(measurements[0] - ref_measure), std1])
            pass
        pass

        # add_info_logging(f"angles = '{np.mean(statistics_angles, axis=0)}'", "result_logger")
        # add_info_logging(f"distances = '{np.mean(statistics_dists, axis=0)}'", "result_logger")
        return np.mean(statistics_angles, axis=0), np.mean(statistics_dists, axis=0)

# simulation = landmarking_testSimualtion(heart_nnUnet + '/Landmarking/temp/result/HOM_M32_H185_W90_YA.nii.gz', heart_nnUnet + '/Landmarking/temp/result/HOM_M32_H185_W90_YA.npz', heart_nnUnet + '/Landmarking/temp/temp1/')
# simulation = landmarking_MonteCarlo(heart_data + '/json_markers_info/Homburg pathology/HOM_M32_H185_W90_YA.json', heart_nnUnet + '/Landmarking/temp/temp1/')
# simulation.run_simulation()


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


class landmarking_testing:

    def compute_metrics_direct_nii(self, mask_nii):
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
        # measurerer = landmarking_computeMeasurements(centers_of_mass[1], centers_of_mass[2], centers_of_mass[3],
        #                                              centers_of_mass[4], centers_of_mass[5], centers_of_mass[6])
        # metrics = measurerer.compute_metrics()
        return centers_of_mass

    def compute_metrics_direct_npz(self, mask_nii, mask_npz):
        # Get image metadata
        mask_image = sitk.ReadImage(mask_nii)
        spacing = np.array(mask_image.GetSpacing())  # (x, y, z) voxel size
        origin = np.array(mask_image.GetOrigin())  # World coordinate of (0,0,0)
        direction = np.array(mask_image.GetDirection()).reshape(3, 3)  # Reshape to 3x3 matrix

        prob_map_all = np.load(mask_npz)
        labels = len(prob_map_all["probabilities"])

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
        for label in range(1, labels):
            binary_mask = prob_map_all["probabilities"][label]  # Create binary mask for current label
            binary_mask[binary_mask < np.max(binary_mask)*0.2] = 0
            center_world = compute_center_of_mass(binary_mask, spacing, origin, direction)
            if center_world is not None:
                centers_of_mass[label] = center_world

        # R_land, L_land, N_land, RLC_land, RNC_land, LNC_land
        # measurerer = landmarking_computeMeasurements(centers_of_mass[1], centers_of_mass[2], centers_of_mass[3],
        #                                              centers_of_mass[4], centers_of_mass[5], centers_of_mass[6])
        # metrics = measurerer.compute_metrics()
        return centers_of_mass


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


def evaluate_segmentation(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = 1,
                          average: Literal['macro', 'weighted'] = 'macro'):
    """
    Вычисляет Dice и IoU между масками.

    Parameters:
        true_mask (np.ndarray): Ground truth маска (2D или 3D).
        pred_mask (np.ndarray): Предсказанная маска (2D или 3D).
        num_classes (int): Количество классов. Если 1 — бинарная сегментация.
        average (str): Способ усреднения ('macro' или 'weighted').

    Returns:
        dict: {'Dice': ..., 'IoU': ...}
    """
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()

    if num_classes == 1:
        dice = f1_score(true_flat, pred_flat)
        iou = jaccard_score(true_flat, pred_flat)
    else:
        dice = f1_score(true_flat, pred_flat, average=average, labels=range(num_classes))
        iou = jaccard_score(true_flat, pred_flat, average=average, labels=range(num_classes))

    return {
        'Dice': dice,
        'IoU': iou
    }


def hausdorff_distance_sitk(mask1: np.ndarray, mask2: np.ndarray,
                            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """
    Вычисляет Hausdorff и HD95 расстояния между двумя 3D масками с учетом физического размера вокселя.

    Parameters:
        mask1 (np.ndarray): Ground truth бинарная 3D маска.
        mask2 (np.ndarray): Предсказанная бинарная 3D маска.
        spacing (tuple): Физический размер вокселя (мм), по осям (z, y, x).

    Returns:
        dict: {'Hausdorff': ..., 'HD95': ...}
    """
    mask1_itk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_itk = sitk.GetImageFromArray(mask2.astype(np.uint8))

    mask1_itk.SetSpacing(spacing)
    mask2_itk.SetSpacing(spacing)

    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(mask1_itk, mask2_itk)

    return {
        'Hausdorff': hd_filter.GetHausdorffDistance(),
        'HD95': hd_filter.Get95PercentHausdorffDistance()
    }

