import os
import json
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
import logging
from datetime import datetime
from data_preprocessing.text_worker import add_info_logging


class LandmarkingMonteCarlo:

    def __init__(self, json_file, nii_file, npy_file):
        # Load the JSON file with landmark coordinates
        self.N = 10  # Total candidate points per landmark
        self.N_land = 6  # Total landmarks
        self.X = 50  # Percentage for selecting the center of mass (in %)
        self.num_simulations = 1000  # Number of Monte Carlo simulations
        self.landmark_names = {0: 'R', 1: 'L', 2: 'N', 3: 'RLC', 4: 'RNC', 5: 'LNC'}
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

                    ref_measure = self.__compute_3d_angle(np.asarray(self.landmarks_ref[self.landmark_names[t]]),
                                                          np.asarray(self.landmarks_ref[self.landmark_names[k]]),
                                                          np.asarray(self.landmarks_ref[self.landmark_names[r]]))
                    mean1 = np.mean(measurements)
                    std1 = np.std(measurements)
                    statistics_angles.append([np.abs(mean1 - ref_measure), np.abs(measurements[0] - ref_measure), std1])

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

                ref_measure = np.linalg.norm(np.asarray(self.landmarks_ref[self.landmark_names[t]]) - np.asarray(
                    self.landmarks_ref[self.landmark_names[k]]))
                mean1 = np.mean(measurements)
                std1 = np.std(measurements)
                statistics_dists.append([np.abs(mean1 - ref_measure), np.abs(measurements[0] - ref_measure), std1])

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


def experiment_analysis(data_path,
                        dict_case,
                        find_center_mass=False,
                        find_monte_carlo=False):
    for radius, case_name in dict_case.items():
        ds_folder_name = f"Dataset{case_name}_AortaLandmarks"
        process_analysis(data_path, ds_folder_name,
                         find_monte_carlo=find_monte_carlo,
                         probabilities_map=True)
        # data_path = Path(data_path)
        # result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
        # original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
        # json_path = data_path / "nnUNet_folder" / "json_info"


def process_analysis(data_path, ds_folder_name,
                     find_monte_carlo=False,
                     probabilities_map=False):
    # add_info_logging("start analysis", "work_logger")
    data_path = Path(data_path)
    result_landmarks_folder = data_path / "nnUNet_folder" / "nnUNet_test" / ds_folder_name
    original_mask_folder = data_path / "nnUNet_folder" / "original_mask" / ds_folder_name
    json_path = data_path / "nnUNet_folder" / "json_info"

    if find_monte_carlo:
        arr_mean_angles_ger_pat = np.array([]).reshape(0, 3)
        arr_mean_dists_ger_pat = np.array([]).reshape(0, 3)
        arr_mean_angles_slo_pat = np.array([]).reshape(0, 3)
        arr_mean_dists_slo_pat = np.array([]).reshape(0, 3)
        arr_mean_angles_slo_norm = np.array([]).reshape(0, 3)
        arr_mean_dists_slo_norm = np.array([]).reshape(0, 3)
        files = list(result_landmarks_folder.glob("*.npz"))
        for file in files:
            if file.name[0] == "H":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path/file.name[:-4]) + ".json",
                                                    nii_file=str(result_landmarks_folder/file.name[:-4]) + ".nii.gz" ,
                                                    npy_file=str(file))
                cur_angles, cur_dists =  simulation.run_simulation()
                arr_mean_angles_ger_pat = np.vstack([arr_mean_angles_ger_pat, cur_angles])
                arr_mean_dists_ger_pat = np.vstack([arr_mean_dists_ger_pat, cur_dists])
            if file.name[0] == "p":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
                                                    nii_file=str(result_landmarks_folder / file.name[:-4]) + ".nii.gz",
                                                    npy_file=str(file))
                cur_angles, cur_dists = simulation.run_simulation()
                arr_mean_angles_slo_pat = np.vstack([arr_mean_angles_slo_pat, cur_angles])
                arr_mean_dists_slo_pat = np.vstack([arr_mean_dists_slo_pat, cur_dists])
            if file.name[0] == "n":
                simulation = LandmarkingMonteCarlo(json_file=str(json_path / file.name[:-4]) + ".json",
                                                    nii_file=str(result_landmarks_folder / file.name[:-4]) + ".nii.gz",
                                                    npy_file=str(file))
                cur_angles, cur_dists = simulation.run_simulation()
                arr_mean_angles_slo_norm = np.vstack([arr_mean_angles_slo_norm, cur_angles])
                arr_mean_dists_slo_norm = np.vstack([arr_mean_dists_slo_norm, cur_dists])
        add_info_logging("German pathology")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_ger_pat, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_ger_pat, axis=0)}'")
        add_info_logging("Slovenian pathology")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_slo_pat, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_slo_pat, axis=0)}'")
        add_info_logging("Slovenian normal")
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles_slo_norm, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists_slo_norm, axis=0)}'")
        add_info_logging("Sum")
        arr_mean_angles = np.vstack([arr_mean_angles_ger_pat, arr_mean_angles_slo_pat, arr_mean_angles_slo_norm])
        arr_mean_dists = np.vstack([arr_mean_dists_ger_pat, arr_mean_dists_slo_pat, arr_mean_dists_slo_norm])
        add_info_logging(f"mean angles = '{np.mean(arr_mean_angles, axis=0)}'")
        add_info_logging(f"mean distances = '{np.mean(arr_mean_dists, axis=0)}'")


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    add_info_logging("Start", "work_logger")

    experiment_analysis(data_path=data_path,
                        dict_case={10: 491, 9: 499, 8: 498, 7: 497, 6: 496, 5: 495, 4: 494},
                        find_monte_carlo=True)
    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    current_time = datetime.now()
    # Логгер для хода программы
    log_file_name = current_time.strftime("log_%H_%M__%d_%m_%Y.log")
    work_log_path = os.path.join(data_path, log_file_name)
    work_logger = logging.getLogger("work_logger")
    work_logger.setLevel(logging.INFO)
    work_handler = logging.FileHandler(work_log_path, mode='w')
    work_handler.setFormatter(logging.Formatter('%(message)s'))
    work_logger.addHandler(work_handler)

    # Логгер для результатов
    result_file_name = current_time.strftime("result_%H_%M__%d_%m_%Y.log")
    result_log_path = os.path.join(data_path, result_file_name)
    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)
    result_handler = logging.FileHandler(result_log_path, mode='w')
    result_handler.setFormatter(logging.Formatter('%(message)s'))  # без времени
    result_logger.addHandler(result_handler)
    controller(data_path)