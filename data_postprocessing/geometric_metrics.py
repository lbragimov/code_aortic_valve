import random
import numpy as np
from data_preprocessing.text_worker import add_info_logging


metric_name_function = {
    "inter_commissure_distance": "_inter_commissure_distance",
    "cusp_triangle_area": "_cusp_triangle_area",
    "sinotubular_diameter": "_sinotubular_diameter",
    "valve_height": "_valve_height",
}


def controller_metrics(metric_name, necessary_landmarks, found_landmarks, mc_option=False):
    # Получаем имя функции из словаря
    function_name = metric_name_function.get(metric_name)
    if function_name is None:
        add_info_logging(f"Unknown metric name: {metric_name}", "work_logger")
        raise ValueError(f"Unknown metric name: {metric_name}")

    # Получаем объект функции по имени
    function = globals().get(function_name)
    if function is None:
        add_info_logging(f"Function '{function_name}' is not defined", "work_logger")
        raise ValueError(f"Function '{function_name}' is not defined")

    results_list = []
    if not mc_option:
        for current_set in necessary_landmarks:
            sets_use_landmarks = []
            for new_point in current_set:
                sets_use_landmarks.append(found_landmarks[new_point])
            results_list.append(function(sets_use_landmarks))
    else:
        # Монте-Карло симуляции
        num_samples = 1000
        for current_set in necessary_landmarks:
            for _ in range(num_samples):
                sets_use_landmarks = []
                for new_point in current_set:
                    sets_use_landmarks.append(random.choice(found_landmarks[new_point]))
                results_list.append(function(sets_use_landmarks))

    return np.mean(np.asarray(results_list))


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