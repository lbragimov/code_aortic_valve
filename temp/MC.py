import numpy as np
import json
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
from scipy.ndimage import center_of_mass
import SimpleITK as sitk

class landmarking_MonteCarlo:

    def __init__(self, json_file, probability_map_folder):
        # Load the JSON file with landmark coordinates
        self.N = 10  # Total candidate points per landmark
        self.N_land = 6  # Total landmarks
        self.X = 50  # Percentage for selecting the center of mass (in %)
        self.num_simulations = 1000  # Number of Monte Carlo simulations
        self.lanmdark_names = {0:'R', 1:'L', 2:'N', 3:'RLC', 4:'RNC', 5:'LNC'}
        with open(json_file, 'r') as f:
            self.landmarks_ref = json.load(f)

        self.landmarks = []
        for i in range(1, self.N_land + 1):
            prob_map, spacing, origin, direction = self.__load_probability_map(probability_map_folder + 'class_' + str(i) + '_prob.nii.gz')
            candidate_points, probabilities = self.__get_candidate_points(prob_map, spacing, origin, direction)
            self.landmarks.append({'candidate_points':candidate_points, 'probabilities':probabilities})
        pass

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
    
        selected_world_coords = np.array([self.__voxel_to_world(p, spacing, origin, direction) for p in candidate_voxels])

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
        statistics_dists  = []
        statistics_angles = []

        for t in range(0, self.N_land):
            for k in range(t + 1, self.N_land):
                for r in range(k + 1, self.N_land):
                    measurements = []
                    for _ in range(self.num_simulations):
                        selected_points = []
                        for i in range(0, self.N_land):
                            selected_index = np.random.choice(len(self.landmarks[i]['candidate_points']), p = self.landmarks[i]['probabilities'])
                            selected_voxel = self.landmarks[i]['candidate_points'][selected_index]
                            selected_points.append(selected_voxel)
                        #measurements.append(np.linalg.norm(selected_points[t] - selected_points[k]))
                        measurements.append(self.__compute_3d_angle(selected_points[t], selected_points[k], selected_points[r]))

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
                        selected_index = np.random.choice(len(self.landmarks[i]['candidate_points']), p = self.landmarks[i]['probabilities'])
                        selected_voxel = self.landmarks[i]['candidate_points'][selected_index]
                        selected_points.append(selected_voxel)
                    #measurements.append(np.linalg.norm(selected_points[t] - selected_points[k]))
                    measurements.append(np.linalg.norm(selected_points[t] - selected_points[k]))

                ref_measure = np.linalg.norm(np.asarray(self.landmarks_ref[self.lanmdark_names[t]]) - np.asarray(self.landmarks_ref[self.lanmdark_names[k]]))
                mean1 = np.mean(measurements)
                std1 = np.std(measurements)
                statistics_dists.append([np.abs(mean1 - ref_measure), np.abs(measurements[0] - ref_measure), std1])
            pass
        pass
        
        print('angles = ', np.mean(statistics_angles, axis = 0))
        print('distances = ', np.mean(statistics_dists, axis = 0))


#simulation = landmarking_testSimualtion(heart_nnUnet + '/Landmarking/temp/result/HOM_M32_H185_W90_YA.nii.gz', heart_nnUnet + '/Landmarking/temp/result/HOM_M32_H185_W90_YA.npz', heart_nnUnet + '/Landmarking/temp/temp1/')
#simulation = landmarking_MonteCarlo(heart_data + '/json_markers_info/Homburg pathology/HOM_M32_H185_W90_YA.json', heart_nnUnet + '/Landmarking/temp/temp1/')
#simulation.run_simulation()