import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance


class landmarking_computeMeasurements_simplified:

    class landmark_fit_plane:

        def __init__(self, points):
            self.points = points

        def fit(self):
            # Compute the centroid as the approximate center of the ellipse
            center_3D = np.mean(self.points, axis = 0)

            # Project points into 2D by finding the best fitting plane
            # Using Singular Value Decomposition (SVD) to determine the normal
            mean_centered_points = self.points - center_3D
            _, _, vh = np.linalg.svd(mean_centered_points)
            normal = vh[-1]
            return center_3D, normal

    class landmark_ellipse:
        
        def __init__(self, points):
            self.points = points

        # Define the regularized ellipse fitting function
        def __optimizer_step(self, params, center, points):
            a, b, theta = params
            h, k = center
            reg_weight = 0.1  # Regularization weight to enforce circular shape

            cost = 0
            for x, y in points:
                x_rot = np.cos(theta) * (x - h) + np.sin(theta) * (y - k)
                y_rot = -np.sin(theta) * (x - h) + np.cos(theta) * (y - k)
                ellipse_eq = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
                cost += ellipse_eq ** 2

            # Regularization term to encourage a almost =  b (making it more circular)
            cost += reg_weight * (a - b) ** 2
            return cost

        # Optimize
        def optimize(self):
            # Compute the centroid as the approximate center of the ellipse
            self.center_3D = np.mean(self.points, axis = 0)

            # Project points into 2D by finding the best fitting plane
            # Using Singular Value Decomposition (SVD) to determine the normal
            mean_centered_points = self.points - self.center_3D
            _, _, vh = np.linalg.svd(mean_centered_points)
            self.normal = vh[-1]

            # Choose two orthogonal axes within the plane
            axis_1 = vh[0]
            axis_2 = vh[1]

            # Convert 3D points to 2D coordinates on the plane
            points_2D = np.array([
                [np.dot(point - self.center_3D, axis_1), np.dot(point - self.center_3D, axis_2)]
                for point in self.points
            ])

            # Initial guess (assuming a near-circle)
            initial_guess = [25, 25, 0]

            # Optimize the ellipse fitting with regularization
            result = minimize(self.__optimizer_step, initial_guess, args = (np.zeros(2), points_2D), method='Powell')

            # Extract optimal parameters
            self.radius_a, self.radius_b, self.theta = abs(result.x)

    #IC distances
    class measure_IC_Distance:
        # IC distances typically refer to inter-commissural distances, a key set of geometric measurements 
        # used to describe the size and symmetry of the aortic valve annulus.

        def __init__(self, landmarks):
            self.landmarks = landmarks

        def compute(self):
            dist = (np.linalg.norm(np.asarray(self.landmarks["RLC"]) - np.asarray(self.landmarks["RNC"])) + 
                    np.linalg.norm(np.asarray(self.landmarks["RLC"]) - np.asarray(self.landmarks["LNC"])) + 
                    np.linalg.norm(np.asarray(self.landmarks["RNC"]) - np.asarray(self.landmarks["LNC"])))/3
            return {'IC_R': np.linalg.norm(np.asarray(self.landmarks["RLC"]) - np.asarray(self.landmarks["RNC"])),
                    'IC_L': np.linalg.norm(np.asarray(self.landmarks["RLC"]) - np.asarray(self.landmarks["LNC"])),
                    'IC_N': np.linalg.norm(np.asarray(self.landmarks["RNC"]) - np.asarray(self.landmarks["LNC"])),
                    'IC_distance': dist}

    #R, L, N flat angles. Considering that in Tomaz's document they totals 360 degrees, they must be measured in RLC, RNC and LNC plane
    class measure_FlatAngles:

        def __init__(self, landmarks):
            self.landmarks = landmarks

        def __compute_angle(self, v1, v2):
            """
            Computes the angle (in degrees) between two vectors using the dot product formula.

            Parameters:
                v1 (numpy array): First vector.
                v2 (numpy array): Second vector.

            Returns:
                float: Angle in degrees.
            """
            v1 = v1 / np.linalg.norm(v1)  # Normalize vector 1
            v2 = v2 / np.linalg.norm(v2)  # Normalize vector 2

            cos_theta = np.dot(v1, v2)  # Compute dot product
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure within valid range
            angle = np.degrees(np.arccos(cos_theta))  # Convert to degrees

            return angle

        def compute(self):
            """
            Computes the angles between the commissures (RLC, RNC, LNC) in the annular plane.

            Parameters:
                landmarks (dict): Dictionary containing 3D coordinates of commissural points.

            Returns:
                dict: Angles between commissural landmarks.
            """
            # Compute the plane normal and centroid
            points = np.array([self.landmarks["RLC"], self.landmarks["RNC"], self.landmarks["LNC"]])
            centroid = np.mean(points, axis = 0)  # Compute centroid
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal = normal / np.linalg.norm(normal)  # Normalize
    
            # Compute vectors from centroid to commissural landmarks
            vector_RLC = self.landmarks["RLC"] - centroid
            vector_RNC = self.landmarks["RNC"] - centroid
            vector_LNC = self.landmarks["LNC"] - centroid

            angles = {
                "R_flat_angle": self.__compute_angle(vector_LNC, vector_RLC),
                "L_flat_angle": self.__compute_angle(vector_RLC, vector_RNC),
                "N_flat_angle": self.__compute_angle(vector_RNC, vector_LNC),
            }

            return angles

    # Simplified, maybe need to be locked. BR_perimeter, BR_diameter, BR_max, BR_min
    class measure_BasalRing:

        def __init__(self, landmarks, ellipse_RLN):
            self.landmarks = landmarks
            self.ellipse_RLN = ellipse_RLN

        def compute(self):
            a_opt, b_opt, theta_opt = self.ellipse_RLN.radius_a, self.ellipse_RLN.radius_b, self.ellipse_RLN.theta

            # Compute basal ring metrics
            min_diameter = 2 * min(a_opt, b_opt)
            max_diameter = 2 * max(a_opt, b_opt)
            avg_diameter = (min_diameter + max_diameter) / 2

            # Compute circumference approximation (Ramanujan's approximation for ellipses)
            circumference = np.pi * (3 * (a_opt + b_opt) - np.sqrt((3 * a_opt + b_opt) * (a_opt + 3 * b_opt)))

            # Return results
            min_diameter, max_diameter, avg_diameter, circumference
            return {'BR_perimeter':circumference, 'BR_max':max_diameter, 'BR_min':min_diameter, 'BR_diameter':avg_diameter}

    # Rl, RN, LN and mean commisural heights
    class measure_meanCommisuralHeight:

        def __init__(self, landmarks, ellipse_RLN):
            self.landmarks = landmarks
            self.ellipse_RLN = ellipse_RLN

        def __distance_3D_ellipse(self, center, normal, radius1, radius2, theta, D):
            """
            Computes the shortest distance from point D to the circumference of a 3D ellipse.

            Parameters:
                center: np.array([x, y, z]) - Center of the ellipse.
                normal: np.array([x, y, z]) - Normal vector of the ellipse's plane.
                radius1: float - Semi-major axis of the ellipse.
                radius2: float - Semi-minor axis of the ellipse.
                theta: float - Rotation angle of the ellipse in its plane.
                D: np.array([x, y, z]) - External point.

            Returns:
                distance: float - Shortest Euclidean distance from D to the ellipse's circumference.
                closest_point: np.array([x, y, z]) - The closest point on the ellipse.
            """
            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Find two perpendicular vectors to define the ellipse plane
            arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            major_axis = np.cross(normal, arbitrary_vector)
            major_axis /= np.linalg.norm(major_axis)  # Normalize major axis

            minor_axis = np.cross(normal, major_axis)  # Ensure orthogonality
            minor_axis /= np.linalg.norm(minor_axis)  # Normalize minor axis

            # Apply rotation by theta
            major_rotated = np.cos(theta) * major_axis + np.sin(theta) * minor_axis
            minor_rotated = -np.sin(theta) * major_axis + np.cos(theta) * minor_axis

            # Project D onto the ellipse's plane
            CD = D - center
            distance_to_plane = np.dot(CD, normal)
            projected_D = D - distance_to_plane * normal  # Projection onto ellipse plane

            # Convert projected_D to ellipse local coordinates
            local_coords = projected_D - center
            x_proj = np.dot(local_coords, major_rotated)
            y_proj = np.dot(local_coords, minor_rotated)

            # Compute the normalized ellipse equation
            ellipse_ratio = (x_proj / radius1) ** 2 + (y_proj / radius2) ** 2

            # If the projected point is already on the ellipse's boundary, return zero distance
            if abs(ellipse_ratio - 1) < 1e-6:
                return 0

            # Find the closest point on the ellipse's boundary
            scale_factor = 1 / np.sqrt(ellipse_ratio)  # Scale to the ellipse boundary
            x_closest = x_proj * scale_factor
            y_closest = y_proj * scale_factor

            # Convert back to 3D coordinates
            closest_point = center + x_closest * major_rotated + y_closest * minor_rotated
            distance = np.linalg.norm(D - closest_point)

            return distance

        def compute(self):

            dist1 = self.__distance_3D_ellipse(self.ellipse_RLN.center_3D, self.ellipse_RLN.normal, self.ellipse_RLN.radius_a,
                                               self.ellipse_RLN.radius_b,  self.ellipse_RLN.theta,  np.asarray(self.landmarks["RLC"]))
            dist2 = self.__distance_3D_ellipse(self.ellipse_RLN.center_3D, self.ellipse_RLN.normal, self.ellipse_RLN.radius_a,
                                               self.ellipse_RLN.radius_b,  self.ellipse_RLN.theta,  np.asarray(self.landmarks["RNC"]))
            dist3 = self.__distance_3D_ellipse(self.ellipse_RLN.center_3D, self.ellipse_RLN.normal, self.ellipse_RLN.radius_a,
                                               self.ellipse_RLN.radius_b,  self.ellipse_RLN.theta,  np.asarray(self.landmarks["LNC"]))
            return {'RL_comm_height': dist1, 'RN_comm_height': dist2, 'LN_comm_height': dist3,
                    'mean_comm_heigh':(dist1 + dist2 + dist3) / 3}

    # Simplified, maybe need to be locked. ST_perimeter, ST_diameter, ST_max, ST_min
    class measure_Sinutubular:

        def __init__(self, landmarks, ellipse_Sinu):
            self.landmarks = landmarks
            self.ellipse_Sinu = ellipse_Sinu

        def compute(self):
            a_opt, b_opt, theta_opt = self.ellipse_Sinu.radius_a, self.ellipse_Sinu.radius_b, self.ellipse_Sinu.theta

            # Compute basal ring metrics
            min_diameter = 2 * min(a_opt, b_opt)
            max_diameter = 2 * max(a_opt, b_opt)
            avg_diameter = (min_diameter + max_diameter) / 2

            # Compute circumference approximation (Ramanujan's approximation for ellipses)
            circumference = np.pi * (3 * (a_opt + b_opt) - np.sqrt((3 * a_opt + b_opt) * (a_opt + 3 * b_opt)))

            # Return results
            min_diameter, max_diameter, avg_diameter, circumference
            return {'ST_perimeter':circumference, 'ST_max':max_diameter, 'ST_min':min_diameter, 'ST_diameter':avg_diameter}

    # angles between commissural landmarks and the basal ring plane normal
    class measure_BR_VerticalAngles:

        def __init__(self, landmarks, ellipse_RLN):
            self.landmarks = landmarks
            self.ellipse_RLN = ellipse_RLN

        def __angle(self, v1, v2):
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)

            
            # The SVD based plane fitting does not provide a consistent normal
            # direction. Depending on the orientation, the dot product can be
            # positive or negative which leads to angles in the range
            # [0, 180].  The vertical angle is unsigned, therefore we take the
            # absolute value of the cosine to always obtain the acute angle.
            cos_theta = abs(cos_theta)
            return np.degrees(np.arccos(cos_theta))

        def __closest_point(self, center, normal, radius1, radius2, theta, D):
            normal = normal / np.linalg.norm(normal)
            arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            major_axis = np.cross(normal, arbitrary_vector)
            major_axis /= np.linalg.norm(major_axis)
            minor_axis = np.cross(normal, major_axis)
            minor_axis /= np.linalg.norm(minor_axis)

            major_rot = np.cos(theta) * major_axis + np.sin(theta) * minor_axis
            minor_rot = -np.sin(theta) * major_axis + np.cos(theta) * minor_axis

            CD = D - center
            distance_to_plane = np.dot(CD, normal)
            projected_D = D - distance_to_plane * normal

            local = projected_D - center
            x_proj = np.dot(local, major_rot)
            y_proj = np.dot(local, minor_rot)

            ellipse_ratio = (x_proj / radius1) ** 2 + (y_proj / radius2) ** 2
            if abs(ellipse_ratio - 1) < 1e-6:
                closest_point = projected_D
            else:
                scale_factor = 1 / np.sqrt(ellipse_ratio)
                x_closest = x_proj * scale_factor
                y_closest = y_proj * scale_factor
                closest_point = center + x_closest * major_rot + y_closest * minor_rot
            return closest_point

        def compute(self):

            normal = self.ellipse_RLN.normal
            center = self.ellipse_RLN.center_3D

            cp_R = self.__closest_point(center, normal, self.ellipse_RLN.radius_a, self.ellipse_RLN.radius_b, self.ellipse_RLN.theta, np.asarray(self.landmarks['RLC']))
            cp_N = self.__closest_point(center, normal, self.ellipse_RLN.radius_a, self.ellipse_RLN.radius_b, self.ellipse_RLN.theta, np.asarray(self.landmarks['RNC']))
            cp_L = self.__closest_point(center, normal, self.ellipse_RLN.radius_a, self.ellipse_RLN.radius_b, self.ellipse_RLN.theta, np.asarray(self.landmarks['LNC']))

            vec_RLC = np.asarray(self.landmarks['RLC']) - cp_R
            vec_RNC = np.asarray(self.landmarks['RNC']) - cp_N
            vec_LNC = np.asarray(self.landmarks['LNC']) - cp_L

            ang_R = self.__angle(vec_RLC, normal)
            ang_N = self.__angle(vec_RNC, normal)
            ang_L = self.__angle(vec_LNC, normal)

            return {
                'R_vertical_angle': ang_R,
                'N_vertical_angle': ang_N,
                'L_vertical_angle': ang_L,
                'mean_vertical_angle': (ang_R + ang_N + ang_L) / 3
            }

    # commissural diameter
    class measure_CommissuralDiameter:
        #commissural diameter is the largest distance between two commissural points, which represents the widest opening of the aortic valve at the commissural level.

        def __init__(self, landmarks):
            self.landmarks = landmarks

        def compute(self):
            """
            Computes the commissural angles between RLC_land, RNC_land, and LNC_land.
            """
            dist1 = np.linalg.norm(np.asarray(self.landmarks['RNC']) - np.asarray(self.landmarks['RLC']))
            dist2 = np.linalg.norm(np.asarray(self.landmarks['LNC']) - np.asarray(self.landmarks['RNC']))
            dist3 = np.linalg.norm(np.asarray(self.landmarks['RLC']) - np.asarray(self.landmarks['LNC']))

            return {'commissural_diameter': max(dist1, max(dist2, dist3))}

    # Mean commissural angle
    # does not work yet
    class measure_meanCommissuralAngle:

        def __init__(self, landmarks):
            self.landmarks = landmarks

        # Define the regularized ellipse fitting function
        def __angle_between_vectors(self, v1, v2):
            """
            Computes the angle (in degrees) between two vectors.
    
            Parameters:
                v1, v2: np.array([x, y, z]) - Two vectors
    
            Returns:
                angle_deg: float - Angle between the vectors in degrees.
            """
            # Normalize the vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm == 0 or v2_norm == 0:
                raise ValueError("One of the vectors has zero length, cannot compute angle.")

            v1_unit = v1 / v1_norm
            v2_unit = v2 / v2_norm

            # Compute dot product and clamp to avoid floating point errors
            dot_product = np.dot(v1_unit, v2_unit)
            dot_product = np.clip(dot_product, -1.0, 1.0)

            # Compute the angle in radians and convert to degrees
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)

            return angle_deg

        def compute(self):
            """
            Computes the commissural angles between RLC_land, RNC_land, and LNC_land.
            """
            vector_RL = np.asarray(self.landmarks['RNC']) - np.asarray(self.landmarks['RLC'])
            vector_RN = np.asarray(self.landmarks['LNC']) - np.asarray(self.landmarks['RNC'])
            vector_LN = np.asarray(self.landmarks['RLC']) - np.asarray(self.landmarks['LNC'])
    
            angles = {
                "RL_angle": self.__angle_between_vectors(vector_RL, -vector_LN),
                "RN_angle": self.__angle_between_vectors(vector_RN, -vector_RL),
                "LN_angle": self.__angle_between_vectors(vector_LN, -vector_RN),
            }
    
            return {'RL_angle':angles["RL_angle"], 'RN_angle':angles["RN_angle"], 'LN_angle':angles["LN_angle"], 
                    'mean_commissural_angle': np.mean(list(angles.values()))}

    # the angle between basal ring and commissural angle
    class measure_BR_C_plane_angle:

        def __init__(self, landmarks):
            self.landmarks = landmarks

        def __angle_between_planes(self, n1, n2):
            """
            Compute the angle (in degrees) between two plane normals.
            """
            cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # for numerical stability
            angle_rad = np.arccos(cos_theta)
            angle_deg = np.degrees(angle_rad)
            return min(angle_deg, np.abs(180 - angle_deg))

        def compute(self):
            points_C = []
            points_C.append(self.landmarks['RLC'])
            points_C.append(self.landmarks['RNC'])
            points_C.append(self.landmarks['LNC'])
            points_BR = []
            points_BR.append(self.landmarks['R'])
            points_BR.append(self.landmarks['L'])
            points_BR.append(self.landmarks['N'])

            plane_C  = landmarking_computeMeasurements_simplified.landmark_fit_plane(points_C)
            plane_BR = landmarking_computeMeasurements_simplified.landmark_fit_plane(points_BR)
            _, normal_comm  = plane_C.fit()
            _, normal_nadir = plane_BR.fit()

            return {'BR_C_plane_angle': self.__angle_between_planes(-normal_comm, normal_nadir)}

    # centroid valve height
    class measure_centroid_valve_height:

        def __init__(self, landmarks):
            self.landmarks = landmarks

        def __centroid(self, points):
            return np.mean(points, axis = 0)

        def compute(self):
            points_C = []
            points_C.append(self.landmarks['RLC'])
            points_C.append(self.landmarks['RNC'])
            points_C.append(self.landmarks['LNC'])
            points_BR = []
            points_BR.append(self.landmarks['R'])
            points_BR.append(self.landmarks['L'])
            points_BR.append(self.landmarks['N'])

            centroid_C = self.__centroid(points_C)
            centroid_BR = self.__centroid(points_BR)
            return {'centroid_valve_height': np.linalg.norm(centroid_BR - centroid_C)}

    def __init__(self, landmarks):
        self.landmarks = landmarks
        points_3D = [self.landmarks['R'], self.landmarks['L'], self.landmarks['N']]
        self.ellipse_RLN = landmarking_computeMeasurements_simplified.landmark_ellipse(points_3D)
        self.ellipse_RLN.optimize()
        points_3D = [self.landmarks['RLC'], self.landmarks['RNC'], self.landmarks['LNC']]
        self.ellipse_Sinu = landmarking_computeMeasurements_simplified.landmark_ellipse(points_3D)
        self.ellipse_Sinu.optimize()

    def compute_metrics(self):
        measure_ICD   = landmarking_computeMeasurements_simplified.measure_IC_Distance(self.landmarks)
        measure_FA    = landmarking_computeMeasurements_simplified.measure_FlatAngles(self.landmarks)
        measure_BR    = landmarking_computeMeasurements_simplified.measure_BasalRing(self.landmarks)
        measure_mCH = landmarking_computeMeasurements_simplified.measure_meanCommisuralHeight(self.landmarks)
        measure_ST = landmarking_computeMeasurements_simplified.measure_Sinutubular(self.landmarks)
        measure_BR_VA = landmarking_computeMeasurements_simplified.measure_BR_VerticalAngles(self.landmarks)
        measure_CD = landmarking_computeMeasurements_simplified.measure_CommissuralDiameter(self.landmarks)
        measure_BR_C_PA = landmarking_computeMeasurements_simplified.measure_BR_C_plane_angle(self.landmarks)
        measure_CVH = landmarking_computeMeasurements_simplified.measure_centroid_valve_height(self.landmarks)
        
        res1 = measure_ICD.compute()
        res2 = measure_FA.compute()
        res3 = measure_BR.compute()
        res4 = measure_mCH.compute()
        res5 = measure_ST.compute()
        res6 = measure_BR_VA.compute()
        res7 = measure_CD.compute()
        res8 = measure_BR_C_PA.compute()
        res9 = measure_CVH.compute()
        return res1 | res2 | res3 | res4 | res5 | res6 | res7 | res8 | res9

    def compute_individual_metric(self, metric_name):
        if metric_name == 'IC_R':
            return landmarking_computeMeasurements_simplified.measure_IC_Distance(self.landmarks).compute()['IC_R']
        elif metric_name == 'IC_L':
            return landmarking_computeMeasurements_simplified.measure_IC_Distance(self.landmarks).compute()['IC_L']
        elif metric_name == 'IC_N':
            return landmarking_computeMeasurements_simplified.measure_IC_Distance(self.landmarks).compute()['IC_N']
        elif metric_name == 'IC_distance':
            return landmarking_computeMeasurements_simplified.measure_IC_Distance(self.landmarks).compute()['IC_distance']
        elif metric_name == 'R_flat_angle':
            return landmarking_computeMeasurements_simplified.measure_FlatAngles(self.landmarks).compute()['R_flat_angle']
        elif metric_name == 'L_flat_angle':
            return landmarking_computeMeasurements_simplified.measure_FlatAngles(self.landmarks).compute()['L_flat_angle']
        elif metric_name == 'N_flat_angle':
            return landmarking_computeMeasurements_simplified.measure_FlatAngles(self.landmarks).compute()['N_flat_angle']
        elif metric_name == 'BR_perimeter':
            return landmarking_computeMeasurements_simplified.measure_BasalRing(self.landmarks, self.ellipse_RLN).compute()['BR_perimeter']
        elif metric_name == 'BR_max':
            return landmarking_computeMeasurements_simplified.measure_BasalRing(self.landmarks, self.ellipse_RLN).compute()['BR_max']
        elif metric_name == 'BR_min':
            return landmarking_computeMeasurements_simplified.measure_BasalRing(self.landmarks, self.ellipse_RLN).compute()['BR_min']
        elif metric_name == 'BR_diameter':
            return landmarking_computeMeasurements_simplified.measure_BasalRing(self.landmarks, self.ellipse_RLN).compute()['BR_diameter']
        elif metric_name == 'RL_comm_height':
            return landmarking_computeMeasurements_simplified.measure_meanCommisuralHeight(self.landmarks, self.ellipse_RLN).compute()['RL_comm_height']
        elif metric_name == 'RN_comm_height':
            return landmarking_computeMeasurements_simplified.measure_meanCommisuralHeight(self.landmarks, self.ellipse_RLN).compute()['RN_comm_height']
        elif metric_name == 'LN_comm_height':
            return landmarking_computeMeasurements_simplified.measure_meanCommisuralHeight(self.landmarks, self.ellipse_RLN).compute()['LN_comm_height']
        elif metric_name == 'mean_comm_heigh':
            return landmarking_computeMeasurements_simplified.measure_meanCommisuralHeight(self.landmarks, self.ellipse_RLN).compute()['mean_comm_heigh']
        elif metric_name == 'ST_perimeter':
            return landmarking_computeMeasurements_simplified.measure_Sinutubular(self.landmarks, self.ellipse_Sinu).compute()['ST_perimeter']
        elif metric_name == 'ST_max':
            return landmarking_computeMeasurements_simplified.measure_Sinutubular(self.landmarks, self.ellipse_Sinu).compute()['ST_max']
        elif metric_name == 'ST_min':
            return landmarking_computeMeasurements_simplified.measure_Sinutubular(self.landmarks, self.ellipse_Sinu).compute()['ST_min']
        elif metric_name == 'ST_diameter':
            return landmarking_computeMeasurements_simplified.measure_Sinutubular(self.landmarks, self.ellipse_Sinu).compute()['ST_diameter']
        elif metric_name == 'R_vertical_angle':
            return landmarking_computeMeasurements_simplified.measure_BR_VerticalAngles(self.landmarks, self.ellipse_RLN).compute()['R_vertical_angle']
        elif metric_name == 'N_vertical_angle':
            return landmarking_computeMeasurements_simplified.measure_BR_VerticalAngles(self.landmarks, self.ellipse_RLN).compute()['N_vertical_angle']
        elif metric_name == 'L_vertical_angle':
            return landmarking_computeMeasurements_simplified.measure_BR_VerticalAngles(self.landmarks, self.ellipse_RLN).compute()['L_vertical_angle']
        elif metric_name == 'mean_vertical_angle':
            return landmarking_computeMeasurements_simplified.measure_BR_VerticalAngles(self.landmarks, self.ellipse_RLN).compute()['mean_vertical_angle']
        elif metric_name == 'commissural_diameter':
            return landmarking_computeMeasurements_simplified.measure_CommissuralDiameter(self.landmarks).compute()['commissural_diameter']
        elif metric_name == 'RL_angle':
            return landmarking_computeMeasurements_simplified.measure_meanCommissuralAngle(self.landmarks).compute()['RL_angle']
        elif metric_name == 'RN_angle':
            return landmarking_computeMeasurements_simplified.measure_meanCommissuralAngle(self.landmarks).compute()['RN_angle']
        elif metric_name == 'LN_angle':
            return landmarking_computeMeasurements_simplified.measure_meanCommissuralAngle(self.landmarks).compute()['LN_angle']
        elif metric_name == 'mean_commissural_angle':
            return landmarking_computeMeasurements_simplified.measure_meanCommissuralAngle(self.landmarks).compute()['mean_commissural_angle']
        elif metric_name == 'BR_C_plane_angle':
            return landmarking_computeMeasurements_simplified.measure_BR_C_plane_angle(self.landmarks).compute()['BR_C_plane_angle']
        elif metric_name == 'centroid_valve_height':
            return landmarking_computeMeasurements_simplified.measure_centroid_valve_height(self.landmarks).compute()['centroid_valve_height']
        return None

    def get_all_metrics(self):
        metrics = {}
        metrics['IC_R']                   = self.compute_individual_metric('IC_R')
        metrics['IC_L']                   = self.compute_individual_metric('IC_L')
        metrics['IC_N']                   = self.compute_individual_metric('IC_N')
        metrics['IC_distance']            = self.compute_individual_metric('IC_distance')
        metrics['R_flat_angle']           = self.compute_individual_metric('R_flat_angle')
        metrics['L_flat_angle']           = self.compute_individual_metric('L_flat_angle')
        metrics['N_flat_angle']           = self.compute_individual_metric('N_flat_angle')
        metrics['BR_perimeter']           = self.compute_individual_metric('BR_perimeter')
        metrics['BR_max']                 = self.compute_individual_metric('BR_max')
        metrics['BR_min']                 = self.compute_individual_metric('BR_min')
        metrics['BR_diameter']            = self.compute_individual_metric('BR_diameter')
        metrics['RL_comm_height']         = self.compute_individual_metric('RL_comm_height')
        metrics['RN_comm_height']         = self.compute_individual_metric('RN_comm_height')
        metrics['LN_comm_height']         = self.compute_individual_metric('LN_comm_height')
        metrics['mean_comm_heigh']        = self.compute_individual_metric('mean_comm_heigh')
        metrics['ST_perimeter']           = self.compute_individual_metric('ST_perimeter')
        metrics['ST_max']                 = self.compute_individual_metric('ST_max')
        metrics['ST_min']                 = self.compute_individual_metric('ST_min')
        metrics['ST_diameter']            = self.compute_individual_metric('ST_diameter')
        metrics['R_vertical_angle']       = self.compute_individual_metric('R_vertical_angle')
        metrics['N_vertical_angle']       = self.compute_individual_metric('N_vertical_angle')
        metrics['L_vertical_angle']       = self.compute_individual_metric('L_vertical_angle')
        metrics['mean_vertical_angle']    = self.compute_individual_metric('mean_vertical_angle')
        metrics['commissural_diameter']   = self.compute_individual_metric('commissural_diameter')
        metrics['RL_angle']               = self.compute_individual_metric('RL_angle')
        metrics['RN_angle']               = self.compute_individual_metric('RN_angle')
        metrics['LN_angle']               = self.compute_individual_metric('LN_angle')
        metrics['mean_commissural_angle'] = self.compute_individual_metric('mean_commissural_angle')
        metrics['BR_C_plane_angle']       = self.compute_individual_metric('BR_C_plane_angle')
        metrics['centroid_valve_height']  = self.compute_individual_metric('centroid_valve_height')
        return metrics

    @staticmethod
    def get_measurement_names():
        return {'IC_R':['RLC', 'RNC', 'LNC'], 'IC_L':['RLC', 'RNC', 'LNC'], 'IC_N':['RLC', 'RNC', 'LNC'], 'IC_distance':['RLC', 'RNC', 'LNC'],
                'R_flat_angle':['RLC', 'RNC', 'LNC'], 'L_flat_angle':['RLC', 'RNC', 'LNC'], 'N_flat_angle':['RLC', 'RNC', 'LNC'],
                'BR_perimeter':['R', 'L', 'N'], 'BR_max':['R', 'L', 'N'], 'BR_min':['R', 'L', 'N'], 'BR_diameter':['R', 'L', 'N'],
                'RL_comm_height':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'RN_comm_height':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'LN_comm_height':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'mean_comm_heigh':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'],
                'ST_perimeter':['RLC', 'RNC', 'LNC'], 'ST_max':['RLC', 'RNC', 'LNC'], 'ST_min':['RLC', 'RNC', 'LNC'], 'ST_diameter':['RLC', 'RNC', 'LNC'],
                'R_vertical_angle':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'N_vertical_angle':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'L_vertical_angle':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'mean_vertical_angle':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'],
                'commissural_diameter':['RLC', 'RNC', 'LNC'], 'RL_angle':['RLC', 'RNC', 'LNC'], 'RN_angle':['RLC', 'RNC', 'LNC'], 'LN_angle':['RLC', 'RNC', 'LNC'], 'mean_commissural_angle':['RLC', 'RNC', 'LNC'],
                'BR_C_plane_angle':['R', 'L', 'N', 'RLC', 'RNC', 'LNC'], 'centroid_valve_height':['R', 'L', 'N', 'RLC', 'RNC', 'LNC']}