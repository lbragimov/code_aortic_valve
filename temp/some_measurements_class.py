import numpy as np


class landmarking_computeMeasurements_simplified:

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

    # BR_perimeter, BR_diameter, BR_max, BR_min
    class measure_BasalRing:

        def __init__(self, landmarks):
            self.landmarks = landmarks

        def compute(self):
            points_3D = []
            points_3D.append(self.landmarks["R"])
            points_3D.append(self.landmarks["L"])
            points_3D.append(self.landmarks["N"])
            ellipse = landmarking_computeMeasurements_simplified.landmark_ellipse(points_3D)
            ellipse.optimize()
            a_opt, b_opt, theta_opt = ellipse.radius_a, ellipse.radius_b, ellipse.theta

            # Compute basal ring metrics
            min_diameter = 2 * min(a_opt, b_opt)
            max_diameter = 2 * max(a_opt, b_opt)
            avg_diameter = (min_diameter + max_diameter) / 2

            # Compute circumference approximation (Ramanujan's approximation for ellipses)
            circumference = np.pi * (3 * (a_opt + b_opt) - np.sqrt((3 * a_opt + b_opt) * (a_opt + 3 * b_opt)))

            # Return results
            min_diameter, max_diameter, avg_diameter, circumference
            return {'BR_perimeter':circumference, 'BR_max':max_diameter, 'BR_min':min_diameter, 'BR_diameter':avg_diameter}

    # BR vertical angles. Possibly wrong. 
    # Idea is to compute angle from RLN(etc) landmarks to basal ring. I assume that it should be measured to ring itself, 
    # not basal plane (angles are always zero, nor basal ellipse.
    class measure_BR_VerticalAngles:

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

        def __closest_point_on_3D_ellipse(self, center, normal, radius1, radius2, theta, D):
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
                return projected_D

            # Find the closest point on the ellipse's boundary
            scale_factor = 1 / np.sqrt(ellipse_ratio)  # Scale to the ellipse boundary
            x_closest = x_proj * scale_factor
            y_closest = y_proj * scale_factor

            # Convert back to 3D coordinates
            closest_point = center + x_closest * major_rotated + y_closest * minor_rotated

            return closest_point

        def __compute_angle(self, ellipse, D):
            # Compute two vectors on the plane
            closest_point = self.__closest_point_on_3D_ellipse(ellipse.center_3D, ellipse.normal, ellipse.radius_a,
                                                               ellipse.radius_b, ellipse.theta, D)  # Projection onto the plane

            vector1 = D - closest_point
            return self.__angle_between_vectors(vector1, ellipse.normal)

        def compute(self):
            points_3D = []
            points_3D.append(self.landmarks["R"])
            points_3D.append(self.landmarks["L"])
            points_3D.append(self.landmarks["N"])
            ellipse = landmarking_computeMeasurements_simplified.landmark_ellipse(points_3D)
            ellipse.optimize()

            RLc_vertical_angle = self.__compute_angle(ellipse, np.asarray(self.landmarks['RLC']))
            RNc_vertical_angle = self.__compute_angle(ellipse, np.asarray(self.landmarks['RNC']))
            LNc_vertical_angle = self.__compute_angle(ellipse, np.asarray(self.landmarks['LNC']))
            return {'RLc_vertical_angle':min(RLc_vertical_angle, 180 - RLc_vertical_angle), 
                    'RNc_vertical_angle':min(RNc_vertical_angle, 180 - RNc_vertical_angle), 
                    'LNc_vertical_angle':min(LNc_vertical_angle, 180 - LNc_vertical_angle)}

    # Mean commisural height
    class measure_meanCommisuralHeight:

        def __init__(self, landmarks):
            self.landmarks = landmarks

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
            points_3D = []
            points_3D.append(self.landmarks["R"])
            points_3D.append(self.landmarks["L"])
            points_3D.append(self.landmarks["N"])
            ellipse = landmarking_computeMeasurements_simplified.landmark_ellipse(points_3D)
            ellipse.optimize()

            dist1 = self.__distance_3D_ellipse(ellipse.center_3D, ellipse.normal, ellipse.radius_a,
                                               ellipse.radius_b, ellipse.theta, np.asarray(self.landmarks["RLC"]))
            dist2 = self.__distance_3D_ellipse(ellipse.center_3D, ellipse.normal, ellipse.radius_a,
                                               ellipse.radius_b, ellipse.theta, np.asarray(self.landmarks["RNC"]))
            dist3 = self.__distance_3D_ellipse(ellipse.center_3D, ellipse.normal, ellipse.radius_a,
                                               ellipse.radius_b, ellipse.theta, np.asarray(self.landmarks["LNC"]))
            return {'mean_comm_heigh':(dist1 + dist2 + dist3) / 3}

    # ST_perimeter, ST_diameter, ST_max, ST_min
    class measure_Sinutubular:

        def __init__(self, landmarks):
            self.landmarks = landmarks

        def compute(self):
            points_3D = []
            points_3D.append(self.landmarks['RLC'])
            points_3D.append(self.landmarks['RNC'])
            points_3D.append(self.landmarks['LNC'])
            ellipse = landmarking_computeMeasurements_simplified.landmark_ellipse(points_3D)
            ellipse.optimize()
            a_opt, b_opt, theta_opt = ellipse.radius_a, ellipse.radius_b, ellipse.theta

            # Compute basal ring metrics
            min_diameter = 2 * min(a_opt, b_opt)
            max_diameter = 2 * max(a_opt, b_opt)
            avg_diameter = (min_diameter + max_diameter) / 2

            # Compute circumference approximation (Ramanujan's approximation for ellipses)
            circumference = np.pi * (3 * (a_opt + b_opt) - np.sqrt((3 * a_opt + b_opt) * (a_opt + 3 * b_opt)))

            # Return results
            min_diameter, max_diameter, avg_diameter, circumference
            return {'ST_perimeter':circumference, 'ST_max':max_diameter, 'ST_min':min_diameter, 'ST_diameter':avg_diameter}

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
    
            return {'Mean_commissural_angle': np.mean(list(angles.values()))}

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

    def __init__(self, landmarks):
        self.landmarks = landmarks

    def compute_metrics(self):
        measure_ICD   = landmarking_computeMeasurements_simplified.measure_IC_Distance(self.landmarks)
        measure_FA    = landmarking_computeMeasurements_simplified.measure_FlatAngles(self.landmarks)
        measure_BR    = landmarking_computeMeasurements_simplified.measure_BasalRing(self.landmarks)
        measure_BR_VA = landmarking_computeMeasurements_simplified.measure_BR_VerticalAngles(self.landmarks)
        measure_mCH = landmarking_computeMeasurements_simplified.measure_meanCommisuralHeight(self.landmarks)
        measure_ST = landmarking_computeMeasurements_simplified.measure_Sinutubular(self.landmarks)
        #measure_mCA = landmarking_computeMeasurements_simplified.measure_meanCommissuralAngle(self.landmarks)
        measure_CD = landmarking_computeMeasurements_simplified.measure_CommissuralDiameter(self.landmarks)

        res1 = measure_ICD.compute()
        res2 = measure_FA.compute()
        res3 = measure_BR.compute()
        res4 = measure_BR_VA.compute()
        res5 = measure_mCH.compute()
        res6 = measure_ST.compute()
        #res7 = measure_mCA.compute()
        res8 = measure_CD.compute()
        return res1 | res2 | res3 | res4 | res5 | res6 | res8
