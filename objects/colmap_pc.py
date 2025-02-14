import math
import numpy as np
import open3d as o3d

from itertools import combinations
from sklearn.cluster import DBSCAN

from objects.point import Point
from objects.point_cloud import PointCloud

class ColmapPointCloud(PointCloud): 
    def __init__(self, points_txt_path, calibration_colors): 
        super().__init__(points_txt_path)
        self.calibration_colors = calibration_colors
        self.calibration_centroids = None 

        # default sklearn DBSCAN params, can be adjusted
        self.dbscan_eps = 0.5
        self.dbscan_min_points = 5

    def get_calibration_colors(self): 
        return self.calibration_colors 
    
    def get_calibration_centroids(self): 
        return self.calibration_centroids

    def read_points_from_txt(self):
        ''' 
        Read point id, point coords and colors from COLMAP points3D.txt file.
        '''
        f = open(self.get_points_txt_path(), 'r')
        lines = f.readlines()[3:] # data starts from 4th line in COLMAP points3D.txt 

        points_dict = dict() 
        for line in lines: 
            point_id = int(line.split()[0])
            xyz = np.array([float(i) for i in line.split()[1:4]])
            colors = np.array([float(i) for i in line.split()[4:7]]) / 256

            point = Point(point_id, xyz, colors)
            points_dict[point_id] = point
        
        return points_dict 
    
    def detect_calibration_points(self): 
        ''' 
        Detect all points with colors closed to the calibration colors. 
        Return the points as clusters defined by colors.
        '''
        calibration_points = {key: [] for key in self.get_calibration_colors()}
        
        for point in self.get_points_dict().values():
            for calibration_color, color_info in self.get_calibration_colors().items(): 
                if point.is_color_closed(color_info): 
                    calibration_points[calibration_color].append(point)

        self.calibration_points = calibration_points 

        return calibration_points
    
    def get_candidate_cluster_centroids(self, calibration_dict): 
        ''' 
        Use DBSCAN to remove noises and find candidate calibration cluster centroids. 

        To find cluster centroid, we use DBSCAN instead of just calculating centroid within each CalibrationColour 
        group inside calibration_points because 
            - the markers can have similar colours.
            - the object can have similar colour with one of the markers.
        '''
        calibration_points = [point for points in calibration_dict.values() for point in points]
        calibration_xyz = np.array([point.get_xyz() for point in calibration_points])

        # Step 1: Apply DBSCAN
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_points) 
        labels = dbscan.fit_predict(calibration_xyz)

        # Step 2: Mask out noise points (where label is -1)
        valid_labels = labels != -1
        valid_points = calibration_xyz[valid_labels]
        valid_labels = labels[valid_labels]

        # Step 3: Calculate centroids using numpy
        unique_labels = np.unique(valid_labels)
        candidate_centroids = np.array([valid_points[valid_labels == label].mean(axis=0) for label in unique_labels])

        return candidate_centroids

    def evaluate_centroid_combination(self, candidate_centroids): 
        ''' 
        Given 4 points, evaluate if they can be corners of a rectangle.

        The metric consider coplanarity and orthogonality of the 4 points.
        '''
        pass

    def get_calibration_centroids(self): 
        ''' 
        Obtain the centroids of the 4 calibration clusters.
        ''' 
        calibration_points = self.detect_calibration_points()
        candidate_centroids = self.get_candidate_cluster_centroids(calibration_points)

        #TODO: implement evaluate_centroid_combination
        # centroids = self.evaluate_centroid_combination(candidate_centroids)

        self.calibration_centroids = candidate_centroids
        # self.calibration_centroids = centroids
        # print(self.get_calibration_centroids()) #TODO: why calling print(self.get_calibration_centroids()) takes super long???

        return candidate_centroids

    def get_center(self): 
        ''' 
        Find the center of the point cloud by calculating the center of calibration centroids.
        ''' 
        if self.calibration_colors is None: 
            raise Exception("calibration centroids does not exist")
        
        center = self.get_calibration_centroids().mean(axis=0)

        return center

    def get_z_axis_aligned_rotation(self): 
        ''' 
        Find the rotation matrix to align the point cloud to be z-axis perpendicular. 

        This is done by aligning the 4 calibration centroids to be z-axis perpendicular.
        '''
        def fit_plane_to_4_points(points): 
            A = np.zeros((len(points), 3)) 
            b = np.zeros((len(points), 1)) 

            for i, point in enumerate(points): 
                x, y, z = point
                A[i, 0] = x 
                A[i, 1] = y 
                A[i, 2] = 1 
                b[i] = z 

            fit = np.linalg.inv(A.T @ A) @ A.T @ b
            return fit
        
        def vector_angle(u, v):
            return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))
        
        if self.get_calibration_centroids() is None: 
            raise Exception("calibration centroids does not exist")
        
        # find the normal of the plane that fit the 4 calibration centroids
        plane_normal = fit_plane_to_4_points(self.get_calibration_centroids())
        a, b, c = plane_normal

        # calculate rotation angle between plane normal & z-axis
        z_axis = (0,0,1)
        rotation_angle = vector_angle(tuple(plane_normal.T), z_axis)

        # calculate rotation axis
        plane_normal_length = math.sqrt(a**2 + b**2 + c**2)
        u1 = b / plane_normal_length
        u2 = -a / plane_normal_length
        rotation_axis = (u1, u2, 0)

        # generate axis-angle representation
        optimization_factor = 1.4
        axis_angle = tuple([x * rotation_angle * optimization_factor for x in rotation_axis])

        return axis_angle

    def get_calibrated_scale(self): 
        ''' 
        Obtain calibrated scale from the 4 calibration centroids.

        Calculation: scale = (1 / diagonal length of the calibration rectangle)
        '''
        max_dist = 0 

        for point1, point2 in combinations(self.calibration_centroids, 2): 
            dist = np.linalg.norm(point1 - point2)
            if dist > max_dist: 
                max_dist = dist 

        scale = 1 / max_dist 

        return scale

    def visualize_calibration_points(self, with_non_calibration=True, with_centroids=True): 
        ''' 
        Visualize the point cloud of calibration points (marked in red) and centroids (marked in green) in Open3d. 

        This function should be used to adjust the calibration's RGB colors and threshold for CALIBRATION_COLOURS.
        '''
        red = np.array([255,0,0]) / 256
        green = np.array([0,255,0]) / 256

        pc = self.get_o3d_pc()

        calibration_points = self.detect_calibration_points()
        # convert the calibration_points format to use the create_o3d_pc function 
        dct = {} 
        for color, points in calibration_points.items(): 
            dct[color] = dict()
            for point in points: 
                dct[color][point.get_point_id()] = point

        calibration_pcs = [] 
        for calibration_color, points_dct in dct.items(): 
            calibration_pc = super().create_o3d_pc(points_dct)
            if not with_non_calibration:
                color = self.get_calibration_colors()[calibration_color][0]
            else: 
                color = red
            calibration_pc.paint_uniform_color(color)
            calibration_pcs.append(calibration_pc)

        if with_centroids: 
            centroids_xyz = self.get_calibration_centroids()
            centroids_pc = o3d.geometry.PointCloud() 
            centroids_pc.points = o3d.utility.Vector3dVector(centroids_xyz)
            centroids_pc.paint_uniform_color(green)

        if with_non_calibration and with_centroids:
            o3d.visualization.draw_geometries([pc, centroids_pc] + calibration_pcs) 
        elif with_non_calibration: 
            o3d.visualization.draw_geometries([pc,] + calibration_pcs)
        elif with_centroids:
            o3d.visualization.draw_geometries(calibration_pcs + [centroids_pc])
        else: 
            o3d.visualization.draw_geometries(calibration_pcs)
        