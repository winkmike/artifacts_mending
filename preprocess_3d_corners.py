import open3d as o3d 
import numpy as np
import math
import copy

from enum import Enum
from sklearn.cluster import DBSCAN
from itertools import combinations

#TODO: add typings
#TODO: separate out the RGB values to an Enum class
#TODO: change all self.{attribute} to getter
#TODO: inside CornersPointCloud, link point_dict with its o3d_pc at each preprocessing step
 
# PATH
piece1_colmap_path = "./data/piece-1-txt/new-txt/points3D.txt"
piece2_colmap_path = "./data/piece-2-txt/txt/points3D.txt"

piece1_3d_corners_path = "./output/piece-1-corner-3d.txt"
piece2_3d_corners_path = "./output/piece-2-corner-3d.txt"

piece1_calibrated_3d_corners_path = "./output/piece-1-calibrated-corner-3d.ply"
piece2_calibrated_3d_corners_path = "./output/piece-2-calibrated-corner-3d.ply"

piece1_preprocessed_3d_corners_path = "./output/piece-1-preprocessed-corner-3d.ply"
piece2_preprocessed_3d_corners_path = "./output/piece-2-preprocessed-corner-3d.ply"

# calibration COLORS 
class CalibrationColour(Enum):
    BLUE = 1
    PINK = 2
    YELLOW = 3
    ORANGE = 4

class Point: 
    def __init__(self, point_id, xyz, color): 
        self.point_id = point_id
        self.xyz = xyz 
        self.color = color

    def get_point_id(self): 
        return self.point_id 

    def get_xyz(self): 
        return self.xyz

    def get_color(self): 
        return self.color
    
    def is_color_closed(self, color_info): 
        ''' 
        Return True if the point's color is closed enough to the calibration_color.
        '''
        calibration_rgb, threshold = color_info
        color = self.get_color() 
        color_dist = np.linalg.norm(color - calibration_rgb)

        return color_dist <= threshold
    

class PointCloud: 
    def __init__(self, points_txt_path): 
        self.points_txt_path = points_txt_path
        self.points_dict = self.read_points_from_txt()
        self.o3d_pc = self.create_o3d_pc(self.get_points_dict()) 

    def get_points_txt_path(self): 
        return self.points_txt_path 
    
    def get_points_dict(self): 
        return self.points_dict 
    
    def get_o3d_pc(self): 
        return self.o3d_pc

    def read_points_from_txt(self): 
        pass 

    def get_xyz_arr(self, points_dict): 
        ''' 
        Get the position array of all points in the PointCloud. 

        The array follows the order of sorted point_ids to make sure the corresponding rows of xyz array and color array are from the same Point. 
        '''
        point_ids = sorted(points_dict.keys())
        xyz_arr = np.zeros((len(point_ids), 3))

        for i in range(len(point_ids)): 
            point_id = point_ids[i]
            xyz_arr[i] = points_dict[point_id].get_xyz()

        return xyz_arr 

    def get_color_arr(self, points_dict): 
        ''' 
        Get the color array of all points in the PointCloud. 

        The array follows the order of sorted point_ids to make sure the corresponding rows of xyz array and color array are from the same Point. 
        '''
        point_ids = sorted(points_dict.keys())
        color_arr = np.zeros((len(point_ids), 3))

        for i in range(len(point_ids)): 
            point_id = point_ids[i]
            color_arr[i] = points_dict[point_id].get_color()

        return color_arr 

    def create_o3d_pc(self, points_dict): 
        ''' 
        Create an Open3d PointCloud object from xyz array and color array.
        '''
        xyz_arr = self.get_xyz_arr(points_dict) 
        color_arr = self.get_color_arr(points_dict) 

        pcd = o3d.geometry.PointCloud()    
        pcd.points = o3d.utility.Vector3dVector(xyz_arr)
        pcd.colors = o3d.utility.Vector3dVector(color_arr)

        return pcd

    def visualize(self): 
        ''' 
        Visualize the Open3d PointCloud attribute.
        ''' 
        if not self.get_o3d_pc(): 
            raise Exception("Open3d PointCloud does not exist.")
        o3d.visualization.draw_geometries([self.get_o3d_pc()])


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
        

class CornersPointCloud(PointCloud): 
    def __init__(self, points_txt_path): 
        super().__init__(points_txt_path)

        self.calibrated_corners_dict = None 
        self.calibrated_o3d_pc = None

        self.preprocessed_corners_dict = None
        self.preprocessed_o3d_pc = None 

        self.o3d_downsample_voxel_size = 1
        self.o3d_outlier_radius = 0.01
        self.o3d_outlier_nb_points = 10
    
    def read_points_from_txt(self):
        ''' 
        Read point id, point coords and colors from COLMAP points3D.txt file.
        '''
        f = open(self.get_points_txt_path(), 'r')
        lines = f.readlines()

        points_dict = dict()
        for line in lines: 
            point_id = int(line.split()[0])
            xyz = np.array([float(i) for i in line.split()[1:4]])
            colors = np.array([float(i) for i in line.split()[4:7]]) / 256
            
            point = Point(point_id, xyz, colors)
            points_dict[point_id] = point

        return points_dict
    
    def set_calibration_params(self, center, axis_angle, scale): 
        ''' 
        
        '''
        self.center = center 
        self.axis_angle = axis_angle 
        self.scale = scale

    def calibrate(self):
        ''' 
        Calibrate the point cloud w.r.t the calibration markers.
        '''
        pcd = self.get_o3d_pc() 

        center = self.center 
        axis_angle = self.axis_angle 
        scale = self.scale 
        
        pcd_centered_calibration = copy.deepcopy(pcd).translate(-center)
        pcd_centered_origin = copy.deepcopy(pcd_centered_calibration).translate(-pcd_centered_calibration.get_center())

        pcd_scaled = copy.deepcopy(pcd_centered_origin).scale(scale, center=(0,0,0))
        R = pcd_centered_calibration.get_rotation_matrix_from_axis_angle(axis_angle)
        pcd_aligned = copy.deepcopy(pcd_scaled).rotate(R, center=(0,0,0))

        self.calibrated_o3d_pc = pcd_aligned

    def remove_outlier_points(self): 
        ''' 
        Remove non-object points by ...
        '''
        pcd = self.calibrated_o3d_pc
        new_pcd = o3d.geometry.PointCloud()
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        sphere = 1 / 4 #TODO: why 
        center = (0,0,0)

        # Calculate distances to center, set new points
        distances = np.linalg.norm(points - center, axis=1)
        new_pcd.points = o3d.utility.Vector3dVector(points[distances <= sphere])
        new_pcd.colors = o3d.utility.Vector3dVector(colors)

        self.preprocessed_o3d_pc = new_pcd
    
    def remove_noise(self): 
        ''' 
        Downsample and remove points that have few neighbors around them.
        '''
        if self.preprocessed_o3d_pc is None: 
            raise Exception("corners have not been calibrated.")
        
        pcd = self.preprocessed_o3d_pc 
        # pcd_downsampled = copy.deepcopy(pcd).voxel_down_sample(self.o3d_downsample_voxel_size)
        pcd_downsampled = pcd
        pcd_preprocessed, _  = copy.deepcopy(pcd_downsampled).remove_radius_outlier(nb_points=self.o3d_outlier_nb_points, radius=self.o3d_outlier_radius)

        self.preprocessed_o3d_pc = pcd_preprocessed


class Piece: 
    def __init__(self, colmap_txt_path, corners_txt_path, calibration_colors): 
        self.colmap_pc = ColmapPointCloud(colmap_txt_path, calibration_colors)
        self.corners_pc = CornersPointCloud(corners_txt_path)
        self.calibration_colors = calibration_colors

    def get_colmap_pc(self): 
        return self.colmap_pc 
    
    def get_corners_pc(self): 
        return self.corners_pc
    
    def get_calibrated_corner_pc(self): 
        return self.corners_pc.calibrated_o3d_pc
    
    def get_preprocessed_corner_pc(self): 
        return self.corners_pc.preprocessed_o3d_pc
    
    def get_calibration_colors(self): 
        return self.calibration_colors

    def calibrate_corners_pc(self): 
        ''' 
        
        '''
        self.colmap_pc.get_calibration_centroids()
        pc_center = self.colmap_pc.get_center() 
        pc_axis_angle = self.colmap_pc.get_z_axis_aligned_rotation()
        pc_scale = self.colmap_pc.get_calibrated_scale()

        self.corners_pc.set_calibration_params(pc_center, pc_axis_angle, pc_scale)
        self.corners_pc.calibrate()

    def preprocess_corners_pc(self): 
        ''' 
        
        '''
        self.corners_pc.remove_outlier_points()
        self.corners_pc.remove_noise()


    def write_to_file(self, filepath, pcd): 
        ''' 
        Write point cloud to .ply file.
        '''
        o3d.io.write_point_cloud(filepath, pcd)

    # have different options for visualization and inspection
    def visualize(self): 
        pass

# dictionary to store calibration (marker) RGB colours and the distance threshold to be considered matched.
CUP_CALIBRATION_COLOURS = {
        CalibrationColour.BLUE: [np.array([151,194,218]) / 256, 0.15],
        CalibrationColour.PINK: [np.array([231,173,197]) / 256, 0.05], # the cup is purple, so the threshold is stricter to filter out cup points
        CalibrationColour.YELLOW: [np.array([235,244,150]) / 256, 0.15]
    }

piece1 = Piece(piece1_colmap_path, piece1_3d_corners_path, CUP_CALIBRATION_COLOURS)
piece2 = Piece(piece2_colmap_path, piece2_3d_corners_path, CUP_CALIBRATION_COLOURS)

# piece2.colmap_pc.get_calibration_centroids()
# piece1.corners_pc.visualize()
# piece1.colmap_pc.visualize_calibration_points(with_non_calibration=False, with_centroids=True)

piece1.calibrate_corners_pc()
piece2.calibrate_corners_pc() 

piece1.write_to_file(piece1_calibrated_3d_corners_path, piece1.get_calibrated_corner_pc())
piece2.write_to_file(piece2_calibrated_3d_corners_path, piece2.get_calibrated_corner_pc())

piece1.preprocess_corners_pc()
piece2.preprocess_corners_pc()

piece1.write_to_file(piece1_preprocessed_3d_corners_path, piece1.get_preprocessed_corner_pc())
piece2.write_to_file(piece2_preprocessed_3d_corners_path, piece2.get_preprocessed_corner_pc())




