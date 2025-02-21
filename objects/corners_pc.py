import copy
import numpy as np
import open3d as o3d

from objects.point import Point 
from objects.point_cloud import PointCloud

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

        o3d.visualization.draw_geometries([new_pcd])
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