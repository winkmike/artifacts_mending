import open3d as o3d

from objects.colmap_pc import ColmapPointCloud 
from objects.corners_pc import CornersPointCloud

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