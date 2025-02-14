import numpy as np
import open3d as o3d

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




