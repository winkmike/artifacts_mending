import numpy as np

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