
import numpy as np
import yaml
from easydict import EasyDict

from enum import Enum

from objects.piece import Piece

#TODO: add typings
#TODO: separate out the RGB values to an Enum class
#TODO: change all self.{attribute} to getter
#TODO: inside CornersPointCloud, link point_dict with its o3d_pc at each preprocessing step


# calibration COLORS 
class CalibrationColour(Enum):
    BLUE = 1
    PINK = 2
    YELLOW = 3
    ORANGE = 4


# dictionary to store calibration (marker) RGB colours and the distance threshold to be considered matched.
CUP_CALIBRATION_COLOURS = {
        CalibrationColour.BLUE: [np.array([151,194,218]) / 256, 0.15],
        CalibrationColour.PINK: [np.array([231,173,197]) / 256, 0.05], # the cup is purple, so the threshold is stricter to filter out cup points
        CalibrationColour.YELLOW: [np.array([235,244,150]) / 256, 0.15]
    }


f = open("configs/configs.yaml")
cfg = EasyDict(yaml.safe_load(f))

piece1_colmap_path = cfg.piece1_colmap_path
piece2_colmap_path = cfg.piece2_colmap_path

piece1_3d_corners_path = cfg.piece1_3d_corners_path
piece2_3d_corners_path = cfg.piece2_3d_corners_path

piece1_calibrated_3d_corners_path = cfg.piece1_calibrated_3d_corners_path
piece2_calibrated_3d_corners_path = cfg.piece2_calibrated_3d_corners_path

piece1_preprocessed_3d_corners_path = cfg.piece1_preprocessed_3d_corners_path
piece2_preprocessed_3d_corners_path = cfg.piece2_preprocessed_3d_corners_path

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




