import open3d as o3d
import numpy as np

# PATH 
piece1_3d_corners_path = "./output/piece-1-corner-3d.txt"
piece2_3d_corners_path = "./output/piece-2-corner-3d.txt"

def read_3d_corners_txt(txt_path): 
    f = open(txt_path,'r')
    lines = f.readlines()

    corners = [] 
    for line in lines: 
        point_id = int(line.split()[0])
        xyz = np.array([float(i) for i in line.split()[1:4]])
        colors = np.array([float(i) for i in line.split()[4:7]]) / 256
        corner = np.hstack((xyz, colors))
        corners.append(corner)

    return np.array(corners)

def visualize_3d_corners(corners): 
    points = corners[:, :3]
    colors = corners[:, 3:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def visualize(piece1_3d_corners_path, piece2_3d_corners_path): 
    for path in [piece1_3d_corners_path, piece2_3d_corners_path]: 
        corners = read_3d_corners_txt(path)
        pcd = visualize_3d_corners(corners)

if __name__ == "__main__": 
    visualize(piece1_3d_corners_path, piece2_3d_corners_path)