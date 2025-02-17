import open3d as o3d
import numpy as np
import os
import copy

import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

from sklearn.cluster import DBSCAN

piece1_preprocessed_corners = "./output/piece-1-preprocessed-corner-3d.ply"
piece2_preprocessed_corners = "./output/piece-2-preprocessed-corner-3d.ply"

ref_points_path = "./output/ref_points.txt"

#TODO: tidy up IO code
def choose_ref_correspondence(piece1_pcd, piece2_pcd, ref_points_path): 
    '''
    Use Open3D to manually choose 1 ref correspondence and write to file.
    '''
    def pick_points(pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        return vis.get_picked_points()

    if not os.path.isfile(ref_points_path):
        ref1_id = pick_points(piece1_pcd)[0]
        ref2_id = pick_points(piece2_pcd)[0]

        txt = f"{ref1_id} \n{ref2_id}"
        with open(f"{ref_points_path}", "w") as f: 
            f.write(txt)
    else: 
        with open(f"{ref_points_path}") as f: 
            lines = [line.rstrip() for line in f]
        ref1_id = int(lines[0])
        ref2_id = int(lines[1])

    return ref1_id, ref2_id

#TODO: optimization: once ref_point is found, compute the dist array inside PointCloud class -> each time changing ball radius r can just query from dist array
def ball_intersect(ref_id, r, piece_pc, thickness): 
    '''
    Find intersection points between the ball created from (center ref_point, radius r) and the piece's point cloud.

    As the points are sparse, intersection is defined as lying within the "ball shell" between (center ref_point, radius r) 
    and (center ref_point, radius r + thickness)
    '''
    ref_point = piece_pc.points[ref_id]
    points = np.asarray(piece_pc.points)
    intersections = [] 

    for i in range(points.shape[0]): 
        point = points[i]
        if r <= np.linalg.norm(point - ref_point) <= r + thickness: 
            intersections.append(i)

    return intersections 

def angle(ref_point, intersection): 
    ''' 
    Find 3D angles (alpha, beta, gamma) between ref_point and intersection, assuming the origin is set at ref_point.
    '''
    # move ref_point to the origin
    vec = intersection - ref_point
    vec_norm = np.linalg.norm(vec)

    # define x,y,z-axis 
    x_axis = np.array([1,0,0])
    y_axis = np.array([0,1,0])
    z_axis = np.array([0,0,1])

    # calculate angles 
    alpha = np.arccos(vec.dot(x_axis) / vec_norm)
    beta = np.arccos(vec.dot(y_axis) / vec_norm)
    gamma = np.arccos(vec.dot(z_axis) / vec_norm)

    return alpha, beta, gamma

def angle_centroid(angles): 
    ''' 
    
    '''
    angles = np.array(angles)

    dbscan = DBSCAN() 
    labels = dbscan.fit_predict(angles)
    valid_labels = np.array(labels) != -1
    valid_angles = angles[valid_labels]
    valid_labels = labels[valid_labels]

    # valid_angles = angles
    # valid_labels = labels

    unique_labels = np.unique(valid_labels)
    angle_centroids = np.array([valid_angles[valid_labels == label].mean(axis=0) for label in unique_labels])

    return angle_centroids


def plot_ball_pcd(piece_pcd, r_init, r_end, inc): 
    ''' 
    
    '''
    sphere_linesets = []
    r = r_init
    while r < r_end:
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=7)
        sphere_mesh.compute_vertex_normals()

        # Extract edges to form a wireframe
        lines = []
        for triangle in np.asarray(sphere_mesh.triangles):
            lines.append([triangle[0], triangle[1]])
            lines.append([triangle[1], triangle[2]])
            lines.append([triangle[2], triangle[0]])

        # Create LineSet (wireframe)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(sphere_mesh.vertices)),
            lines=o3d.utility.Vector2iVector(lines),
        )
        sphere_linesets.append(line_set)

        r += inc

    print(f'no of line sets = {len(sphere_linesets)}')
    o3d.visualization.draw_geometries([piece_pcd] + sphere_linesets)    


def plot_graph(records): 
    ''' 
    Plot 3D heat map to demonstrate relationship between radius r and angles (alpha, beta, gamma).
    '''
    record1, record2 = records 

    r1, alpha1, beta1, gamma1 = np.array(record1)[:, 0], np.array(record1)[:, 1], np.array(record1)[:, 2], np.array(record1)[:, 3]
    r2, alpha2, beta2, gamma2 = np.array(record2)[:, 0], np.array(record2)[:, 1], np.array(record2)[:, 2], np.array(record2)[:, 3]

    # Find global min and max for color scaling
    C_min = min(r1.min(), r2.min())
    C_max = max(r1.max(), r2.max())

    # Normalize C values to [0, 1] based on global scale
    C1_norm = (r1 - C_min) / (C_max - C_min)
    C2_norm = (r2 - C_min) / (C_max - C_min)

    # Define colormap
    colorscale = 'viridis'  # Choose any colorscale

    # Map normalized values to colors
    C1_colors = [pc.sample_colorscale(colorscale, v)[0] for v in C1_norm]
    C2_colors = [pc.sample_colorscale(colorscale, v)[0] for v in C2_norm]

    # Create subplots (1 row, 2 columns)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],  # 3D plots in both columns
        subplot_titles=("Piece 1", "Piece 2")  # Titles
    )

    # First dataset (left plot)
    fig.add_trace(go.Scatter3d(
        x=alpha1, y=beta1, z=gamma1,
        mode='markers',
        marker=dict(
            size=5,
            color=C1_colors,  # Use mapped colors
            showscale=False
        ),
        name='Piece 1'
    ), row=1, col=1)

    # Second dataset (right plot)
    fig.add_trace(go.Scatter3d(
        x=alpha2, y=beta2, z=gamma2,
        mode='markers',
        marker=dict(
            size=5,
            color=C2_colors,  # Use mapped colors (same as C1)
            showscale=False
        ),
        name='Piece 2'
    ), row=1, col=2)

    # Add a reference color bar
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],  # Dummy trace for color bar
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            cmin=C_min, cmax=C_max,
            colorbar=dict(title="Radius", len=0.8),
            showscale=True
        ),
        name='Color Scale'
    ), row=1, col=2)  # Attach color bar to right plot

    # Update layout
    fig.update_layout(
        title='Side-by-Side 3D Angles Scatter Plots of 2 Pieces',
        width=1700,  # Adjust width for better display
        height=800,
        scene=dict(
            xaxis_title='alpha',
            yaxis_title='beta',
            zaxis_title='gamma'
        ),
        scene2=dict(  # Ensure same axis settings for both plots
            xaxis_title='alpha',
            yaxis_title='beta',
            zaxis_title='gamma'
        )
    )

    fig.show()

def find_correspondences(piece1_corners_path, piece2_corners_path, ref_points_path):
    piece1_pcd, piece2_pcd = o3d.io.read_point_cloud(piece1_corners_path), o3d.io.read_point_cloud(piece2_corners_path)
    ref1_id, ref2_id = choose_ref_correspondence(piece1_pcd, piece2_pcd, ref_points_path)

    piece1_pcd = copy.deepcopy(piece1_pcd).translate(-piece1_pcd.points[ref1_id])
    piece2_pcd = copy.deepcopy(piece2_pcd).translate(-piece2_pcd.points[ref2_id])

    pieces = [piece1_pcd, piece2_pcd]
    records = []
    r_init = 0.05
    r_end = 0.25
    thickness = 0.01
    inc = 0.01

    for piece_pcd in pieces: 
        r = r_init
        piece_record = []
        ref_id = ref1_id if piece_pcd == piece1_pcd else ref2_id 
        print(f'ref_id = {ref_id}')
        while True: 
            intersections = ball_intersect(ref_id, r, piece_pcd, thickness)

            angles = []
            for intersection in intersections: 
                ref_point, intersection_point = piece_pcd.points[ref_id], piece_pcd.points[intersection]
                alpha, beta, gamma = angle(ref_point, intersection_point)
                angles.append((alpha, beta, gamma))
                # piece_record.append((r, alpha, beta, gamma))

            angle_centroids = angle_centroid(angles)
            for (alpha, beta, gamma) in angle_centroids: 
                piece_record.append((r, alpha, beta, gamma))

            r += inc 

            if r > r_end: 
                records.append(piece_record)
                plot_ball_pcd(piece_pcd, r_init, r_end, inc)
                break
    
    plot_graph(records)

def main():
    find_correspondences(piece1_preprocessed_corners, piece2_preprocessed_corners, ref_points_path)

if __name__ == "__main__": 
    main()

