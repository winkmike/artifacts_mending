import numpy
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

from min_spanning import * 

''' 
Steps: 
- Create the MST graph from point cloud.
- Create a curve by applying cubic spline interpolation to the MST graph. 
- Sample equally distanced points from the curve.
- For each point, sample d points to its "left" and "right" and fit another cubic spline interpolation.  
    + The cubic spline weights folows formula in the paper.
    + Compute the curvature vector. 
    + Compute the surface normal of the underlying triangle mesh (?).
    + Assign sign to the curvature vector accordingly based on the surface normal direction.
- Visualize the curves
'''

# PATH
graph_paths = ["./output/graph_1.pkl", "./output/graph_2.pkl"]

# PARAMS
sampling_rate_inc = 25 # interpolate a smoother curve, with sampling_rate_inc as many points, that goes exactly through the same data points.
d = 0.005 # sample distance along the fitted cubic spline interpolation of the graph
scale = 10

def read_graphs(paths): 
    graphs = []
    for path in paths: 
        with open(path, "rb") as f: 
            graph = pickle.load(f)
        graphs.append(graph)

    return graphs


def cubic_spline_interpolation(points, sampling_rate_inc=None, visualization=False): 
    ''' 
    Form a curve by applying cubic spline interpolation to the points.
    '''
    path_x = np.asarray([point[0] for point in points])
    path_y = np.asarray([point[1] for point in points])
    path_z = np.asarray([point[2] for point in points])

    path_t = np.linspace(0, 1, path_x.size)
    original_points = np.hstack((path_x.reshape((path_x.size,1)),path_y.reshape((path_y.size,1)),path_z.reshape((path_z.size,1))))

    spline = CubicSpline(path_t, original_points)

    if sampling_rate_inc and visualization:
        t = numpy.linspace(numpy.min(path_t),numpy.max(path_t),path_x.size * sampling_rate_inc)
        r = spline(t)

        # Plot using Plotly
        fig = go.Figure()

        # Plot original points
        fig.add_trace(go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='markers',
            marker=dict(size=2, color='red'),
            name='Original Points'
        ))

        # Plot interpolated spline curve
        fig.add_trace(go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2],
            mode='lines',
            line=dict(width=4, color='black'),
            name='Cubic Spline Curve'
        ))

        # Layout settings
        fig.update_layout(
            title='3D Cubic Spline Interpolation',
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()

    return spline, original_points

def compute_curvatures(curve, scale, sampling_rate_inc): 
    ''' 
    
    '''
    def generate_weights(): 
        pass

    num_points = 2 * scale + 1 # number of examined points with given scale
    curvatures = []

    # for sub_curve in sub_curves: 
    for i, point in enumerate(curve):
        if i < scale or i > curve.shape[0] - scale: 
            curvatures.append(0)
            continue 


        sub_curve = curve[(i-scale):(i+scale+1)]
        # fit cubic spline interpolation 
        spline, _ = cubic_spline_interpolation(sub_curve)

        # Fine sampling for smooth curvature computation
        t_original = np.linspace(0, 1, len(sub_curve))

        # First derivative (velocity)
        dS = spline(t_original, 1)

        # Second derivative (acceleration)
        ddS = spline(t_original, 2)

        # Compute cross product of r'(t) and r''(t)
        cross_product = np.cross(dS, ddS)

        # Compute curvature
        numerator = np.linalg.norm(cross_product, axis=1)  # |S'(t) × S''(t)|
        denominator = np.linalg.norm(dS, axis=1) ** 3  # |S'(t)|^3

        # Avoid division by zero
        curvature = np.zeros_like(numerator)
        valid = denominator > 1e-6
        curvature[valid] = numerator[valid] / denominator[valid]

        curvatures.append(curvature[scale])

    indices = np.arange(len(curvatures))  # Indices from 0 to len(curvature_list) - 1

    # fig = go.Figure()

    # fig.add_trace(go.Scatter(
    #     x=indices,
    #     y=curvatures,
    #     mode='markers+lines',
    #     marker=dict(size=6, color='blue'),
    #     line=dict(width=2,color='red',dash='dot',shape='spline'),  # Smooth dotted line
    #     name='Curvature'
    # ))

    # fig.update_layout(
    #     title="Curvature vs. Index",
    #     xaxis_title="Index",
    #     yaxis_title="Curvature"
    # )

    # fig.show()

    return curvatures
        

def sample_equidistant_points(spline, original_points, d, scale, sampling_rate_inc): 
    ''' 
    Sample equidistant points with distance `d` from the cubic spline. The distance is approximated by arc length.
    '''
    # Compute the arc-length approximation
    num_points = original_points.shape[0]
    t_fine = np.linspace(0, 1, num_points * sampling_rate_inc)  # high resolution sampling of spline
    fine_points = spline(t_fine)  # Interpolated points
    distances = np.sqrt(np.sum(np.diff(fine_points, axis=0) ** 2, axis=1))  # approximate arc length
    cumulative_lengths = np.hstack(([0], np.cumsum(distances)))  # cumulative arc length

    # Sample equidistant points at intervals of `d`
    num_samples = int(cumulative_lengths[-1] // d) + 1
    desired_lengths = np.linspace(0, cumulative_lengths[-1], num_samples)
    equidistant_t = np.interp(desired_lengths, cumulative_lengths, t_fine)  # get parameter t for equal distances
    equidistant_points = spline(equidistant_t)  # get (x, y, z) coordinates

    curvatures = compute_curvatures(equidistant_points, scale, sampling_rate_inc)

    #Plot with Plotly
    # fig = go.Figure()

    # # Plot original points
    # fig.add_trace(go.Scatter3d(
    #     x=[p[0] for p in original_points], y=[p[1] for p in original_points], z=[p[2] for p in original_points],
    #     mode='markers', marker=dict(size=4, color='red'),
    #     name='Original Points'
    # ))

    # # Plot smooth spline curve
    # fig.add_trace(go.Scatter3d(
    #     x=fine_points[:, 0], y=fine_points[:, 1], z=fine_points[:, 2],
    #     mode='lines', line=dict(width=2, color='blue'),
    #     name='Cubic Spline'
    # ))

    # # Plot equidistant points``
    # fig.add_trace(go.Scatter3d(
    #     x=equidistant_points[:, 0], y=equidistant_points[:, 1], z=equidistant_points[:, 2],
    #     mode='markers', marker=dict(size=3, color='green'),
    #     name='Equidistant Points'
    # ))

    # # Layout settings
    # fig.update_layout(
    #     title="3D Cubic Spline Interpolation with Equidistant Points",
    #     scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    #     margin=dict(l=0, r=0, b=0, t=40)
    # )

    # fig.show()

    # Plot with Plotly
    fig = go.Figure()

    # Plot original points
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in original_points], y=[p[1] for p in original_points], z=[p[2] for p in original_points],
        mode='markers', marker=dict(size=2, color='red'),
        name='Original Points'
    ))

    # Plot smooth spline curve
    fig.add_trace(go.Scatter3d(
        x=fine_points[:, 0], y=fine_points[:, 1], z=fine_points[:, 2],
        mode='lines', line=dict(width=2, color='blue'),
        name='Cubic Spline'
    ))

    # Normalize curvatures for color scaling
    norm_curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))

    # Plot equidistant points with curvature info
    fig.add_trace(go.Scatter3d(
        x=equidistant_points[:, 0], y=equidistant_points[:, 1], z=equidistant_points[:, 2],
        mode='markers', 
        marker=dict(
            size=3, 
            color=curvatures,  # Color points based on curvature values
            colorscale='Viridis',  # Use Viridis colormap (change if needed)
            showscale=True,  # Show color legend
            colorbar=dict(title='Curvature')
        ),
        name='Equidistant Points (Curvature)'
    ))

    # Layout settings
    fig.update_layout(
        title=f"3D Cubic Spline with Curvature with scale = {scale}, equidistance = {d}",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

    return equidistant_points


def curvature(graph_paths): 
    graphs = read_graphs(graph_paths)
    
    for graph in graphs:
        spline, original_points = cubic_spline_interpolation(graph, sampling_rate_inc, visualization=False)
        equidistant_points = sample_equidistant_points(spline, original_points, d, scale, sampling_rate_inc)

        # curvatures = compute_curvatures(equidistant_points, 15, sampling_rate_inc)
        # print(len(curvatures))
        
def main(): 
    curvature(graph_paths)

if __name__ == "__main__": 
    main()