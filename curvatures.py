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

def read_graphs(paths): 
    graphs = []
    for path in paths: 
        with open(path, "rb") as f: 
            graph = pickle.load(f)
        graphs.append(graph)

    return graphs


def whole_cubic_spline_interpolation(graph, sampling_rate_inc): 
    ''' 
    Form a curve by applying cubic spline interpolation to the graph. Then, sample equidistant points with distance d.
    '''
    path_x = np.asarray([point[0] for point in graph])
    path_y = np.asarray([point[1] for point in graph])
    path_z = np.asarray([point[2] for point in graph])

    path_t = np.linspace(0, 1, path_x.size)
    original_points = np.hstack((path_x.reshape((path_x.size,1)),path_y.reshape((path_y.size,1)),path_z.reshape((path_z.size,1))))

    spline = CubicSpline(path_t, original_points)

    t = numpy.linspace(numpy.min(path_t),numpy.max(path_t),path_x.size * sampling_rate_inc)
    r = spline(t)

    # Plot using Plotly
    fig = go.Figure()

    # Plot original points
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='markers',
        marker=dict(size=3, color='red'),
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

def sample_equidistant_points(spline, original_points, d): 
    ''' 
    
    '''
     # Step 3: Compute the arc-length parameterization
    t_fine = np.linspace(0, 1, 1000)  # High-res sampling of spline
    fine_points = spline(t_fine)  # Interpolated points
    distances = np.sqrt(np.sum(np.diff(fine_points, axis=0) ** 2, axis=1))  # Segment lengths
    cumulative_lengths = np.hstack(([0], np.cumsum(distances)))  # Cumulative arc length

    # Step 4: Sample equidistant points at intervals of `d`
    num_samples = int(cumulative_lengths[-1] // d) + 1
    desired_lengths = np.linspace(0, cumulative_lengths[-1], num_samples)
    equidistant_t = np.interp(desired_lengths, cumulative_lengths, t_fine)  # Get parameter t for equal distances
    equidistant_points = spline(equidistant_t)  # Get (x, y, z) coordinates

    #Plot with Plotly
    fig = go.Figure()

    # Plot original points
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in original_points], y=[p[1] for p in original_points], z=[p[2] for p in original_points],
        mode='markers', marker=dict(size=4, color='red'),
        name='Original Points'
    ))

    # Plot smooth spline curve
    fig.add_trace(go.Scatter3d(
        x=fine_points[:, 0], y=fine_points[:, 1], z=fine_points[:, 2],
        mode='lines', line=dict(width=2, color='blue'),
        name='Cubic Spline'
    ))

    # Plot equidistant points
    fig.add_trace(go.Scatter3d(
        x=equidistant_points[:, 0], y=equidistant_points[:, 1], z=equidistant_points[:, 2],
        mode='markers', marker=dict(size=6, color='green'),
        name='Equidistant Points'
    ))

    # Layout settings
    fig.update_layout(
        title="3D Cubic Spline Interpolation with Equidistant Points",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

    return equidistant_points

def interpolate_cubic_spline(curve, scale): 
    ''' 
    
    '''
    def generate_points():
        pass 

    def generate_weights(): 
        pass

def compute_curvature(): 
    pass 

def curvature(graph_paths): 
    graphs = read_graphs(graph_paths)
    
    for graph in graphs:
        spline, original_points = whole_cubic_spline_interpolation(graph, sampling_rate_inc)
        equidistant_points = sample_equidistant_points(spline, original_points, d)

def main(): 
    curvature(graph_paths)

if __name__ == "__main__": 
    main()