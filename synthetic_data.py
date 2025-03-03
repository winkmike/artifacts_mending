import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from min_spanning import *

''' 
Steps: 
    - Generate a 3D template curve with function and end points. The data must fit the unit sphere.
    - (optional) Extend different functions outside the 2 end points.
    - Generate search curve
        + Apply translation and rotation transformation. 
        + Shorten the 2 end points
    - Add noise.
    - Randomly sample along the noisy curve.  
    - Use MST to generate graph. 
    - Write graph to file.
    - Visualize graph.
'''

# synthetic functions
fnc_1 = [lambda t: 0.8* np.cos(5*t) + 0.6*np.sin(4*t),
         lambda t: 0.3* np.sin(5*t) + 0.5*np.cos(4*t), 
         lambda t: t]
t_1 = np.linspace(-1, 1, 1000)


def generate_template_curve(fnc, t_input): 
    ''' 
    Generate template curve given 2 endpoints and the function to generate xyz coordinates.
    ''' 
    func_x, func_y, func_z = fnc

    # template curve is represented by 100 control points along the curve.
    template_curve = np.array([(func_x(i), func_y(i), func_z(i)) for i in t_input])

    return template_curve

def generate_search_curve(fnc, t_input, min_gap=0.67, max_gap=0.75): 
    ''' 
    Generate search curve by randomly shorten the endpoints of template curve and applying random isometry transformations.
    '''

    # randomly choose new endpoints from template's input 
    id_min_gap = t_input.shape[0] * min_gap
    id_max_gap = t_input.shape[0] * max_gap

    # make sure the new endpoints not too short or long
    new_start_id = np.random.randint(0, t_input.shape[0] - id_min_gap)
    new_end_id = np.random.randint(new_start_id + min_gap, min(new_start_id + id_max_gap, t_input.shape[0] - 1))
    new_t_input = np.linspace(t_input[new_start_id], t_input[new_end_id], 100)

    # generate control points for search curve
    func_x, func_y, func_z = fnc
    search_control_points = np.array([(func_x(i), func_y(i), func_z(i)) for i in new_t_input])
    search_control_points = search_control_points - np.mean(search_control_points, axis=0) # centering search curve

    # generate params for a random isometry transformation
    eps = np.random.choice([-1, 1]) # reflection if eps = -1, orientation preserving if eps = 1 
    theta = np.random.choice([-2*np.pi, 2*np.pi])
    tx,ty = 0,0  # translation is already been mimicked by shortening the template curve then centering 
    
    # create transformation matrix T 
    T = np.array([[eps*np.cos(theta), -1*np.sin(theta), tx],
                  [eps*np.sin(theta), np.cos(theta), ty],
                  [0, 0, 1]])
    
    search_curve = np.matmul(T, search_control_points.T).T

    return search_curve

def add_noise(curve, gs_mean=0, gs_std=0.03):
    ''' 
    Add Gaussian noise to the curve data. 
    ''' 
    noise = np.random.normal(gs_mean, gs_std, curve.shape[0])
    x, y, z = curve[:, 0], curve[:, 1], curve[:, 2]
    x_noise, y_noise, z = x + noise, y + noise, z 
    curve_noise = np.vstack((x_noise, y_noise, z)).T

    return curve_noise
    
def sample_points_along_curve(curve, min_thresh=0.9, max_thresh=0.95): 
    ''' 
    Randomly picked out a number of points from curve. The number of points left is at least min_thresh * original number of points.
    ''' 
    keep_rate = np.random.uniform(low=min_thresh, high=max_thresh)
    print(keep_rate)
    sampled_ids = np.random.choice(len(curve) - 1, size=int(curve.shape[0] * keep_rate), replace=False)
    sampled_points = curve[sampled_ids]

    return sampled_points

def get_graph(curve): 
    ordered_graph = get_mst_graph_from_arr(curve)

    xyz_graphs = [] 

    for ordered_branch in ordered_graph: 
        xyz_graph = []  
        for point_id in ordered_branch: 
            point = curve[point_id]
            xyz_graph.append(point)
        
        xyz_graphs.append(xyz_graph) 

    return np.asarray(xyz_graphs)

def write_to_file(template_graph, search_graph): 
    pass 

def visualize_curve_data(template_curve, search_curve): 
    # create subplots with 1 row and `num_graphs` columns
    fig = make_subplots(
        rows=1, cols=2,    
        specs=[[{'type': 'scatter3d'}] * 2]  
    )

    # iterate over each graph and add it to a separate subplot
    graphs = [template_curve, search_curve]
    for i, graph in enumerate(graphs):
        x, y, z = zip(*graph) 
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',  
                marker=dict(size=4),  
                line=dict(width=3)
            ),
            row=1, col=i + 1  #
        )

    fig.update_layout(
        title="Synthetic Fragments",
        showlegend=True
    )

    fig.show()

def visualize_graph_data(template_graph, search_curve):
    # create subplots with 1 row and `num_graphs` columns
    fig = make_subplots(
        rows=1, cols=2,    
        specs=[[{'type': 'scatter3d'}] * 2]  
    )

    # iterate over each graph and add it to a separate subplot
    graphs = [template_graph, search_curve]
    for i, graph in enumerate(graphs):
        for branch in graph: 
            x, y, z = zip(*branch) 
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',  
                    marker=dict(size=4),  
                    line=dict(width=3)
                ),
                row=1, col=i + 1  #
            )

    fig.update_layout(
        title="Synthetic Fragments",
        showlegend=True
    )

    fig.show()

def generate_synthetic_data(fnc, t_input): 
    template_curve = generate_template_curve(fnc, t_input)
    search_curve = generate_search_curve(fnc, t_input)

    template_curve = sample_points_along_curve(add_noise(template_curve))
    search_curve = sample_points_along_curve(add_noise(search_curve))

    visualize_curve_data(template_curve, search_curve)

    template_graph = get_graph(template_curve)
    search_graph = get_graph(search_curve)

    # write_to_file(template_graph, search_graph)
    visualize_graph_data(template_graph, search_graph)


def main(): 
    generate_synthetic_data(fnc_1, t_1)

if __name__ == "__main__": 
    main()




