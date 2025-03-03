import plotly.graph_objects as go
from plotly.subplots import make_subplots

cup_graphs = "./output/graphs/cup.txt"

def read_graphs_txt(txt_filepath): 
    graphs = [] 
    with open(txt_filepath, "r") as f: 
        lines = f.readlines()[1:] # skip header row 
        for line in lines: 
            coords = line.split()[1:] # skip piece_id
            graph = [(float(coords[i]), float(coords[i+1]), float(coords[i+2])) for i in range(0, len(coords), 3)]
            graphs.append(graph)

    return graphs 

def visualize_graphs(graph_filepath): 
    graphs = read_graphs_txt(graph_filepath)
    
    # visualize with Plotly
    num_graphs = len(graphs)

    # create subplots with 1 row and `num_graphs` columns
    fig = make_subplots(
        rows=1, cols=num_graphs,
        specs=[[{'type': 'scatter3d'}] * num_graphs]  
    )

    # iterate over each graph and add it to a separate subplot
    for i, graph in enumerate(graphs):
        x, y, z = zip(*graph) 
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',  
                marker=dict(size=4),  
                line=dict(width=3)
            ),
            row=1, col=i + 1  #
        )

    fig.update_layout(
        title="3D Graphs of Fragments",
        showlegend=True
    )

    fig.show()


def main():
    visualize_graphs(cup_graphs)

if __name__ == "__main__": 
    main()
