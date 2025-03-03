import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

piece1_preprocessed_pcd = "./output/piece-1-preprocessed-corner-3d.ply"
piece2_preprocessed_pcd = "./output/piece-2-preprocessed-corner-3d.ply"

output_graph = "./output/graphs"

def read_ply_file(pcd_file): 
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd)

    return points

def get_mst(points):
    n = len(points)
    edges = []
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i]-points[j])
            if d < 0.1:
                edges.append((i, j, d))

    edges = sorted(edges, key=lambda x: x[2])
    parents = [i for i in range(n)]
    ranks = [0 for i in range(n)]

    def find(i):
        if parents[i] != i:
            parents[i] = find(parents[i])
        return parents[i]
    
    def union(i, j):
        i_root = find(i)
        j_root = find(j)
        if i_root == j_root:
            return
        if ranks[i_root] < ranks[j_root]:
            parents[i_root] = j_root
        elif ranks[i_root] > ranks[j_root]:
            parents[j_root] = i_root
        else:
            parents[j_root] = i_root
            ranks[i_root] += 1

    mst = []
    for edge in tqdm(edges):
        i, j, _ = edge
        if find(i) != find(j):
            union(i, j)
            mst.append(edge)
    return mst

def find_longest_path(mst):
    # find a sequence of indices that lead to the longest path
    # in the minimum spanning tree
    adj_list = {}
    for edge in mst:
        i, j, _ = edge
        if i not in adj_list:
            adj_list[i] = []
        if j not in adj_list:
            adj_list[j] = []
        adj_list[i].append(j)
        adj_list[j].append(i)

    def dfs(i, visited):
        visited.add(i)
        max_len = 0
        max_path = []
        for j in adj_list[i]:
            if j not in visited:
                path, length = dfs(j, visited)
                if length > max_len:
                    max_len = length
                    max_path = path
        return [i] + max_path, max_len + 1
    
    path, _ = dfs(0, set())
    return path

def get_node_counts(edges):
    counts = {}
    for edge in edges:  
        i, j, _ = edge
        if i not in counts:
            counts[i] = 0
        if j not in counts:
            counts[j] = 0
        counts[i] += 1
        counts[j] += 1
    return counts

def prune_mst(mst, min_N=10):
    # prune the mst such that the depth is at least min_N for 2 separate branches
    
    edges = mst.copy()
    for _ in range(min_N):
        # remove all leaf nodes
        counts = get_node_counts(edges)
        leaf_nodes = [i for i in counts if counts[i] == 1]
        if len(leaf_nodes) == 0:
            break
        new_edges = []
        for edge in edges:
            i, j, _ = edge
            if i not in leaf_nodes and j not in leaf_nodes:
                new_edges.append(edge)
        edges = new_edges
    return edges

def split_graph(edges):
    # any node with degree > 2 is a branching node
    # split at branch nodes to get separate branches
    node_counts = get_node_counts(edges)
    visited = set()
    branches = []
    for n, c in node_counts.items():
        if c > 2 or n in visited:
            continue
            
        branch = []
        branch_length = 0
        to_checks = [n]

        while True:
            if len(to_checks) == 0:
                break

            to_check = to_checks.pop()
            node_edges = [e for e in edges if e[0] == to_check or e[1] == to_check]
            for edge in node_edges:
                branch.append(edge)
                branch_length += edge[2]
                
                if edge[0] not in visited and node_counts[edge[0]] < 3:
                    visited.add(edge[0])
                    to_checks.append(edge[0])

                if edge[1] not in visited and node_counts[edge[1]] < 3:
                    visited.add(edge[1])
                    to_checks.append(edge[1])
        if branch_length > 1:
            branches.append(branch)

    return branches

def order_graph(branches):
    ordered_branches = []

    for branch in branches: 
        adj_list = {}
        for edge in branch:
            i, j, _ = edge
            if i not in adj_list:
                adj_list[i] = []
            if j not in adj_list:
                adj_list[j] = []
            adj_list[i].append(j)
            adj_list[j].append(i)
        counts = get_node_counts(branch)
        print(counts)
        leaf_nodes = [i for i in counts if counts[i] == 2]
         
        def dfs(i, visited):
            visited.add(i)
            path = []
            for j in adj_list[i]:
                if j not in visited:
                    path = dfs(j, visited)
            return [i] + path
        
        start_node = leaf_nodes[0]
        ordered_branch = dfs(start_node, set())
        ordered_branches.append(ordered_branch)

    return ordered_branches        

def get_mst_graph_from_path(pcd_path): 
    points = read_ply_file(pcd_path)
    mst = get_mst(points)
    mst_prune = prune_mst(mst)
    branches = split_graph(mst_prune)
    ordered_graph = order_graph(branches)

    return ordered_graph

def get_mst_graph_from_arr(points): 
    mst = get_mst(points)
    mst_prune = prune_mst(mst)
    branches = split_graph(mst_prune)
    ordered_graph = order_graph(branches)

    return ordered_graph

def write_graph_to_json(graphs, output_filepath):
    folder = os.path.dirname(output_filepath)
    if not os.path.exists(folder):
        os.mkdir(folder)

    res = "piece_id, x, y, z" 
    res += "\n"

    for i, graph in enumerate(graphs):
        res += f"{i} "
        for point in graph:
            x, y, z = point 
            res += f"{x} {y} {z} "
        res += "\n"

    with open(output_filepath, "w") as f: 
        f.write(res)

def create_graphs(piece1_pcd_path, piece2_pcd_path, output_filepath):
    points1, points2 = read_ply_file(piece1_pcd_path), read_ply_file(piece2_pcd_path)
    graph1, graph2 = get_mst_graph_from_arr(points1), get_mst_graph_from_arr(points2)

    graphs = [graph1, graph2]
    xyz_graphs = [] 

    for graph in graphs:
        points = points1 if graph == graph1 else points2
        for ordered_branch in graph: 
            xyz_graph = []  
            for point_id in ordered_branch: 
                point = points[point_id]
                xyz_graph.append(point)
            
            xyz_graphs.append(xyz_graph) 

            pcd_viz = o3d.geometry.PointCloud()
            pcd_viz.points = o3d.utility.Vector3dVector(np.array(xyz_graph))
            o3d.visualization.draw_geometries([pcd_viz])

    write_graph_to_json(xyz_graphs, output_filepath)

    return xyz_graphs

def main(): 
    create_graphs(piece1_preprocessed_pcd, piece2_preprocessed_pcd, output_filepath=f"{output_graph}/cup.txt")

if __name__ == "__main__": 
    main()