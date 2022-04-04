import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from networkx.algorithms.simple_paths import all_simple_paths

def get_path(combination_matrix, ind_start, ind_end, step_size=None, hypotheses_num=None):
    if step_size is None:
        dist_matrix, predecessors = dijkstra(csgraph=-np.log(combination_matrix), directed=True,
                                             return_predecessors=True, indices=ind_start)
    else:
        dist_matrix, predecessors = dijkstra(csgraph=-np.log(combination_matrix)-np.log(1-step_size), directed=True,
                                             return_predecessors=True, indices=ind_start)
        dist_matrix = dist_matrix - np.log(step_size) + np.log(1-step_size)
    dist_matrix = np.exp(-dist_matrix)

    distance = dist_matrix[ind_end]
    path = [ind_end]
    ind_cur = ind_end
    while ind_cur != ind_start:
        ind_cur = predecessors[ind_cur]
        path.append(ind_cur)

    if hypotheses_num is not None:
        distance *= hypotheses_num
    return distance, path[::-1]

def get_most_influencial_node_path(combination_matrix, ind_start, step_size=None, hypotheses_num=None,
                                   include_itself=False, tol=1e-6):
    combination_matrix[combination_matrix<tol] = tol
    if step_size is None:
        dist_matrix, predecessors = dijkstra(csgraph=-np.log(combination_matrix), directed=True,
                                             return_predecessors=True, indices=ind_start)
    else:
        dist_matrix, predecessors = dijkstra(csgraph=-np.log(combination_matrix)-np.log(1-step_size), directed=True,
                                             return_predecessors=True, indices=ind_start)
        dist_matrix = dist_matrix - np.log(step_size) + np.log(1-step_size)
    dist_matrix = np.exp(-dist_matrix)
    ind_end = np.argmax(dist_matrix)
    if not include_itself:
        dist_matrix[ind_end] = 0
        ind_end = np.argmax(dist_matrix)
    distance = dist_matrix[ind_end]

    path = [ind_end]
    ind_cur = ind_end
    while ind_cur != ind_start:
        ind_cur = predecessors[ind_cur]
        path.append(ind_cur)

    if hypotheses_num is not None:
        distance *= hypotheses_num
    return distance, path[::-1]

def get_all_most_influenced_paths(combination_matrix, ind_start, step_size=None, hypotheses_num=None,
                                  include_itself=False, tol=1e-6):
    combination_matrix[combination_matrix<tol] = tol
    if step_size is None:
        dist_matrix, predecessors = dijkstra(csgraph=-np.log(combination_matrix), directed=True,
                                             return_predecessors=True, indices=ind_start)
    else:
        dist_matrix, predecessors = dijkstra(csgraph=-np.log(combination_matrix)-np.log(1-step_size), directed=True,
                                             return_predecessors=True, indices=ind_start)
        dist_matrix = dist_matrix - np.log(step_size) + np.log(1-step_size)
    dist_matrix = np.exp(-dist_matrix)
    ind_end = np.argmax(dist_matrix)
    if not include_itself:
        dist_matrix[ind_end] = 0
        ind_end = np.argmax(dist_matrix)

    if hypotheses_num is not None:
        dist_matrix *= hypotheses_num

    return dist_matrix

def get_all_influences(combination_matrix, ind_start, step_size, hypotheses_num, hop=1):
    def helper(path):
        constant = hypotheses_num * step_size * (1-step_size)**(len(path)-1)
        influence = 1
        for i, j in zip(path[:-1], path[1:]):
            influence *= combination_matrix[i, j]
        return constant * influence

    G = nx.from_numpy_matrix(combination_matrix)
    nodes_num = combination_matrix.shape[0]
    influences = np.zeros(nodes_num)
    for i in range(nodes_num):
        paths = all_simple_paths(G, i, ind_start, cutoff=hop)
        for path in paths:
            influences[i] += helper(path)
    print(influences)
    return influences