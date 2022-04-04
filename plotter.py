import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl

LIGHT_BLUE = "#8ECAE6"
BLUE = "#3A7CB8"#"#219EBC"
DARK_BLUE = "#3A7CB8" #"#035177"
YELLOW = "#FFB703"
ORANGE = "#E5AD02" #"#FB8500"
RED = "#F04532" #"#F00314"
GREEN = "#75A43A" #"#14B37D"

def plot_agents(agents, times, belief_history, state_true, state_1=None):
    fig, axs = plt.subplots(2, agents // 2, figsize=(agents * 2, 6))
    axs = [item for sublist in axs for item in sublist]
    for agent, ax in enumerate(axs):
        ax.title.set_text('Agent ' + str(agent))
        ax.plot(list(range(len(belief_history))),
                np.array(belief_history)[:, state_true, agent],
                label='True state', color='red')
        if state_1 is not None:
            ax.plot(list(range(len(belief_history))),
                    np.array(belief_history)[:, state_1, agent],
                    label='False state', color='navy')
        ax.legend()
    plt.show()


def plot_combination_matrix_evolution(agents, times, combination_matrix, combination_matrix_list, projection,
                                      matrix_index=0):
    fig, axs = plt.subplots(2, agents // 2, figsize=(agents * 2, 6))
    fig.suptitle('$C[' + str(matrix_index) + ', :]$')
    axs = [item for sublist in axs for item in sublist]
    for i, ax in enumerate(axs):
        cs = [c[matrix_index * agents + i] for c in combination_matrix_list]
        if projection:
            ax.set_ylim([0, 1])
        ax.plot(list(range(times + 1)), cs, color='navy', label='learned')
        ax.set_ylim((0, 1))
        ax.hlines(combination_matrix[matrix_index, i], xmin=0, xmax=times - 1, color='red', label='true')
        ax.legend()
    plt.show()


def plot_combination_matrix_evolution_one(agents, times, combination_matrix, combination_matrix_list, projection,
                                          matrix_indexes=(0, 0)):
    plt.figure(figsize=(6, 4))
    plt.title('$A[' + str(matrix_indexes[0]) + ',' + str(matrix_indexes[1]) + ']$')
    cs = [c[matrix_indexes[0] * agents + matrix_indexes[1]] for c in combination_matrix_list]
    if projection:
        plt.ylim([0, 1])
    plt.plot(list(range(times + 1)), cs, color='navy', label='learned')
    plt.ylim((0, 1))
    plt.hlines(combination_matrix[matrix_indexes[0], matrix_indexes[1]], xmin=0, xmax=times - 1, color='red',
               label='true')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('combination weight')
    plt.show()

def plot_error(times, errors, scale='linear', error_comparison=None,
               change_true_state=-1, perturbe_time=-1, perturbe=-1, t1=None, t2=None):
    if error_comparison is not None and perturbe<0 and perturbe_time<0:
        plt.plot(list(range(times)), error_comparison,
                 color=ORANGE, label='true state', alpha=1., linewidth=2.)
    plt.plot(list(range(times)), errors,
             color=DARK_BLUE, label='learned true state', alpha=1., linewidth=2.)
    if change_true_state>0:
        plt.vlines(change_true_state, ymin=0, ymax=np.array(errors).max(),
                   color=BLUE, label='time when the true state changes',
                   linestyles='dotted', linewidth=2)
    if perturbe_time>0:
        plt.vlines(perturbe_time, ymin=0, ymax=np.array(errors).max(),
                   color=BLUE, label='time when graph changes',
                   linestyles='dotted', linewidth=2)
    if perturbe>0:
        perturbe_list = []
        t = perturbe
        while t < times:
            perturbe_list.append(t)
            t += perturbe
        plt.vlines(perturbe_list, ymin=0, ymax=np.array(errors).max(),
                   color=BLUE, label='time when graph changes',
                   linestyles='dotted', linewidth=2)
    plt.legend()
    plt.yscale(scale)
    plt.xlabel('Time')
    plt.ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
    plt.grid()
    plt.show()
    if t1 and t2:
        if error_comparison is not None:
            plt.plot(np.arange(times)[t1:t2], np.array(error_comparison)[t1:t2],
                     color=ORANGE, label='true state', alpha=1., linewidth=2.)
        plt.plot(np.arange(times)[t1:t2], np.array(errors)[t1:t2],
                 color=DARK_BLUE, label='learned true state', alpha=1., linewidth=2.)
        if change_true_state > 0:
            plt.vlines(change_true_state,
                       ymin=np.array(errors)[t1:t2].min(),
                       ymax=np.array(errors)[t1:t2].max(),
                       color=BLUE, label='time when the true state changes',
                       linestyles='dotted', linewidth=2)
        plt.legend()
        plt.yscale(scale)
        plt.xlabel('Time')
        plt.ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
        plt.grid()
        plt.show()


def plot_error(times, errors, scale='linear', error_comparison=None,
               change_true_state=-1, perturbe_time=-1, perturbe=-1, t1=None, t2=None):
    if error_comparison is not None and perturbe<0 and perturbe_time<0:
        plt.plot(list(range(times)), error_comparison,
                 color=ORANGE, label='true state', alpha=1., linewidth=2.)
    plt.plot(list(range(times)), errors,
             color=DARK_BLUE, label='learned true state', alpha=1., linewidth=2.)
    if change_true_state>0:
        plt.vlines(change_true_state, ymin=0, ymax=np.array(errors).max(),
                   color=BLUE, label='time when the true state changes',
                   linestyles='dotted', linewidth=2)
    if perturbe_time>0:
        plt.vlines(perturbe_time, ymin=0, ymax=np.array(errors).max(),
                   color=BLUE, label='time when graph changes',
                   linestyles='dotted', linewidth=2)
    if perturbe>0:
        perturbe_list = []
        t = perturbe
        while t < times:
            perturbe_list.append(t)
            t += perturbe
        plt.vlines(perturbe_list, ymin=0, ymax=np.array(errors).max(),
                   color=BLUE, label='time when graph changes',
                   linestyles='dotted', linewidth=2)
    plt.legend()
    plt.yscale(scale)
    plt.xlabel('Time')
    plt.ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
    plt.grid()
    plt.show()
    if t1 and t2:
        if error_comparison is not None:
            plt.plot(np.arange(times)[t1:t2], np.array(error_comparison)[t1:t2],
                     color=ORANGE, label='true state', alpha=1., linewidth=2.)
        plt.plot(np.arange(times)[t1:t2], np.array(errors)[t1:t2],
                 color=DARK_BLUE, label='learned true state', alpha=1., linewidth=2.)
        if change_true_state > 0:
            plt.vlines(change_true_state,
                       ymin=np.array(errors)[t1:t2].min(),
                       ymax=np.array(errors)[t1:t2].max(),
                       color=BLUE, label='time when the true state changes',
                       linestyles='dotted', linewidth=2)
        plt.legend()
        plt.yscale(scale)
        plt.xlabel('Time')
        plt.ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
        plt.grid()
        plt.show()


def plot_graph(adjacency_matrix, adjacency_matrix_learned):
    recovered = adjacency_matrix_learned * adjacency_matrix
    missed = adjacency_matrix - recovered
    new = adjacency_matrix_learned - recovered
    G = nx.from_numpy_matrix(recovered)
    G_missed = nx.from_numpy_matrix(missed)
    G_new = nx.from_numpy_matrix(new)

    pos = nx.spring_layout(nx.from_numpy_matrix(adjacency_matrix))
    nx.draw_networkx_nodes(G, pos=pos, node_color=BLUE, node_size=50)
    nx.draw_networkx_edges(G, pos=pos, edge_color=DARK_BLUE, width=1., label='correctly recovered')
    nx.draw_networkx_edges(G_missed, pos=pos, edge_color='red', width=2., label='missed')
    nx.draw_networkx_edges(G_new, pos=pos, edge_color='blue', width=2., label='new')
    plt.legend()
    plt.show()


def plot_weighted_graph(combination_matrix, adjacency_matrix):
    G = nx.from_numpy_matrix(combination_matrix)
    edges = G.edges()
    weights = [G[u][v]['weight']*4 for u, v in edges]
    pos = nx.spring_layout(nx.from_numpy_matrix(adjacency_matrix))

    nx.draw_networkx_nodes(G, pos=pos, node_color=BLUE, node_size=50)
    nx.draw_networkx_edges(G, pos=pos, edge_color=DARK_BLUE, width=weights)
    plt.show()

def plot_weighted_graphs(combination_matrix, combination_matrix_learned,
                         adjacency_matrix, by_side=False, adj_regime=1):
    combination_matrix = combination_matrix + combination_matrix.T
    combination_matrix_learned = combination_matrix_learned + combination_matrix_learned.T

    pos = nx.spring_layout(nx.from_numpy_matrix(adjacency_matrix))
    if adj_regime == 5:
        N = combination_matrix.shape[0]
        pos = {}
        pos[0] = np.array([-1, 0])
        pos[1] = np.array([1, 0])
        for k in range(2, N):
            pos[k] = np.array([0, -1 + 2*(k-2) / (N-3)])
    G = nx.from_numpy_matrix(combination_matrix)
    edges = G.edges()
    weights = [G[u][v]['weight']*5 for u, v in edges]

    G_learned = nx.from_numpy_matrix(combination_matrix_learned)
    edges_learned = G_learned.edges()
    weights_learned = [G_learned[u][v]['weight']*5 for u, v in edges_learned]
    if by_side:
        fig, axs = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=[4, 4]), figsize=(10, 4))

        axs[0].title.set_text('true graph')
        nx.draw_networkx_nodes(G, pos=pos, node_color=BLUE, node_size=100, ax=axs[0])
        nx.draw_networkx_edges(G, pos=pos, edge_color=DARK_BLUE, width=weights, ax=axs[0])

        axs[1].title.set_text('estimated graph')
        nx.draw_networkx_nodes(G_learned, pos=pos, node_color=BLUE, node_size=100, ax=axs[1])
        nx.draw_networkx_edges(G_learned, pos=pos, edge_color=DARK_BLUE, width=weights_learned, ax=axs[1])
        plt.show()
    else:
        plt.figure(figsize=(8, 7))
        nx.draw_networkx_nodes(G, pos=pos, node_color=ORANGE, node_size=350)
        nx.draw_networkx_edges(G, pos=pos, edge_color=DARK_BLUE, width=weights)
        plt.show()

        plt.figure(figsize=(8, 7))
        nx.draw_networkx_nodes(G_learned, pos=pos, node_color=ORANGE, node_size=350)
        nx.draw_networkx_edges(G_learned, pos=pos, edge_color=DARK_BLUE, width=weights_learned)
        plt.show()



def plot_adjacency_old(matrix, matrix_learned):
    vmin = min(matrix_learned.min(), matrix.min())
    vmax = max(matrix_learned.max(), matrix.max())
    sns.heatmap(matrix, vmin=vmin, vmax=vmax, cmap='mako')
    plt.show()
    sns.heatmap(matrix_learned, vmin=vmin, vmax=vmax, cmap='mako')
    plt.show()


def plot_adjacency(matrix, matrix_learned, by_side=False):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [DARK_BLUE, LIGHT_BLUE, 'white'])
    #
    matrix_learned[matrix_learned<0.] = 0.
    #
    vmin = min(matrix_learned.min(), matrix.min())
    vmax = max(matrix_learned.max(), matrix.max())

    if by_side:
        fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[4, 4, 0.2]), figsize=(10, 4))

        axs[0].title.set_text('true graph')
        sns.heatmap(matrix, yticklabels=False, xticklabels=False, cbar=False,
                    ax=axs[0], vmin=vmin, cmap=cmap)
        axs[1].title.set_text('estimated graph')
        sns.heatmap(matrix_learned, yticklabels=False, xticklabels=False, cbar=False,
                    ax=axs[1], vmax=vmax, cmap=cmap)

        fig.colorbar(axs[1].collections[0], cax=axs[2])
        plt.show()
    else:
        plt.figure()
        sns.heatmap(matrix, yticklabels=False, xticklabels=False, cbar=True,
                    vmin=vmin, cmap=cmap)
        plt.show()

        plt.figure()
        sns.heatmap(matrix_learned, yticklabels=False, xticklabels=False, cbar=True,
                    vmin=vmin, cmap=cmap)
        plt.show()


def plot_losses(loss, avg_loss=True, loss_comparison=None):
    T = len(loss)
    if loss_comparison:
        if avg_loss:
            loss_comparison = np.cumsum(np.array(loss_comparison)) / np.arange(1, T + 1)
        plt.plot(np.arange(T), loss_comparison, color='red', label='loss (true state)', alpha=0.5, linewidth=5)
    if avg_loss:
        loss = np.cumsum(np.array(loss)) / np.arange(1, T+1)
    plt.plot(np.arange(T), loss, color='navy', label='loss', alpha=1.)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.show()


def plot_state_estimation(state_true_rate):
    plt.plot(state_true_rate, color='navy', label='rate of correcly classified states')
    plt.xlabel('time')
    plt.legend()
    plt.show()

def plot_state(state_true, change_true_state, t1=None, t2=None):
    if t1 is None:
        t1 = 0
        t2 = state_true.shape[0]
    plt.plot(np.arange(t1,t2), state_true[t1:t2], color='navy', label='correct true state indicator')
    plt.vlines(change_true_state, ymin=0, ymax=np.array(state_true[t1:t2]).max(),
               color=BLUE, label='time when state changes',
               linestyles='dotted', linewidth=2)
    plt.xlabel('time')
    plt.legend()
    plt.show()

def plot_belief(belief_history, change_true_state, t1=None, t2=None):
    if t1 is None:
        t1 = 0
        t2 = state_true.shape[0]
    plt.plot(np.arange(t1, t2), belief_history[t1:t2], color='navy', label='agent 1 belief')
    plt.vlines(change_true_state, ymin=np.array(belief_history[t1:t2]).min(), ymax=np.array(belief_history[t1:t2]).max(),
               color=BLUE, label='time when state changes',
               linestyles='dotted', linewidth=2)
    plt.xlabel('time')
    plt.legend()
    plt.show()


def plot_logs(logs, mean=None):
    plt.title('lambda evolution')
    plt.plot(np.arange(logs.shape[0]), logs[:,0,0], color='navy', alpha=0.5)
    if mean is not None:
        plt.hlines(mean[0, 0], 0, logs.shape[0]-1, color='red')
    plt.xlabel('time')
    plt.show()

def plot_weighted_graphs_path(combination_matrix_learned, adjacency_matrix, distance, path, adj_regime=1):
    combination_matrix_learned = combination_matrix_learned + combination_matrix_learned.T
    combination_matrix_learned[combination_matrix_learned<5e-2] = 0

    pos = nx.spring_layout(nx.from_numpy_matrix(adjacency_matrix))
    if adj_regime == 5:
        N = adjacency_matrix.shape[0]
        pos = {}
        pos[0] = np.array([-1, 0])
        pos[1] = np.array([1, 0])
        for k in range(2, N):
            pos[k] = np.array([0, -1 + 2*(k-2) / (N-3)])

    G_learned = nx.from_numpy_matrix(combination_matrix_learned)
    edges_learned = G_learned.edges()
    weights_learned = [G_learned[u][v]['weight'] * 5 for u, v in edges_learned]

    plt.figure(figsize=(8, 7))
    nx.draw_networkx_nodes(G_learned, pos=pos, node_color=ORANGE, node_size=350)
    #nx.draw_networkx_nodes(G_learned, pos=pos, node_color=ORANGE, node_size=25)
    nx.draw_networkx_edges(G_learned, pos=pos, edge_color=BLUE, width=weights_learned)

    edges_path = [(e_0, e_1) for e_0, e_1 in zip(path[:-1], path[1:])]
    weights_path = [G_learned[u][v]['weight'] * 5 for u, v in edges_path]
    weights_path_highlight = [G_learned[u][v]['weight'] * 25 for u, v in edges_path]
    nx.draw_networkx_nodes(G_learned, pos=pos, node_color=GREEN, node_size=350,
                           nodelist=[path[0]])
    nx.draw_networkx_nodes(G_learned, pos=pos, node_color=RED, node_size=350,
                           nodelist=[path[-1]])
    nx.draw_networkx_edges(G_learned, pos=pos, edge_color=RED, width=weights_path_highlight,
                           edgelist=edges_path, alpha=0.3)
    nx.draw_networkx_edges(G_learned, pos=pos, edge_color=RED, width=weights_path,
                           edgelist=edges_path)
    plt.show()


def plot_heatmap(combination_matrix_learned, adjacency_matrix, ind_0, distances, adj_regime=1):
    distances = distances/distances.max()
    combination_matrix_learned = combination_matrix_learned + combination_matrix_learned.T

    pos = nx.spring_layout(nx.from_numpy_matrix(adjacency_matrix))
    if adj_regime == 5:
        N = adjacency_matrix.shape[0]
        pos = {}
        pos[0] = np.array([-1, 0])
        pos[1] = np.array([1, 0])
        for k in range(2, N):
            pos[k] = np.array([0, -1 + 2*(k-2) / (N-3)])

    G_learned = nx.from_numpy_matrix(combination_matrix_learned)
    edges_learned = G_learned.edges()
    weights_learned = [G_learned[u][v]['weight'] * 5 for u, v in edges_learned]

    palette = cm.Reds
    colors = palette((distances - distances.min()) / (distances.max() - distances.min()))
    cmap = palette
    norm = mpl.colors.Normalize(vmin=distances.min(), vmax=distances.max())

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 15]}, figsize=(8, 7))
    cb1 = mpl.colorbar.ColorbarBase(ax[0], cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    nx.draw_networkx_nodes(G_learned, pos=pos, node_color=DARK_BLUE, node_size=350, ax=ax[1])
    nx.draw_networkx_nodes(G_learned, pos=pos, node_color=colors, node_size=250, ax=ax[1])
    nx.draw_networkx_nodes([ind_0], pos=pos, node_color=GREEN, node_size=300, ax=ax[1])
    nx.draw_networkx_edges(G_learned, pos=pos, edge_color=DARK_BLUE, width=weights_learned, ax=ax[1])
    plt.show()