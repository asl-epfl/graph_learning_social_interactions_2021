"""Social Learning: Graph Topology Learning"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils
from social_learning import Network

np.random.seed(42)

agents = 4
states = 2
state_true = 0
times = 500000
params = 2
step_size = None

adj_matrix = utils.create_network(agents, 3, 0.5)
combination_matrix, centrality, connected = utils.generate_combination_weights(adj_matrix, 0)

initial_belief = np.random.rand(states, agents)
initial_belief = initial_belief / initial_belief.sum(0)[None, :]

likelihood = utils.create_likelihoods(agents, states, 0, params)

generator = utils.Generator(likelihood, state_true, 0)

network = Network(agents, states, state_true, adj_matrix, combination_matrix, likelihood, generator, initial_belief,
                  step_size=step_size)

for _ in tqdm(range(times)):
    network.step()

fig, axs = plt.subplots(2, 2)
(ax1, ax2), (ax3, ax4) = axs
for ax, agent in zip([ax1, ax2, ax3, ax4], range(agents)):
    ax.plot(list(range(times + 1)), np.array(network.belief_history)[:, state_true, agent],
            label='Agent , state true' + str(agent), color='red')
    ax.plot(list(range(times + 1)), np.array(network.belief_history)[:, 1, agent],
            label='Agent ' + str(agent), color='blue')
    ax.legend()
plt.show()