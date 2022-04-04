"""Social Learning: Graph Topology Learning"""
import numpy as np
from tqdm import tqdm
import matplotlib

import utils
import optimization
import plotter
import path
from parser import get_parser
from social_learning import Network

'''
pre-setup
'''
font = {'size': 12}
matplotlib.rc('font', **font)

parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)

'''
plots bool
'''
plot_agents = False
plot_logs = False
plot_loss = False
plot_error = True
plot_state_estimation = True
plot_evolution = False
plot_graph = True
plot_adjacency = True
plot_path = True
plot_explainability = True

'''
setup
'''
adj_matrix = utils.create_network(args.agents, args.adjacency_regime, args.er_prob)
combination_matrix, centrality, connected = utils.generate_combination_weights(adj_matrix, 0)
if not connected:
    raise (ValueError, "Not connected")

initial_belief = np.ones((args.states, args.agents))
initial_belief = initial_belief / initial_belief.sum(0)[None, :]
likelihood = utils.create_likelihoods(args.agents, args.states, args.likelihood_regime,
                                      args.params, var=args.likelihood_var)
generator = utils.Generator(likelihood, args.state_true, 0)
print('Likelihood\n', likelihood)

'''
procedure initialization
'''
network = Network(args.agents, args.states, args.state_true, adj_matrix, combination_matrix, likelihood, generator,
                  initial_belief, step_size=args.step_size)
network.step()
if args.start_fc:
    combination_matrix_learn = 1 / args.agents * np.ones_like(combination_matrix)
else:
    combination_matrix_learn = utils.random_combination_matrix_init(args.agents)

log_prev = network.get_log_beliefs(-1, args.state_0, args.state_1, args.multistate)
error = []
c_list = [combination_matrix_learn.reshape(-1)]

state_adaptive = utils.state_estimate_psi(network.intermediate_belief_history[-1])
states_adaptive = [state_adaptive]
state_true_rate = [1. if state_adaptive == args.state_true else 0.]
state_true_beliefs_ag1 = [initial_belief[args.state_true, 0]]
KL_divergence = network.get_log_likelihood_expectation(
    args.state_0, args.state_1, state_adaptive, args.multistate
)
lr = args.lr
if not args.no_verbose:
    iterator = tqdm(range(args.times))
else:
    iterator = range(args.times)
logs = []
loss = []
'''
if want to compare with the known true state case
'''
if args.comparison:
    combination_matrix_learn_comparison = combination_matrix_learn.copy()
    c_list_comparison = [combination_matrix_learn.reshape(-1)]
    loss_comparison = []
    KL_divergence_true = network.get_log_likelihood_expectation(
        args.state_0, args.state_1, args.state_true, args.multistate
    )
    loss_comparison = []
    error_comparison = []
'''
graph learning
'''
for step in iterator:
    if step == args.perturbe_time:
        adj_matrix = utils.create_network(args.agents, args.adjacency_regime, args.er_prob)
        combination_matrix, _, _ = utils.generate_combination_weights(adj_matrix, 0)
        network.A = adj_matrix
        network.C = combination_matrix

    if step > 0 and step < args.times - 1 and args.perturbe>0 and step % args.perturbe == 0:
        adj_matrix = utils.perturbe_network(adj_matrix, 0.005)
        combination_matrix, _, _ = utils.generate_combination_weights(adj_matrix, 0)
        network.A = adj_matrix
        network.C = combination_matrix

    if step == args.change_true_state:
        new_state = utils.get_new_state(args.state_true, args.states)
        args.state_true = new_state
        generator = utils.Generator(likelihood, args.state_true, 0)
        network.state_true = new_state
        network.generator = generator
        KL_divergence_true = network.get_log_likelihood_expectation(
            args.state_0, args.state_1, args.state_true, args.multistate
        )
    network.step()
    state_adaptive = utils.state_estimate_psi(network.intermediate_belief_history[-1])
    if state_adaptive != states_adaptive[-1]:
        KL_divergence = network.get_log_likelihood_expectation(
            args.state_0, args.state_1, state_adaptive, args.multistate
        )
        #mean = KL_divergence.mean()
        #KL_divergence[:] = 0
    log_cur = network.get_log_beliefs(-1, args.state_0, args.state_1, args.multistate)
    combination_matrix_learn = optimization.optimization_step(
        log_cur, log_prev, combination_matrix_learn, KL_divergence, lr=lr,
        projection=args.projection, step_size=args.step_size, alpha=args.alpha, multistate=args.multistate
    )
    if args.comparison:
        combination_matrix_learn_comparison = optimization.optimization_step(
            log_cur, log_prev, combination_matrix_learn_comparison, KL_divergence_true, lr=lr,
            projection=args.projection, step_size=args.step_size, alpha=args.alpha, multistate=args.multistate
        )
        c_list_comparison.append(combination_matrix_learn_comparison)
        loss_comparison.append(optimization.get_loss(combination_matrix_learn_comparison, log_prev, log_cur,
                                                     KL_divergence_true, args.step_size, args.multistate))
        error_comparison.append(utils.estimation_error(combination_matrix_learn_comparison, combination_matrix))
    c_list.append(combination_matrix_learn.reshape(-1))
    error.append(utils.estimation_error(combination_matrix_learn, combination_matrix))
    states_adaptive.append(state_adaptive)
    state_true_rate.append(1. if state_adaptive == args.state_true else 0.)
    state_true_beliefs_ag1.append(network.belief_history[-1][args.state_true, 0])
    loss.append(optimization.get_loss(combination_matrix_learn, log_prev, log_cur, KL_divergence,
                                      args.step_size, args.multistate))

    log_prev = log_cur
    logs.append(log_cur)
    lr *= args.lr_decay

    #np.save('figs/paper/error_perturbe_repeat.npy', np.array(error))
    #np.save('figs/paper/error.npy', np.array(error))
    #np.save('figs/paper/error_true.npy', np.array(error_comparison))

'''
plots
'''
if plot_agents:
    if args.state_1 == args.state_true:
        plotter.plot_agents(args.agents, args.times, network.belief_history, args.state_true, args.state_0)
    else:
        plotter.plot_agents(args.agents, args.times, network.belief_history, args.state_true, args.state_1)

if plot_logs:
    expectation = network.get_log_belief_expectation(1000, args.state_0, args.state_1,
                                                     None, args.multistate)
    plotter.plot_logs(np.array(logs), expectation)

if plot_loss:
    if args.comparison:
        plotter.plot_losses(loss, True, loss_comparison)
    else:
        plotter.plot_losses(loss, True)

if plot_error:
    if args.comparison:
        if args.change_true_state > 0:
            plotter.plot_error(args.times, error, 'linear', None,
                               args.change_true_state, args.perturbe_time, args.perturbe,
                               t1=args.change_true_state - 250, t2=args.change_true_state + 250)
            t_center = 4000
            plotter.plot_error(args.times, error, 'linear', error_comparison,
                               0, args.perturbe_time, args.perturbe,
                               t1=t_center - 100, t2=t_center + 100)
        else:
            t_center = 2000
            plotter.plot_error(args.times, error, 'linear', error_comparison,
                               args.change_true_state, args.perturbe_time, args.perturbe,
                               t1=t_center - 500, t2=t_center + 500)
        # plotter.plot_error(args.times, error, 'log', error_comparison)
    else:
        if args.change_true_state > 0:
            plotter.plot_error(args.times, error, 'linear', None,
                               args.change_true_state, args.perturbe_time, args.perturbe,
                               t1=args.change_true_state - 500, t2=args.change_true_state + 500)
        else:
            t_center = 2000
            plotter.plot_error(args.times, error, 'linear', None,
                               args.change_true_state, args.perturbe_time, args.perturbe,
                               t1=t_center - 500, t2=t_center + 500)
        # plotter.plot_error(args.times, error, 'log')

if plot_state_estimation:
    if args.change_true_state == -1:
        сhange_true_state = 100
    else:
        сhange_true_state = args.change_true_state
    plotter.plot_state(np.array(state_true_rate), сhange_true_state,
                       t1=сhange_true_state - 100, t2=сhange_true_state + 100)
    plotter.plot_belief(np.array(state_true_beliefs_ag1), args.change_true_state,
                        t1=сhange_true_state - 100, t2=сhange_true_state + 100)
    state_true_rate = np.cumsum(np.array(state_true_rate)) / np.arange(1, len(state_true_rate) + 1)
    plotter.plot_state_estimation(state_true_rate)

if plot_evolution:
    plotter.plot_combination_matrix_evolution(args.agents, args.times, combination_matrix, c_list, args.projection, 2)

'''
graph estimation/threshold
'''
try:
    adj_learn = utils.estimate_adjacency(combination_matrix_learn)
    adj_learn_cw = utils.estimate_adjacency_colwise(combination_matrix_learn)

    adj_error = np.linalg.norm(np.abs(np.triu(adj_learn - adj_matrix, 1)).reshape(-1), ord=0)
    adj_error_cw = np.linalg.norm(np.abs(np.triu(adj_learn_cw - adj_matrix, 1)).reshape(-1), ord=0)
    combination_ok = True
except:
    combination_ok = False

if plot_graph and combination_ok:
    #np.random.seed(args.seed)
    #plotter.plot_graph(adj_matrix, adj_learn_cw)
    np.random.seed(args.seed)
    plotter.plot_weighted_graphs(combination_matrix, combination_matrix_learn,
                                 adj_matrix, adj_regime=args.adjacency_regime)

if plot_adjacency and combination_ok:
    plotter.plot_adjacency(combination_matrix, combination_matrix_learn)

'''
results
'''
print('Combination matrix\n', combination_matrix)
print('Combination matrix learned\n', combination_matrix_learn)
print('Error', error[-1])
if combination_ok:
    print('Misclassified', adj_error)
    print('Misclassified colwise', adj_error_cw)
    print('Adj error', adj_error / args.agents ** 2)
    print('Adj error colwise', adj_error_cw / args.agents ** 2)
print('Left stochastic', combination_matrix_learn.sum(0))

if plot_path and args.path:
    distance, path_ = path.get_path(combination_matrix_learn, args.path_0, args.path_1, args.step_size, args.states)
    np.random.seed(args.seed)
    print('The most influencial path from ' + str(path_[-1]) + ' to ' + str(path_[0]))
    print(path_, distance)
    plotter.plot_weighted_graphs_path(combination_matrix_learn, adj_matrix, distance, path_,
                                      args.adjacency_regime)

    #distance, path_ = path.get_most_influencial_node_path(combination_matrix_learn, args.path_0, args.step_size,
    #                                                     args.states)
    #np.random.seed(args.seed)
    #print('The most influencial neighbor of ' + str(path_[0]) + ' is ' + str(path_[-1]))
    #print(path_, distance)
    #plotter.plot_weighted_graphs_path(combination_matrix_learn, adj_matrix, distance, path_)

if plot_explainability:
    distances = path.get_all_influences(combination_matrix, args.path_0, hop=args.hop,
                                        step_size=args.step_size, hypotheses_num=args.states)
    np.random.seed(args.seed)
    plotter.plot_heatmap(combination_matrix, adj_matrix, args.path_0, distances,
                         adj_regime=args.adjacency_regime)