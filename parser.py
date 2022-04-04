import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--agents', default=4, type=int, help='agents number')
    parser.add_argument('--states', default=2, type=int, help='number of states')
    parser.add_argument('--params', default=2, type=int, help='number of parameters for multinomial distribution')
    parser.add_argument('--step_size', default=None, type=float,
                        help='enables adaptive social learning procedure; step size')
    parser.add_argument('--alpha', default=0., type=float,
                        help='regularization parameter')
    parser.add_argument('--adjacency_regime', default=2, type=int,
                        help='1:Random (undirected), 2:Fully connected, 3:Erdos-Renyi, 4:Star, 5:2-Star, 6:Manual')
    parser.add_argument('--er_prob', default=0.5, type=float, help='edge probability for ER case')
    parser.add_argument('--combination_regime', default=0, type=int,
                        help='0:Uniform (left stochastic), 1: Doubly stochastic, 2: Left stochastic (random)')
    parser.add_argument('--likelihood_regime', default=1, type=int,
                        help='0:discrete with close states, 1:discrete, 2:gaussian')
    parser.add_argument('--likelihood_var', default=.1, type=float,
                        help='likelihood variance between the states')

    parser.add_argument('--state_true', default=1, type=int, help='true state')
    parser.add_argument('--state_0', default=0, type=int, help='numerator state in recursion')
    parser.add_argument('--state_1', default=1, type=int, help='denominator state in recursion')
    parser.add_argument('--times', default=500, type=int, help='number of timestamps')
    parser.add_argument('--exp_num', default=1, type=int, help='experiments number for MC estimate')

    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=1, type=float, help='learning rate decay')

    parser.add_argument('--projection', default=False, action='store_true', help='do projection')
    parser.add_argument('--start_fc', default=False, action='store_true',
                        help='start with fc matrix (we start from the random initialization normally)')

    parser.add_argument('--find_state', default=False, action='store_true')
    parser.add_argument('--multistate', default=False, action='store_true',
                        help='use all the states to construct lambda')
    parser.add_argument('--comparison', default=False, action='store_true',
                        help='comparison errors with true theta if adaptive theta')
    parser.add_argument('--change_true_state', default=-1, type=int,
                        help='change true state after change_true_state steps')
    #parser.add_argument('--perturbe_time', default=-1, type=int,
    #                    help='each perturbe_time every adjacency matrix edge changes 0/1 with p=0.05')
    parser.add_argument('--perturbe_time', default=-1, type=int,
                        help='at perturbe_time the adjacency matrix changes completely')
    parser.add_argument('--perturbe', default=-1, type=int,
                        help='each perturbe time every adjacency matrix edge changes 0/1 with p=0.05')

    parser.add_argument('--path', default=False, action='store_true')
    parser.add_argument('--path_0', default=0, type=int, help='node_0 for path search')
    parser.add_argument('--path_1', default=1, type=int, help='node_1 for path search')
    parser.add_argument('--hop', default=3, type=int, help='explainability hop')

    parser.add_argument('--no_verbose', default=False, action='store_true')
    return parser