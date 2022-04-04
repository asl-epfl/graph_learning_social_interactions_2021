import numpy as np
from utils import kl_divergence

class Network():
    def __init__(self, agents, states, state_true, adjacency_matrix, combination_weights, likelihood, generator,
                 belief_init, step_size=None):
        self.agents = agents
        self.states = states
        self.state_true = state_true
        self.A = adjacency_matrix
        self.C = combination_weights
        self.likelihood = likelihood # agents x states x params
        self.generator = generator
        self.belief_init = belief_init
        self.step_size = step_size
        self._init_history()

    def _init_history(self):
        self.belief_history = [self.belief_init]
        self.intermediate_belief_history = [np.zeros((self.states, self.agents))]
        self.observation_history = [None]

    def step(self):
        sample = self.generator.sample()
        self.observation_history.append(np.copy(sample))

        # combination step (intermediate beliefs)
        denominator = np.zeros(self.agents)
        intermediate_belief = np.zeros((self.states, self.agents))
        for state in range(self.states):
            likelihood = self.likelihood[:, state, :][np.arange(self.agents), sample]
            if self.step_size:
                intermediate_belief[state] = np.power(likelihood, self.step_size) *\
                                             np.power(self.belief_history[-1][state], 1 - self.step_size)
            else:
                intermediate_belief[state] = likelihood * self.belief_history[-1][state]
            denominator += intermediate_belief[state]
        intermediate_belief /= denominator[None, :]
        self.intermediate_belief_history.append(intermediate_belief)

        # adaptation step
        denominator = np.zeros(self.agents)
        belief = np.zeros((self.states, self.agents))
        for state in range(self.states):
            belief[state] = np.exp(self.C.T @ np.log(self.intermediate_belief_history[-1][state, :]))
            denominator += belief[state]
        belief /= denominator[None, :]
        self.belief_history.append(belief)

    def get_log_beliefs(self, time, state_0=None, state_1=None, multistate=False):
        if not multistate:
            log = np.log(
                self.intermediate_belief_history[time][state_0, :] / \
                self.intermediate_belief_history[time][state_1, :]
            )
            log = log.reshape(-1, 1)
        else:
            log = np.array([np.log(
                self.intermediate_belief_history[time][0, :] / \
                self.intermediate_belief_history[time][n, :]) for n in range(1, self.states)
            ]).T
        return log

    def get_log_belief_expectation(self, time, state_0, state_1, combination_matrix=None, multistate=False):
        kl = self.get_log_likelihood_expectation(state_0, state_1, self.state_true, multistate)
        if combination_matrix is None:
            combination_matrix = self.C
        if self.step_size:
            combination_matrix = (1-self.step_size)*combination_matrix
            kl = self.step_size * kl
        if time == 1:
            return np.zeros_like(kl)

        res = np.zeros_like(kl)
        comb_pow = np.eye(combination_matrix.shape[0], combination_matrix.shape[1])
        for i in range(1, time):
            res += comb_pow.T @ kl
            comb_pow = combination_matrix @ comb_pow
        return res

    def get_log_belief_r0(self, time, state_0, state_1, combination_matrix=None, multistate=False):
        if combination_matrix is None:
            combination_matrix = self.C
        kl = self.get_log_likelihood_expectation(state_0, state_1, self.state_true, multistate)
        if self.step_size:
            combination_matrix = (1-self.step_size)*combination_matrix
            kl = self.step_size * kl
        log_exp = self.get_log_belief_expectation(time, state_0, state_1, None, multistate)

        Q = combination_matrix.T @ log_exp @ kl.T + kl @ log_exp.T @ combination_matrix + kl @ kl.T
        comb_pow = np.eye(combination_matrix.shape[0], combination_matrix.shape[1])
        R0 = np.zeros_like(Q)
        for t in range(time):
            R0 += comb_pow @ Q @ comb_pow.T
            comb_pow = combination_matrix @ comb_pow
        return R0

    def get_log_likelihood_expectation(self, state_0=None, state_1=None, state_true=None, multistate=False):
        if not multistate:
            div = kl_divergence(self.likelihood, state_true, state_1, option=0) - \
                  kl_divergence(self.likelihood, state_true, state_0, option=0)
            div = div.reshape(-1, 1)
        else:
            div = np.array([
                kl_divergence(self.likelihood, state_true, n, option=0) - \
                kl_divergence(self.likelihood, state_true, 0, option=0) for n in range(1, self.states)
            ]).T
        return div