import numpy as np


class HMM:

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        alphas = []
        for i in range(self.N):
            alphas.append([0]*len(seq))

        t = 0
        for i in self.states:
            p = self.pi[self.states_dict[i]] * \
                self.B[self.states_dict[i]][self.emissions_dict[seq[t]]]
            alphas[self.states_dict[i]][t] = p

        alphas = np.array(alphas)
        ind = np.where(alphas[:, 0] == max(alphas[:, 0]))
        optiseq = []
        optiseq.append(self.states[ind[0][0]])

        for t in range(1, len(seq)):
            for i in range(self.N):
                w = alphas[:, t-1] * self.A[:, i] * \
                    ([self.B[i][self.emissions_dict[seq[0]]]]*self.N)
                alphas[i][t] = max(w)
            ind = np.where(alphas[:, t] == max(alphas[:, t]))
            optiseq.append(self.states[ind[0][0]])
        return optiseq
