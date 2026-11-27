from collections import deque
import numpy as np

class UniformSampler: # equal probability of choosing any state reset

    def __init__(self, n):
        self.n = n
        self.picks = []

    def sample(self, rng):
        i = rng.integers(self.n)
        self.picks.append(i)
        return i

    def update(self, i, success):
        return

class SGSSampler: 

    def __init__(self, n, t, kappa, T, K, eps):
        self.n = n # size of reset space
        self.t = t
        self.T = T
        self.kappa = kappa
        self.K = K # window size for success rate sliding window
        self.eps = eps # epsilon
        self.buffers = [deque(maxlen=K) for _ in range(n)]
        self.sums = np.zeros(n)
        self.counts = np.zeros(n)
        self.picks = []

    def success_rates(self):
        success_rates = np.full((self.n), 0.5)
        success_rates[self.counts > 0] = self.sums[self.counts > 0] / self.counts[self.counts > 0]
        return success_rates

    def sample(self, rng):
        p = self.success_rates()
        w = (p + self.eps)**(self.kappa*self.t) * (1 - p + self.eps)**(self.kappa*(1 - self.t))
        l = np.log(w + self.eps)
        probabilities = np.exp(l/self.T) / np.sum(np.exp(l/self.T))
        i = rng.choice(self.n, p=probabilities)
        self.picks.append(i)
        return i

    def update(self, i, success):
        if len(self.buffers[i]) == self.K:
            self.sums[i] -= self.buffers[i][0]
        else:
            self.counts[i] += 1

        self.sums[i] += success
        self.buffers[i].append(success)

    