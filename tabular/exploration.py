import numpy as np

def argmax_random(q_row, rng): # randomized tie-breaker for argmax
    max_q = np.flatnonzero(q_row == q_row.max())
    return rng.choice(max_q)

def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(Q.shape[1])
    else:
        return argmax_random(Q[s], rng)

# Epsilon Schedules
class ConstantEps: 
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def value(self, step):
        return self.epsilon

class LinearEps: # epsilon decreases linearly from start to end over n_steps
    def __init__(self, start, end, n_steps):
        self.start = start
        self.end = end
        self.n_steps = n_steps

    def value(self, step):
        if step >= self.n_steps:
            return self.end
        else:
            return self.start + (self.end - self.start) * (step/self.n_steps)

class InverseEps: # epsilon decays inversely with step
    def value(self, step):
        return 1/(step + 1)