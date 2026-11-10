import numpy as np

def soft_policy(pi_star, noise=0.2):
    """Turning a deterministic policy into a stochastic one by adding noise."""
    soft_pi = (1 - noise) * pi_star + (noise / pi_star.shape[1])
    return soft_pi

def sample_action(pi, s, rng):
    return rng.choice(len(pi[s]), p=pi[s])

