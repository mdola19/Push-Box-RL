from tabular.dp import policy_iteration, policy_evaluation
from tabular.policies import soft_policy, sample_action
from tabular.rollout import run_episode
from envs.grid import GridPushEnv
import numpy as np

mdp = GridPushEnv(layout="open5", slip=0.1, seed=42) 
policy = np.full((mdp.n_states, mdp.n_actions), 1/mdp.n_actions) # start with equal probability for each action in each state
gamma = 0.99 # Problem has dead ends so we should use a value < 1 to ensure convergence
theta = 1e-6

pi_star, V, _, _ = policy_iteration(mdp.P, mdp.R, policy, gamma, theta)
soft_pi = soft_policy(pi_star, noise=0.2)
V_soft, _ = policy_evaluation(mdp.P, mdp.R, soft_pi, gamma, theta)

seed = 24
mdp.reset(seed=seed)
rng = np.random.default_rng(seed + 1)
act = lambda s: sample_action(soft_pi, s, rng)

visits = np.zeros(mdp.n_states, dtype=int)
for _ in range(2000):
    ep = run_episode(mdp, act)
    for s in ep.states:
        visits[s] += 1
mask = visits >= 20
print(f"States visited at least 20 times: {mask.sum()} / {mdp.n_states}")

