from envs.grid import GridPushEnv
from tabular.dp import policy_evaluation, policy_iteration, value_iteration
from common.plots import plot_value_heatmap
import numpy as np
import matplotlib.pyplot as plt

mdp = GridPushEnv(layout="open5", slip=0.1, seed=42) 
policy = np.full((mdp.n_states, mdp.n_actions), 1/mdp.n_actions) # start with equal probability for each action in each state
gamma = 0.99 # Problem has dead ends so we should use a value < 1 to ensure convergence
theta = 1e-6


# Environment Behaviour vs Model
rng = np.random.default_rng(0)
non_terminal = np.where(~mdp.walls)[0] # ~ = inverts a boolean array
s_a_pairs = [(rng.choice(non_terminal), rng.integers(mdp.n_actions)) for _ in range(20)]
n_samples = 5000
tvs = [] # total variation distances for each pair
for s, a in s_a_pairs:
    counter = np.zeros(mdp.n_states) # how many times each state occured
    for _ in range(n_samples):
        mdp.state = mdp.states[s]
        next_state = mdp.step(a)[0]
        counter[mdp.state_index[next_state]] += 1
    tvs.append(0.5 * np.abs(counter/n_samples - mdp.P[s,a]).sum())
print(f"Largest Total Variation Distance over {len(s_a_pairs)} (s, a) pairs = {max(tvs):.4f}")

# Values for Random Policy
V, _ = policy_evaluation(mdp.P, mdp.R, policy, gamma, theta, True)
fig, ax = plt.subplots(1,3, figsize=(16, 5))
plot_value_heatmap(mdp, V, (4,1), policy=policy, ax=ax[0])
plot_value_heatmap(mdp, V, (2,2), policy=policy, ax=ax[1])
plot_value_heatmap(mdp, V, (3, 4), policy=policy, ax=ax[2])
fig.tight_layout()
fig.savefig("experiments/results/part1/policy_eval.png")

# Convergence
V_optimal, value_policy, value_history = value_iteration(mdp.P, mdp.R, gamma, theta, False)
optimal_policy, _, _, policy_history = policy_iteration(mdp.P, mdp.R, policy, gamma, theta)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].semilogy(list(range(len(value_history))), value_history)
ax[0].set_title(f'Value Iteration Convergence')
ax[0].set_xlabel('Sweep')
ax[0].set_ylabel('log(Value Error)')
for k, entry in enumerate(policy_history):
    ax[1].semilogy(entry[3], label=f"round {k}")
ax[1].legend()
ax[1].set_xlabel("Sweep")
ax[1].set_title(f'Policy Iteration Convergence')
ax[1].set_ylabel('log(Value Error)')
fig.savefig("experiments/results/part1/convergence_plots.png")

# Optimal Policy
fig, ax = plt.subplots(1,1)
plot_value_heatmap(mdp, V_optimal, mdp.box_pos, policy=value_policy, ax=ax)
fig.savefig("experiments/results/part1/optimal_policy_heatmap.png")

# Sync vs In-Place
_, _, value_history2 = value_iteration(mdp.P, mdp.R, gamma, theta, True)
_, _, _, policy_history2 = policy_iteration(mdp.P, mdp.R, policy, gamma, theta, True)
policy_sync_sweeps = 0
policy_inplace_sweeps = 0
for iter in range(len(policy_history)):
    policy_sync_sweeps += len(policy_history[iter][3])
for iter in range(len(policy_history2)):
    policy_inplace_sweeps += len(policy_history2[iter][3])

print(f"Policy Iteration # of Sweeps: Sync = {policy_sync_sweeps}, In-Place = {policy_inplace_sweeps}")
print(f"Value Iteration # of Sweeps: Sync = {len(value_history)}, In-Place = {len(value_history2)}")

# Unrecoverable States
V, _ = policy_evaluation(mdp.P, mdp.R, optimal_policy, gamma, theta, True)
unrecoverable_states = np.argwhere(np.round(V,4) == np.round(np.min(V), 4))
n = len(unrecoverable_states)
print(f"Value in Unrecoverable State = {np.round(np.min(V), 4)}")
print(f"# of Unrecoverable States = {n}")
cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
for ax, s in zip(axs.flat, unrecoverable_states.flatten()):
    ax.axis("off")
    ar, ac, br, bc = mdp.states[s]
    img = np.full((*mdp.walls.shape, 3), 0.85)
    img[mdp.goal_pos] = [0, 0.7, 0]
    img[br, bc] = [1, 0.85, 0]
    img[ar, ac] = [1, 0, 0]
    ax.imshow(img)
fig.suptitle(f"{n} Unrecoverable States - R = Agent, Y = Box, G = Goal")
fig.tight_layout()
fig.savefig("experiments/results/part1/unrecoverable_states.png")
