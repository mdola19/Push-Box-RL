import numpy as np
import matplotlib.pyplot as plt
from tabular.control import q_learning
from tabular.dp import value_iteration, evaluate_greedy
from tabular.exploration import LinearEps
from envs.grid import GridPushEnv
from common.samplers import UniformSampler, SGSSampler
from common.plots import plot_route

# Environment Setup
layout = "zigzag7"
env = GridPushEnv(layout=layout, slip=0.1, seed=42, reward="sparse")
gamma = 0.99
alpha = 0.2
n_episodes = 50000
eval_every = 250
seeds = [0, 1, 2, 3, 4]

def record(i, Q, ep_return):
    lengths.append(env.t)
    if (i + 1) % eval_every == 0:
        values.append(evaluate_greedy(env.P, env.R, Q, gamma)[s0])

def first_step_above(steps, values, threshold):
    for k in range(len(values)):
        if values[k] >= threshold:
            return int(steps[k])
    return None

# Baseline: Optimal Policy from DP
state, _ = env.reset()
s0 = env.state_index[state]
V_star, _, _ = value_iteration(env.P, env.R, gamma, 1e-6, True)
starts = [s for s, terminal in zip(env.states, env.terminal) if not terminal]
blocked = np.array([V_star[env.state_index[s]] < 1e-6 for s in starts])

print(f"V* at start = {V_star[s0]}")
print(f"Number of starts = {len(starts)} ({blocked.mean():.0%} blocked)")

Q_star = env.R + gamma * env.P @ V_star
route_env = GridPushEnv(layout=layout, slip=0.0, seed=0, reward="sparse")
state, _ = route_env.reset()
states = [state]
terminated = False
truncated = False
while not terminated and not truncated:
    a = int(np.argmax(Q_star[route_env.state_index[state]]))
    state, _, terminated, truncated, _ = route_env.step(a)
    states.append(state)
print(f"Optimal route length (no slip) = {len(states) - 1} steps")

fig, ax = plt.subplots()
plot_route(route_env, states, ax)
ax.set_title(f"Optimal Policy Route from DP ({layout})")
fig.tight_layout()
fig.savefig(f"experiments/sgs/results/{layout}_optimal_route.png")

# Model-free training runs using both samplers
uniform_steps = []
uniform_values = []
uniform_picks = []
sgs_steps = []
sgs_values = []
sgs_picks = []
for seed in seeds:
    env.reset(seed=42 + seed)
    rng = np.random.default_rng(seed)
    lengths = []
    values = []
    sampler = UniformSampler(len(starts))
    q_learning(env, n_episodes, alpha, gamma, LinearEps(1.0, 0.05, 500), rng, record, sampler=sampler, starts=starts, alpha_decay=10000)
    uniform_steps.append(np.cumsum(lengths)[eval_every - 1::eval_every]) # cumulative steps at every evaluation episode
    uniform_values.append(values)
    uniform_picks.append(sampler.picks)

    env.reset(seed=42 + seed)
    rng = np.random.default_rng(seed)
    lengths = []
    values = []
    sampler = SGSSampler(len(starts), 0.66, 5, 2, 15, 1e-8)
    q_learning(env, n_episodes, alpha, gamma, LinearEps(1.0, 0.05, 500), rng, record, sampler=sampler, starts=starts, alpha_decay=10000)
    sgs_steps.append(np.cumsum(lengths)[eval_every - 1::eval_every])
    sgs_values.append(values)
    sgs_picks.append(sampler.picks)

# Convergence Study
print("\nseed | reach 95% of V* (uniform, sgs)")
uniform_converged = []
sgs_converged = []
for k in range(len(seeds)):
    u_conv = first_step_above(uniform_steps[k], uniform_values[k], 0.95 * V_star[s0])
    g_conv = first_step_above(sgs_steps[k], sgs_values[k], 0.95 * V_star[s0])
    print(f"{seeds[k]} | {u_conv}, {g_conv}")
    if u_conv is not None and g_conv is not None:
        uniform_converged.append(u_conv)
        sgs_converged.append(g_conv)

if len(uniform_converged) > 0:
    print(f"Reach 95% of V*, averaged over {len(uniform_converged)} seeds where both reached it:")
    print(f"uniform = {np.mean(uniform_converged)}, sgs = {np.mean(sgs_converged)}, difference = {(np.mean(sgs_converged) - np.mean(uniform_converged)) / np.mean(uniform_converged):+.1%}")
else:
    print("Reach 95% of V*: no seed where both samplers reached it")

# Resets where samplers picked blocked states
uniform_blocked = []
sgs_blocked = []
for k in range(len(seeds)):
    uniform_blocked.append(np.mean(blocked[uniform_picks[k]]))
    sgs_blocked.append(np.mean(blocked[sgs_picks[k]]))

print(f"Blocked picks: uniform = {np.mean(uniform_blocked):.1%}, sgs = {np.mean(sgs_blocked):.1%}")
print(f"difference = {np.mean(uniform_blocked) - np.mean(sgs_blocked):.1%} fewer blocked picks ({(np.mean(sgs_blocked) - np.mean(uniform_blocked)) / np.mean(uniform_blocked):+.1%} relative)")

# Learning Curve
budget = int(min([steps[-1] for steps in uniform_steps + sgs_steps])) # stopping everything at the shortest run
grid = np.linspace(0, budget, 200)

fig, ax = plt.subplots()
for name, method_steps, method_values in [("uniform", uniform_steps, uniform_values), ("sgs", sgs_steps, sgs_values)]:
    curves = []
    for steps, values in zip(method_steps, method_values):
        curves.append(np.interp(grid, steps, values))
    ax.plot(grid, np.mean(curves, axis=0), label=name)

ax.axhline(V_star[s0], color="black", linestyle="--", label="V*")
ax.set_title(f"Greedy Policy Value at Start vs Env Steps ({layout}, mean over {len(seeds)} seeds)")
ax.set_xlabel("Env Steps")
ax.set_ylabel("V(start)")
ax.legend()
fig.tight_layout()
fig.savefig(f"experiments/sgs/results/{layout}_learning_curve.png")

# Curriculum: where the box was placed at reset (solvable starts only)
chunks = [(0, 2000), (2000, 10000), (10000, 50000)]

heatmaps = []
for method_picks in [uniform_picks, sgs_picks]:
    for first, last in chunks:
        counts = np.zeros(env.walls.shape)
        for picks in method_picks:
            for i in picks[first:last]:
                if not blocked[i]:
                    counts[starts[i][2], starts[i][3]] += 1
        counts = counts / counts.sum()
        counts[env.walls] = np.nan
        heatmaps.append(counts)
vmax = np.nanmax(heatmaps)

fig, axs = plt.subplots(2, 3, figsize=(13, 8))
k = 0
for row, name in enumerate(["uniform", "sgs"]):
    for c, (first, last) in enumerate(chunks):
        ax = axs[row, c]
        image = ax.imshow(heatmaps[k], cmap="viridis", vmin=0, vmax=vmax)
        k += 1
        ax.scatter(env.goal_pos[1], env.goal_pos[0], marker="*", s=200, color="red", label="Goal")
        ax.scatter(env.box_pos[1], env.box_pos[0], marker="s", s=120, color="tab:orange", edgecolors="black", label="Box Start")
        ax.set_title(f"{name}, episodes {first} to {last}")
        ax.set_xticks([])
        ax.set_yticks([])

fig.colorbar(image, ax=axs, fraction=0.03, pad=0.04, label="Fraction of Solvable Picks")
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.suptitle(f"Box Position at Reset, Solvable Starts Only ({layout}, {len(seeds)} seeds)")
fig.savefig(f"experiments/sgs/results/{layout}_curriculum.png")
