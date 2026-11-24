import numpy as np
import matplotlib.pyplot as plt
from tabular.control import sarsa, q_learning
from tabular.dp import evaluate_greedy, value_iteration
from tabular.exploration import LinearEps, argmax_random
from envs.grid import GridPushEnv
from common.plots import plot_route

env = GridPushEnv(layout="cliff", slip=0, seed=42)
n_episodes = 30000
gamma = 0.99
alpha = 0.2
seeds = [0, 1, 2, 3, 4]
eval_every = 100

state, _ = env.reset()
s0 = env.state_index[state]
V_star, _, _ = value_iteration(env.P, env.R, gamma, 1e-8)

sarsa_returns = []
sarsa_values = []
q_learning_returns = []
q_learning_values = []

def record(i, Q, ep_return):
    returns.append(ep_return)
    if (i + 1) % eval_every == 0:
        values.append(evaluate_greedy(env.P, env.R, Q, gamma)[s0])

for seed in seeds:
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    returns = []
    values = []
    Q_sarsa = sarsa(env, n_episodes, alpha, gamma, LinearEps(1.0, 0, 20000), rng, callback=record)
    sarsa_returns.append(returns)
    sarsa_values.append(values)

    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    returns = []
    values = []
    Q_q = q_learning(env, n_episodes, alpha, gamma, LinearEps(1.0, 0, 20000), rng, callback=record)
    q_learning_returns.append(returns)
    q_learning_values.append(values)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for name, method_returns in [("SARSA", sarsa_returns), ("Q-learning", q_learning_returns)]:
    axs[0].plot(range(20, n_episodes + 1, 20), np.mean(method_returns, axis=0).reshape(-1, 20).mean(axis=1), linewidth=1, label=name)
axs[0].set_title("Training Return vs Episodes (cliff, ε → 0, mean over 5 seeds)")
axs[0].set_xlabel("Episodes")
axs[0].set_ylabel("Return")
axs[0].legend()

for name, method_values in [("SARSA", sarsa_values), ("Q-learning", q_learning_values)]:
    axs[1].plot(range(eval_every, n_episodes + 1, eval_every), np.mean(method_values, axis=0), linewidth=1, label=name)
axs[1].axhline(V_star[s0], color="black", linestyle="--", label="V*")
axs[1].set_title("Greedy Policy Value at Start vs Episodes (cliff, mean over 5 seeds)")
axs[1].set_xlabel("Episodes")
axs[1].set_ylabel("V(start)")
axs[1].legend()
fig.tight_layout()
fig.savefig("experiments/model_free_control/results/cliff_curves.png")

route_env = GridPushEnv(layout="cliff", slip=0.0, seed=0)
route_rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
for name, Q, ax in [("SARSA", Q_sarsa, axs[0]), ("Q-learning", Q_q, axs[1])]:
    state, _ = route_env.reset()
    states = [state]
    terminated = False
    truncated = False
    while not terminated and not truncated:
        a = int(argmax_random(Q[route_env.state_index[state]], route_rng))
        state, _, terminated, truncated, _ = route_env.step(a)
        states.append(state)
    plot_route(route_env, states, ax)
    ax.set_title(f"{name} Optimal Greedy Route")
fig.tight_layout()
fig.savefig("experiments/model_free_control/results/cliff_routes.png")

