import numpy as np
import matplotlib.pyplot as plt
from tabular.control import sarsa, sarsa_lambda, nstep_sarsa, monte_carlo, q_learning
from tabular.dp import evaluate_greedy, value_iteration
from tabular.exploration import LinearEps
from envs.grid import GridPushEnv

envs = [GridPushEnv(layout="open5", slip=0.1, seed=42), GridPushEnv(layout="rooms7", slip=0.1, seed=42)]
layouts = ["open5", "rooms7"]
n_episodes = 6000
gamma = 0.99
alpha = 0.2
seeds = [0, 1, 2, 3, 4]

def record(i, Q, ep_return):
    returns.append(ep_return)
    if (i + 1) % 100 == 0:
        values.append(evaluate_greedy(env.P, env.R, Q, gamma)[s0])

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
value_fig, value_axs = plt.subplots(1, 2, figsize=(14, 5))
for env, layout, ax, value_ax in zip(envs, layouts, axs, value_axs):
    state, _ = env.reset()
    s0 = env.state_index[state]
    V_star, _, _ = value_iteration(env.P, env.R, gamma, 1e-8)

    mc_returns = []
    mc_values = []
    sarsa_returns = []
    sarsa_values = []
    nstep_returns = []
    nstep_values = []
    lambda_returns = []
    lambda_values = []
    q_learning_returns = []
    q_learning_values = []
    for seed in seeds:
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        returns = []
        values = []
        monte_carlo(env, n_episodes, gamma, LinearEps(1.0, 0.05, 500), rng, callback=record)
        mc_returns.append(returns)
        mc_values.append(values)

        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        returns = []
        values = []
        sarsa(env, n_episodes, alpha, gamma, LinearEps(1.0, 0.05, 500), rng, callback=record)
        sarsa_returns.append(returns)
        sarsa_values.append(values)

        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        returns = []
        values = []
        nstep_sarsa(env, n_episodes, alpha, gamma, LinearEps(1.0, 0.05, 500), rng, n=4, callback=record)
        nstep_returns.append(returns)
        nstep_values.append(values)

        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        returns = []
        values = []
        sarsa_lambda(env, n_episodes, alpha, gamma, 0.8, LinearEps(1.0, 0.05, 500), rng, callback=record)
        lambda_returns.append(returns)
        lambda_values.append(values)

        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        returns = []
        values = []
        q_learning(env, n_episodes, alpha, gamma, LinearEps(1.0, 0.05, 500), rng, callback=record)
        q_learning_returns.append(returns)
        q_learning_values.append(values)

    for name, method_returns in [("GLIE MC", mc_returns), ("SARSA", sarsa_returns), ("4-step SARSA", nstep_returns), ("SARSA(0.8)", lambda_returns), ("Q-learning", q_learning_returns)]:
        ax.plot(range(20, n_episodes + 1, 20), np.mean(method_returns, axis=0).reshape(-1, 20).mean(axis=1), label=name)
    ax.set_title(f"Training Return vs Episodes ({layout}, mean over 5 seeds)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return")
    ax.legend()

    for name, method_values in [("GLIE MC", mc_values), ("SARSA", sarsa_values), ("4-step SARSA", nstep_values), ("SARSA(0.8)", lambda_values), ("Q-learning", q_learning_values)]:
        value_ax.plot(range(100, n_episodes + 1, 100), np.mean(method_values, axis=0), label=name)
    value_ax.axhline(V_star[s0], color="black", linestyle="--", label="V*")
    value_ax.set_title(f"Greedy Policy Value at Start vs Episodes ({layout}, mean over 5 seeds)")
    value_ax.set_xlabel("Episodes")
    value_ax.set_ylabel("V(start)")
    value_ax.legend()

fig.tight_layout()
fig.savefig("experiments/model_free_control/results/returns.png")
value_fig.tight_layout()
value_fig.savefig("experiments/model_free_control/results/greedy_value.png")
