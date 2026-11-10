from tabular.prediction import nstep_td, forward_td_lambda, backward_td_lambda
from tabular.policies import sample_action
from experiments.model_free_prediction.scripts.rollouts_test import mdp, gamma, soft_pi, V_soft, mask
import numpy as np
import matplotlib.pyplot as plt

n_episodes = 500
seeds = [0, 1, 2, 3, 4]
alphas = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
m = mask & (V_soft > -99)
act = lambda s: sample_action(soft_pi, s, rng)


ns = [1, 2, 4, 8, 16, 32]
n_errors = np.zeros((len(ns), len(alphas)))
for i, n in enumerate(ns):
    for j, alpha in enumerate(alphas):
        for seed in seeds:
            mdp.reset(seed=seed)
            rng = np.random.default_rng(seed + 1)
            V, _ = nstep_td(mdp, act, n_episodes, alpha=alpha, gamma=gamma, n=n)
            n_errors[i, j] += np.sqrt(np.mean((V[m] - V_soft[m]) ** 2))
n_errors /= len(seeds)

fig, ax = plt.subplots(figsize=(9, 5))
img = ax.imshow(n_errors, cmap="viridis", vmax=8)
fig.colorbar(img, ax=ax, label="RMS Error")
for i in range(len(ns)):
    for j in range(len(alphas)):
        ax.text(j, i, f"{n_errors[i, j]:.1f}" if n_errors[i, j] < 100 else ">100", ha="center", va="center", color="white")
ax.set_xticks(range(len(alphas)), alphas)
ax.set_yticks(range(len(ns)), ns)
ax.set_xlabel("Step size α")
ax.set_ylabel("n")
ax.set_title(f"n-step TD: RMS Error after {n_episodes} Episodes (mean over 5 seeds)")
fig.tight_layout()
fig.savefig("experiments/model_free_prediction/results/n_step_heatmap.png")


lambdas = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1]
forward_errors = np.zeros((len(lambdas), len(alphas)))
backward_errors = np.zeros((len(lambdas), len(alphas)))
for i, lam in enumerate(lambdas):
    for j, alpha in enumerate(alphas):
        for seed in seeds:
            mdp.reset(seed=seed)
            rng = np.random.default_rng(seed + 1)
            V, _ = forward_td_lambda(mdp, act, n_episodes, alpha=alpha, gamma=gamma, Lambda=lam)
            forward_errors[i, j] += np.sqrt(np.mean((V[m] - V_soft[m]) ** 2))

            mdp.reset(seed=seed)
            rng = np.random.default_rng(seed + 1)
            V, _ = backward_td_lambda(mdp, act, n_episodes, alpha=alpha, gamma=gamma, Lambda=lam)
            backward_errors[i, j] += np.sqrt(np.mean((V[m] - V_soft[m]) ** 2))
forward_errors /= len(seeds)
backward_errors /= len(seeds)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
for ax, errors, name in [(axs[0], forward_errors, "Forward view"), (axs[1], backward_errors, "Backward view")]:
    img = ax.imshow(errors, cmap="viridis", vmax=8)
    fig.colorbar(img, ax=ax, label="RMS Error")
    for i in range(len(lambdas)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{errors[i, j]:.1f}" if errors[i, j] < 100 else ">100", ha="center", va="center", color="white")
    ax.set_xticks(range(len(alphas)), alphas)
    ax.set_yticks(range(len(lambdas)), lambdas)
    ax.set_xlabel("Step size α")
    ax.set_ylabel("λ")
    ax.set_title(f"TD(λ) {name}: RMS Error after {n_episodes} Episodes")
fig.tight_layout()
fig.savefig("experiments/model_free_prediction/results/lambda_heatmap.png")
