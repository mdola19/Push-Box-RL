from tabular.prediction import batch_mc, batch_td_0, certainty_equivalence
from tabular.rollout import run_episode
from tabular.policies import sample_action
from common.stats import bootstrap_ci
from experiments.model_free_prediction.scripts.rollouts_test import mdp, gamma, soft_pi, V_soft, mask
import numpy as np
import matplotlib.pyplot as plt

n_batch = 50
seeds = [0, 1, 2, 3, 4]
act = lambda s: sample_action(soft_pi, s, rng)

mc_errors = []
td_errors = []
ce_errors = []
for seed in seeds:
    mdp.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    episodes = [run_episode(mdp, act) for _ in range(n_batch)]

    visited = np.zeros(mdp.n_states, dtype=bool)
    for ep in episodes:
        visited[ep.states] = True
    m = mask & (V_soft > -99) & visited

    V_mc = batch_mc(episodes, mdp.n_states, gamma)
    V_td = batch_td_0(episodes, mdp.n_states, gamma)
    V_ce = certainty_equivalence(episodes, mdp.n_states, mdp.n_actions, gamma)

    mc_errors.append(np.sqrt(np.mean((V_mc[m] - V_soft[m]) ** 2)))
    td_errors.append(np.sqrt(np.mean((V_td[m] - V_soft[m]) ** 2)))
    ce_errors.append(np.sqrt(np.mean((V_ce[m] - V_soft[m]) ** 2)))
    print(f"seed {seed}: batch TD vs CE max difference = {np.max(np.abs(V_td - V_ce)):.1e}")

names = ["Batch MC", "Batch TD", "Certainty equivalence"]
means = []
lows = []
highs = []
for errs in [mc_errors, td_errors, ce_errors]:
    errs = np.array(errs)
    low, high = bootstrap_ci(errs)
    means.append(errs.mean())
    lows.append(errs.mean() - low)
    highs.append(high - errs.mean())

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(names, means, yerr=[lows, highs], capsize=8, color=["tab:blue", "tab:orange", "tab:green"])
for i, mean in enumerate(means):
    ax.text(i, mean / 2, f"{mean:.2f}", ha="center", color="white")
ax.set_ylabel("RMS Error")
ax.set_title(f"Batch Methods on {n_batch} Episodes (mean and 95% CI over 5 seeds)")
fig.tight_layout()
fig.savefig("experiments/model_free_prediction/results/batch_comparison.png")
