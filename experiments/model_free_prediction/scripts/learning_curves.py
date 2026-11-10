from tabular.prediction import monte_carlo, td_0, backward_td_lambda
from tabular.policies import sample_action
from common.stats import bootstrap_ci
from experiments.model_free_prediction.scripts.rollouts_test import mdp, gamma, soft_pi, V_soft, mask
import numpy as np
import matplotlib.pyplot as plt

n_episodes = 1000
seeds = [0, 1, 2, 3, 4]
m = mask & (V_soft > -99) # for computational efficiency, only compute RMS error for states that are visited at least 20 times and have a finite value in the soft policy evaluation

def record(i, V_est): # callback function to record RMS error at every 10 episodes
    if (i + 1) % 10 == 0:
        errors.append(np.sqrt(np.mean((V_est[m] - V_soft[m]) ** 2)))

mc_errors = []
td_errors = []
lambda_errors = []
for seed in seeds:
    mdp.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    act = lambda s: sample_action(soft_pi, s, rng)

    errors = []
    monte_carlo(mdp, act, n_episodes, alpha=0.05, gamma=gamma, callback=record)
    mc_errors.append(errors)

    errors = []
    td_0(mdp, act, n_episodes, alpha=0.1, gamma=gamma, callback=record)
    td_errors.append(errors)

    errors = []
    backward_td_lambda(mdp, act, n_episodes, alpha=0.1, gamma=gamma, Lambda=0.8, callback=record)
    lambda_errors.append(errors)

episodes_axis = range(10, n_episodes + 1, 10)
fig, ax = plt.subplots(figsize=(8, 5))
for name, errs in [("MC", mc_errors), ("TD(0)", td_errors), ("TD(0.8)", lambda_errors)]:
    errs = np.array(errs)
    low, high = bootstrap_ci(errs)
    ax.plot(episodes_axis, errs.mean(axis=0), label=name)
    ax.fill_between(episodes_axis, low, high, alpha=0.2)
ax.set_title("V(s) RMS Error vs Episodes (mean and 95% CI over 5 seeds)")
ax.set_xlabel("Episodes")
ax.set_ylabel("RMS Error")
ax.legend()
fig.tight_layout()
fig.savefig("experiments/model_free_prediction/results/learning_curves.png")


alphas = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
mc_final = []
td_final = []
for seed in seeds:
    mc_row = []
    td_row = []
    for alpha in alphas:
        mdp.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        act = lambda s: sample_action(soft_pi, s, rng)
        V_mc, _ = monte_carlo(mdp, act, 500, alpha=alpha, gamma=gamma)
        mc_row.append(np.sqrt(np.mean((V_mc[m] - V_soft[m]) ** 2)))

        mdp.reset(seed=seed)
        rng = np.random.default_rng(seed + 1) # reset the rng to ensure that the same actions are sampled as in the MC case
        V_td, _ = td_0(mdp, act, 500, alpha=alpha, gamma=gamma)
        td_row.append(np.sqrt(np.mean((V_td[m] - V_soft[m]) ** 2)))
    mc_final.append(mc_row)
    td_final.append(td_row)

fig, ax = plt.subplots(figsize=(8, 5))
for name, errs in [("MC", mc_final), ("TD(0)", td_final)]:
    errs = np.array(errs)
    low, high = bootstrap_ci(errs)
    ax.plot(alphas, errs.mean(axis=0), marker="o", label=name)
    ax.fill_between(alphas, low, high, alpha=0.2)
ax.set_title("V(s) RMS Error after 500 Episodes vs Step Size (mean and 95% CI over 5 seeds)")
ax.set_xlabel("Step size α")
ax.set_ylabel("RMS Error")
ax.legend()
fig.tight_layout()
fig.savefig("experiments/model_free_prediction/results/step_size.png")
