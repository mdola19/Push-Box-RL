import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from envs.grid import DELTAS
from common.stats import bootstrap_ci

def plot_value_heatmap(mdp, V, box_pos, policy=None, ax=None):
    """
    Plots a heatmap of the value function V for a fixed box position.
    Optionally overlays the best policy arrows.
    """
    
    if ax is None:
        _, ax = plt.subplots()

    value_grid = np.full(mdp.walls.shape, np.nan)
    for state in mdp.states:
        if (state[2], state[3]) == box_pos: 
            value_grid[state[0], state[1]] = V[mdp.state_index[state]]

    cax = ax.imshow(value_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    if policy is not None:
        for state in mdp.states:
            if (state[2], state[3]) == box_pos:  
                action = np.argmax(policy[mdp.state_index[state]])
                delta = DELTAS[action]
                ax.arrow(state[1], state[0], delta[1]*0.3, delta[0]*0.3, head_width=0.2, head_length=0.2, fc='red', ec='red')

    ax.set_title(f'Value Function Heatmap for Box Position {box_pos}')
    ax.set_xlabel('Agent Column')
    ax.set_ylabel('Agent Row')

    return ax

def plot_curves(runs_by_method, ax=None, x_key="step", y_key="rms_error", n_points=200):
    if ax is None:
        _, ax = plt.subplots()

    for method, runs in runs_by_method.items():
        xs = [np.array([m[x_key] for m in run["metrics"]], dtype=float) for run in runs]
        ys = [np.array([m[y_key] for m in run["metrics"]], dtype=float) for run in runs]

        x_max = min(x[-1] for x in xs)
        grid = np.linspace(0, x_max, n_points)
        curves = np.array([np.interp(grid, x, y) for x, y in zip(xs, ys)])

        mean = curves.mean(axis=0)
        low, high = bootstrap_ci(curves)

        line, = ax.plot(grid, mean, label=f"{method} (n={len(runs)})")
        ax.fill_between(grid, low, high, color=line.get_color(), alpha=0.2)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend()
    return ax

def plot_route(env, states, ax):
    if ax is None:
        _, ax = plt.subplots()

    grid = np.zeros(env.walls.shape)
    grid[env.walls] = 1
    grid[env.pits] = 2
    grid[env.goal_pos] = 3
    ax.imshow(grid, cmap=ListedColormap(["white", "black", "red", "green"]), vmin=0, vmax=3)

    agent_rows = [s[0] for s in states]
    agent_cols = [s[1] for s in states]
    box_rows = [s[2] for s in states]
    box_cols = [s[3] for s in states]

    ax.quiver(agent_cols[:-1], agent_rows[:-1], np.diff(agent_cols), np.diff(agent_rows), angles="xy", scale_units="xy", scale=1, color="tab:blue", label="Agent")
    ax.quiver(box_cols[:-1], box_rows[:-1], np.diff(box_cols), np.diff(box_rows), angles="xy", scale_units="xy", scale=1, color="tab:orange", label="Box")
    ax.scatter(agent_cols[0], agent_rows[0], s=150, marker="o", color="tab:blue", edgecolors="black", zorder=3, label="Agent Start")
    ax.scatter(box_cols[0], box_rows[0], s=150, marker="s", color="tab:orange", edgecolors="black", zorder=3, label="Box Start")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, borderaxespad=0)
    return ax