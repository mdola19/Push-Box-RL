"""
Model-Free Prediction algorithms for tabular reinforcement learning.
"""

import numpy as np
from tabular.rollout import discounted_returns, run_episode
from collections import deque
from tabular.dp import policy_evaluation

def monte_carlo(env, act, n_episodes, alpha, gamma, first_visit=False, running_average=False, callback = None):
    V = np.zeros(env.n_states)
    N = np.zeros(env.n_states)
    n_truncated = 0
    for i in range(n_episodes):
        visited = np.zeros(env.n_states, dtype=bool)
        ep = run_episode(env, act)
        ep_returns = discounted_returns(ep.rewards, gamma)
        if ep.truncated:
            n_truncated += 1
        for t, s in enumerate(ep.states):
            if first_visit and visited[s]:
                continue
            visited[s] = True
            N[s] += 1
            if running_average:
                alpha = 1 / N[s]
            V[s] += alpha * (ep_returns[t] - V[s])
        if callback is not None:
            callback(i, V)
    return V, n_truncated

def td_0(env, act, n_episodes, alpha, gamma, callback=None):
    V = np.zeros(env.n_states)
    n_truncated = 0
    for i in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            s = env.state_index[state]
            a = act(s)
            next_state, reward, terminated, truncated, _ = env.step(a)
            next_s = env.state_index[next_state]
            V[s] += alpha * (reward + gamma * V[next_s] - V[s])
            state = next_state
        if truncated:
            n_truncated += 1
        if callback is not None:
            callback(i, V)
    return V, n_truncated

def nstep_td(env, act, n_episodes, alpha, gamma, n, callback=None):
    V = np.zeros(env.n_states)
    n_truncated = 0
    for i in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        queue = deque()
        while not terminated and not truncated:
            s = env.state_index[state]
            a = act(s)
            next_state, reward, terminated, truncated, _ = env.step(a)
            next_s = env.state_index[next_state]
            queue.append((s, reward))
            while len(queue) == n or ((terminated or truncated) and len(queue) > 0):
                G = 0
                for t in range(len(queue)):
                    G += gamma**t * queue[t][1]
                if len(queue) == n:
                    G += gamma**n * V[next_s]
                current_s, _ = queue.popleft()
                V[current_s] += alpha * (G - V[current_s])
            state = next_state
        if truncated:
            n_truncated += 1
        if callback is not None:
            callback(i, V)
    return V, n_truncated

def forward_td_lambda(env, act, n_episodes, alpha, gamma, Lambda, callback=None):
    V = np.zeros(env.n_states)
    n_truncated = 0
    for i in range(n_episodes):
        ep = run_episode(env, act)
        if ep.truncated:
            n_truncated += 1
        V_old = V.copy()
        T = len(ep.states)
        for t, s in enumerate(ep.states):
            G = 0
            for n in range(1, T-t):
                G_n = 0
                for k in range(n):
                    G_n += gamma**(k) * ep.rewards[t+k]
                G_n += gamma**n * V_old[ep.states[t+n]]
                G += (Lambda**(n-1)) * G_n
            G *= (1-Lambda)
            G_full = 0
            for j in range(T-t):
                G_full += gamma**j * ep.rewards[t+j]
            G += (Lambda**(T-t-1)) * G_full
            V[s] += alpha * (G - V_old[s])
        if callback is not None:
            callback(i, V)
    return V, n_truncated

def backward_td_lambda(env, act, n_episodes, alpha, gamma, Lambda, callback=None):
    V = np.zeros(env.n_states)
    n_truncated = 0
    for i in range(n_episodes):
        E = np.zeros(env.n_states)
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            s = env.state_index[state]
            a = act(s)
            next_state, reward, terminated, truncated, _ = env.step(a)
            next_s = env.state_index[next_state]
            E *= gamma * Lambda
            E[s] += 1
            delta = reward + gamma * V[next_s] - V[s]
            V += alpha * delta * E 
            state = next_state
        if truncated:
            n_truncated += 1
        if callback is not None:
            callback(i, V)
    return V, n_truncated

def batch_mc(episodes, n_states, gamma):
    V = np.zeros(n_states)
    N = np.zeros(n_states)
    for ep in episodes:
        G = discounted_returns(ep.rewards, gamma)
        for t, s in enumerate(ep.states):
            V[s] += G[t]
            N[s] += 1
    V = np.divide(V, N, out=np.zeros(n_states), where=N > 0)
    return V

def batch_td_0(episodes, n_states, gamma, alpha=0.01, tol=1e-10):
    V = np.zeros(n_states)
    while True:
        dV = np.zeros(n_states)
        for ep in episodes:
            for t, s in enumerate(ep.states):
                if t < len(ep.states) - 1:
                    next_s = ep.states[t+1]
                    target = ep.rewards[t] + gamma * V[next_s]
                else:
                    target = ep.rewards[t]
                dV[s] += alpha * (target - V[s])
        V += dV
        if np.abs(dV).max() < tol:
            return V

def certainty_equivalence(episodes, n_states, n_actions, gamma):
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    for ep in episodes:
        for t, s in enumerate(ep.states):
            a = ep.actions[t]
            if t < len(ep.states) - 1:
                next_s = ep.states[t+1]
                P[s, a, next_s] += 1
            R[s, a] += ep.rewards[t]
            N[s, a] += 1
    P = np.divide(P, N[:, :, None], out=np.zeros_like(P), where=N[:, :, None] > 0)
    R = np.divide(R, N, out=np.zeros_like(R), where=N > 0)
    pi_hat = np.divide(N, N.sum(axis=1, keepdims=True), out=np.zeros_like(N), where=N.sum(axis=1, keepdims=True) > 0)
    V, _ = policy_evaluation(P, R, pi_hat, gamma, theta=1e-12)
    return V

