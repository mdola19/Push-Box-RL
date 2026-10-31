"""
Dynamic Programming (DP) algorithms for tabular reinforcement learning.
"""

import numpy as np

def flatten_mdp(P, R, policy):
    #P_pi = np.zeros((P.shape[0], P.shape[2]))  # transition matrix under policy pi
    # for s in range(mdp.n_states):
    #     for a in range(mdp.n_actions):
    #         for s_next in range(mdp.n_states):
    #             P_pi[s, s_next] += policy[s, a] * mdp.P[s, a, s_next] 
    P_pi = np.sum(policy[:, :, None] * P, axis=1)  # vectorized version of the above loop

    # R_pi = np.zeros(R.shape[0]) # expected reward under policy pi
    # for s in range(mdp.n_states):
    #     for a in range(mdp.n_actions):
    #         R_pi[s] += policy[s, a] * mdp.R[s, a]
    R_pi = np.sum(policy * R, axis=1)  # vectorized version of the above loop

    return P_pi, R_pi

def policy_evaluation(P, R, policy, gamma, theta=1e-6, in_place=False):
    P_pi, R_pi = flatten_mdp(P, R, policy)
    V = np.zeros(len(R_pi))  # initialize value function
    error = float("inf")
    errors = []

    while error > theta:
        if in_place:
            error = 0
            for s in range(len(R_pi)):
                V_old = V[s]
                V[s] = R_pi[s] + gamma * np.dot(P_pi[s], V)
                error = max(error, abs(V[s] - V_old))

        else:
            V_new = R_pi + gamma * P_pi @ V
            error = np.max(np.abs(V_new - V))
            V = V_new

        errors.append(error)

    return V, errors

def q_from_v(P, R, V, gamma):
    return R + gamma * P @ V

def greedy_policy(Q, policy):
    pi = np.zeros_like(policy)
    for s in range(policy.shape[0]):
        best_action = np.argmax(Q[s])
        pi[s, best_action] = 1.0
    return pi

def policy_iteration(P, R, policy, gamma, theta, in_place=False):
    history = []
    while True:
        V, errors = policy_evaluation(P, R, policy, gamma, theta, in_place)
        Q = np.round(q_from_v(P, R, V, gamma), 4) # rounding helps values tie and forces argmax to stick to picking first item and avoiding switching
        new_policy = greedy_policy(Q, policy)
        history.append((policy, V, Q, errors))
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy.copy()

    return new_policy, V, Q, history

def value_iteration(P, R, gamma, theta, in_place=False):
    V = np.zeros(R.shape[0])
    V_new = np.zeros(R.shape[0])
    history = []
    
    while True:
        max_error = 0
        if in_place:
            for s in range(R.shape[0]):
                V_old = V[s]
                V[s] = np.max(R[s] + gamma * P[s] @ V)
                error = abs(V[s] - V_old) 
                max_error = error if error > max_error else max_error
        else:
            V_new = np.max(q_from_v(P, R, V, gamma), axis=1)
            max_error = np.max(np.abs(V_new - V))
            V = V_new

        history.append(max_error)
        if max_error < theta:
            break 

    Q = np.round(q_from_v(P, R, V, gamma), 6) # not needed but used for consistency with policy iteration
    new_policy = greedy_policy(Q, np.zeros_like(Q))

    return V, new_policy, history

