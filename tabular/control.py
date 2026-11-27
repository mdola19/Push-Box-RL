import numpy as np
from collections import deque
from tabular.rollout import run_episode
from tabular.exploration import epsilon_greedy
from tabular.rollout import discounted_returns

def monte_carlo(env, n_episodes, gamma, eps_schedule, rng, callback=None):
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions))

    for i in range(n_episodes):
        epsilon = eps_schedule.value(i)
        act = lambda s: epsilon_greedy(Q, s, epsilon, rng) 
        ep = run_episode(env, act)
        ep_returns = discounted_returns(ep.rewards, gamma)
        for t, (s, a) in enumerate(zip(ep.states, ep.actions)):
            N[s, a] += 1
            Q[s, a] += (1/N[s,a])*(ep_returns[t] - Q[s,a])
        epsilon = 1/(i+1)

        if callback:
            callback(i, Q, sum(ep.rewards))

    return Q

def sarsa(env, n_episodes, alpha, gamma, eps_schedule, rng, callback=None):
    Q = np.zeros((env.n_states, env.n_actions))

    for i in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_return = 0
        epsilon = eps_schedule.value(i)
        s = env.state_index[state]
        a = epsilon_greedy(Q, s, epsilon, rng)

        while not terminated and not truncated:
            state, reward, terminated, truncated, _ = env.step(a)
            ep_return += reward
            next_s = env.state_index[state]
            next_a = epsilon_greedy(Q, next_s, epsilon, rng)
            Q[s, a] += alpha*(reward + gamma*Q[next_s, next_a] - Q[s,a])
            s = next_s
            a = next_a

        if callback:
            callback(i, Q, ep_return)

    return Q

def nstep_sarsa(env, n_episodes, alpha, gamma, eps_schedule, rng, n, callback=None):
    Q = np.zeros((env.n_states, env.n_actions))

    for i in range(n_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_return = 0
        epsilon = eps_schedule.value(i)
        s = env.state_index[state]
        a = epsilon_greedy(Q, s, epsilon, rng)
        queue = deque()

        while not terminated and not truncated:
            state, reward, terminated, truncated, _ = env.step(a)
            ep_return += reward
            next_s = env.state_index[state]
            next_a = epsilon_greedy(Q, next_s, epsilon, rng)
            queue.append((s, a, reward))
            while len(queue) == n or ((terminated or truncated) and len(queue) > 0):
                G = 0
                for t in range(len(queue)):
                    G += gamma**t * queue[t][2]
                if len(queue) == n:
                    G += gamma**n * Q[next_s, next_a]
                current_s, current_a, _ = queue.popleft()
                Q[current_s, current_a] += alpha * (G - Q[current_s, current_a])
            s = next_s
            a = next_a
    
        if callback is not None:
            callback(i, Q, ep_return)

    return Q

def sarsa_lambda(env, n_episodes, alpha, gamma, Lambda, eps_schedule, rng, callback=None):
    Q = np.zeros((env.n_states, env.n_actions))

    for i in range(n_episodes):
        E = np.zeros((env.n_states, env.n_actions))
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_return = 0
        epsilon = eps_schedule.value(i)
        s = env.state_index[state]
        a = epsilon_greedy(Q, s, epsilon, rng)

        while not terminated and not truncated:
            state, reward, terminated, truncated, _ = env.step(a)
            ep_return += reward
            next_s = env.state_index[state]
            next_a = epsilon_greedy(Q, next_s, epsilon, rng)
            E *= gamma * Lambda
            E[s, a] += 1
            delta = reward + gamma * Q[next_s, next_a] - Q[s,a]
            Q += alpha * delta * E 
            s = next_s
            a = next_a
        
        if callback is not None:
            callback(i, Q, ep_return)

    return Q

def q_update(Q, s, a, r, next_s, terminated, alpha, gamma):
    q_row = Q[next_s]
    Q[s,a] += alpha*(r + gamma*max(q_row) * (not terminated) - Q[s,a])
    return Q

def q_learning(env, n_episodes, alpha, gamma, eps_schedule, rng, callback=None, sampler=None, starts=None, alpha_decay=None):
    Q = np.zeros((env.n_states, env.n_actions))
    

    for i in range(n_episodes):
        if sampler is not None and starts is not None:
            start_idx = sampler.sample(rng)
            state, _ = env.reset(start=starts[start_idx])
        else:
            state, _ = env.reset()
        terminated = False
        truncated = False
        ep_return = 0
        epsilon = eps_schedule.value(i)
        step_size = alpha

        while not terminated and not truncated:
            s = env.state_index[state]
            a = epsilon_greedy(Q, s, epsilon, rng)
            state, reward, terminated, truncated, _ = env.step(a)
            ep_return += reward
            next_s = env.state_index[state]
            if alpha_decay is not None:
                step_size = alpha/(1 + i/alpha_decay)
            Q = q_update(Q, s, a, reward, next_s, terminated, step_size, gamma)

        if sampler is not None and starts is not None:
            success = (state[2], state[3]) == env.goal_pos
            sampler.update(start_idx, success)

        if callback:
            callback(i, Q, ep_return)

    return Q

