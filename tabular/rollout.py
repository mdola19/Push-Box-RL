"""
Rollout utilities for tabular reinforcement learning
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Episode:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False

def run_episode(env, act):
    ep = Episode()
    state, _= env.reset()
    while not ep.terminated and not ep.truncated:
        s = env.state_index[state]
        a = act(s)
        state, reward, ep.terminated, ep.truncated, _ = env.step(a)
        ep.states.append(s)
        ep.actions.append(a)
        ep.rewards.append(reward)
    return ep

def discounted_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=float)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns
