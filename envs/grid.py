from envs.layouts import parse_layout, LAYOUTS
import numpy as np

ACTIONS = [0, 1, 2, 3]  # up, down, left, right
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class GridPushEnv:

    def __init__(self, layout, slip=0.1, seed=None):
        self.walls, self.agent_pos, self.box_pos, self.goal_pos = parse_layout(LAYOUTS[layout])
        self.states, self.terminal = self._enumerate_states() # each state is a tuple (agent_row, agent_col, box_row, box_col)
        self.state_index = {state: idx for idx, state in enumerate(self.states)}
        self.state = (self.agent_pos[0], self.agent_pos[1], self.box_pos[0], self.box_pos[1])  # initial state
        self.slip = slip  # probability of slipping to a random action
        self.P, self.R = self._build_model()  # transition and expectedreward matrices
        self.rng = np.random.default_rng(seed)  # random number generator for stochastic transitions
        self.t = 0

        # Attributes
        self.n_actions = len(ACTIONS)
        self.n_states = len(self.states)

    def _enumerate_states(self):
        states = []
        terminal = []  
        for ar, ac in np.ndindex(self.walls.shape[0], self.walls.shape[1]):
            for br, bc in np.ndindex(self.walls.shape[0], self.walls.shape[1]):
                if not self.walls[ar, ac] and not self.walls[br, bc]:  # both agent and box must be on empty floor
                    if (ar, ac) != (br, bc):
                        states.append((ar, ac, br, bc))  

        for state in states:
            if (state[2], state[3]) == self.goal_pos:  # box position matches goal position
                terminal.append(True)
            else:
                terminal.append(False)

        return states, np.array(terminal)

    def move(self, state, action):
        delta = DELTAS[action]
        new_a = (state[0] + delta[0], state[1] + delta[1])  # new agent position

        if self._blocked(new_a):  # agent hits a wall or goes off-grid
            return state

        else:
            if new_a == (state[2], state[3]):  # agent tries to move into the box
                new_b = (state[2] + delta[0], state[3] + delta[1])  
                if self._blocked(new_b):  # box hits a wall or goes off-grid
                    return state
                else:
                    return (new_a[0], new_a[1], new_b[0], new_b[1])  # box and agent move together
            else:
                return (new_a[0], new_a[1], state[2], state[3])  # only agent moves

    def _blocked(self, pos):
        if pos[0] < 0 or pos[0] >= self.walls.shape[0] or pos[1] < 0 or pos[1] >= self.walls.shape[1]:
            return True
        return self.walls[pos]
    
    def action_probs(self, action):
        probs = np.full(4, self.slip / 3)  
        probs[action] = 1 - self.slip
        return probs

    def _build_model(self):
        P = np.zeros((len(self.states), len(ACTIONS), len(self.states))) # each entry holds prob. of transitioning to S' after taking action A in S
        R = np.zeros((len(self.states), len(ACTIONS))) # each entry holds reward for taking action A in state S
        for s_i, s in enumerate(self.states):
            if not self.terminal[s_i]:
                for a in ACTIONS:
                    for possible_a in range(len(ACTIONS)):
                        possible_s = self.move(s, possible_a)
                        possible_s_i = self.state_index[possible_s]
                        p = self.action_probs(a)[possible_a]
                        P[s_i, a, possible_s_i] += p 

                        if self.terminal[possible_s_i]:
                            R[s_i, a] += 0   
                        else:
                            R[s_i, a] += -1 * p
            else:
                for a in ACTIONS:
                    P[s_i, a, s_i] = 1.0  
                    R[s_i, a] = 0.0  

        return P, R

    def reset(self, seed=None):
        self.t = 0
        info = {"seed": seed}
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.state = (self.agent_pos[0], self.agent_pos[1], self.box_pos[0], self.box_pos[1])
        return (self.state, info)

    def step(self, action):
        a = self.rng.choice(ACTIONS, p=self.action_probs(action))  
        self.state = self.move(self.state, a)
        terminated = bool(self.terminal[self.state_index[self.state]])
        reward = 0 if terminated else -1
        self.t += 1

        if self.t >= 100:
            truncated = True
        else:
            truncated = False
        
        info = {"time_step": self.t}

        return (self.state, reward, terminated, truncated, info)

    def render_text(self, state):
        grid = np.full(self.walls.shape, ' . ')
        grid[self.walls] = ' # '
        grid[self.goal_pos] = ' G ' 
        grid[state[0], state[1]] = ' A '  
        grid[state[2], state[3]] = ' B '   
        return "\n".join("".join(row) for row in grid)

    