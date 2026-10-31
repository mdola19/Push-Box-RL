import numpy as np

'''
. = empty floor
# = wall
G = goal
A = where agent starts
B = where box starts
'''

LAYOUTS = {"open5" : np.array([]), "rooms7": np.array([])}

LAYOUTS["open5"] = np.full((5, 5), '.')
LAYOUTS["open5"][0, 0] = 'A'
LAYOUTS["open5"][2, 1] = 'B'
LAYOUTS["open5"][4, 4] = 'G'

LAYOUTS["rooms7"] = np.full((7, 7), '.')
LAYOUTS["rooms7"][:, 3] = '#'
LAYOUTS["rooms7"][3, 3] = '.'
LAYOUTS["rooms7"][0, 0] = 'A'
LAYOUTS["rooms7"][2, 1] = 'B'
LAYOUTS["rooms7"][5, 6] = 'G'

def parse_layout(layout):
    agent_pos = tuple(np.argwhere(layout == 'A')[0].tolist())
    box_pos = tuple(np.argwhere(layout == 'B')[0].tolist())
    goal_pos = tuple(np.argwhere(layout == 'G')[0].tolist())
    walls = layout == '#'
    return walls, agent_pos, box_pos, goal_pos

