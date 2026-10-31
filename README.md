# Push-Box-RL
To supplement the Reinforcement Learning theory I have been studying, I built this open-ended project where the general goal is to create a policy for an agent to **push a box to some goal position**. To apply all of the concepts that I am learning, this project was done in layers of evolution, with each layer adding complexity or redefining the problem: starting with a 2D grid world and building up to a sim world with a robot arm and box. The code builds on top of itself but you can see each isolated "layer" by tracking the commits.

## Grid World

- **Environment -** a 2D grid where each cell represents a position. The agent is represented by "A", box by "B", goal by "G", and the only obstacle is walls represented by "W".
 
- **Actions -** the agent can only move 1 cell  at at time: up, down, left, right represented by integers 0, 1, 2, 3 respectively. If the agent moves into a cell with the box, the box moves by 1 cell in the same direction and the agent takes its place.

 - Hitting a wall doesn't change the position (stays where it was), it is also not something we need to avoid for now.

 - **Slippage -** taking an action doesn't guarantee a transition to a cell in that direction, it can slip into the other 3 directions. e.g. There is a chance a move up can turn into a move right based on the **slip probability**.

- **States -** the state of the MDP is given by a 4 element tuple `(ar, ac, br, bc)` represent the row/column the agent and box are in. The **terminal state** is when the box occupies the goal cell.

### Dynamic Programming for MDPs
To begin with, I started off with the simplest form where the MDP is fully defined: we know all transition probabilities and rewards...
