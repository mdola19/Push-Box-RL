# Push-Box-RL
To supplement the Reinforcement Learning theory I have been studying, I built this open-ended project where the general goal is to create a policy for an agent to **push a box to some goal position**. To apply all of the concepts that I am learning, this project was done in layers of evolution, with each layer adding complexity or redefining the problem: starting with a 2D grid world and building up to a sim world with a robot arm and box. The code builds on top of itself but you can see each isolated "layer" by tracking the commits.

## Grid World

- **Environment -** a 2D grid where each cell represents a position. The agent is represented by `A`, box by `B`, goal by `G`, and the only obstacle is walls represented by `W`.
 
- **Actions -** the agent can only move 1 cell  at at time: up, down, left, right represented by integers 0, 1, 2, 3 respectively. If the agent moves into a cell with the box, the box moves by 1 cell in the same direction and the agent takes its place.

  - Hitting a wall doesn't change the position (stays where it was), it is also not something we need to avoid for now.
 
  - **Slippage -** taking an action doesn't guarantee a transition to a cell in that direction, it can slip into the other 3 directions. e.g. There is a chance a move up can turn into a move right based on the **slip probability**.

- **States -** the state of the MDP is given by a 4 element tuple `(ar, ac, br, bc)` represent the row/column the agent and box are in. The **terminal state** is when the box occupies the goal cell.

### Dynamic Programming for MDPs
To begin with, I started off with the simplest form where the MDP is fully defined. We know all:

- **States -** a list storing all possible tuples you can have for a given layout. Created by looping through all combinations and only keeping combinations where both the agent and the box are not occupying the same cell or a wall cell. An additional dictionary was also used to store index-state mappings for faster lookups.

  - **Terminal States -** an additional array for storing all states where the box is on the goal cell.

- **Transition Probabilities-** the probability of taking an action and actually moving to a state in that direction was defined was `1 - slip` (probability of not slipping). The other 3 directions are assigned an equal probability (`slip/3`). Using this idea, the state transition probability matrix, with dimensions `(s, a, s')` -> each entry defines the probability of taking action `a` in state `s` and ending up in state `s'`.
 
  - e.g. taking the action `UP` at `(1,1,3,3)` and ending up in `(2,1)`, which is down, has a probability of slip/3 since it is not the intended direction.
   
  - Note that transition probabilities are accumulated (`+=`) because you can end up in the same next state in multiple ways.
    
    - e.g. if you are in a corner, hitting the wall keeps you in the same position so multiple movements under the same `(s,a)` can lead to the same `s'` therefore their probabilities must be accumulated.

  - If you are in a terminal state, the transition probability for any action taken remains `0` for all other states and `1` for staying in that terminal state -> reaching the goal means you are done so you should not move.

- **Immediate Reward Matrix -** the matrix has dimensions `(s,a)`, where each entry holds expected immediate reward. Taking an action from a non-terminal state gives an immediate reward of 0 if it results in a transition to a terminal state and -1 for any other state. Since this matrix holds expected immediate rewards, we have to compute the probability weighted sum of all possible transitions from every state-action pair.

  - e.g. taking the action `UP` at `(1,1,3,3)` can result in 4 different next states (not hitting a wall or box). Each one gets a reward of -1 because they are all non-terminal states. To get the expected reward from this situation, we take the sum of products `-1 * p` where p represents the transition probabilities to each of the 4 states.

With these pieces in hand, I had everything needed to implement dynamic programming algorithms to solve the MDP.

- **Policy Evaluation -** given any policy, it computes the state-value function by iteratively applying the recursive Bellman expectation equation using both synchronous backups and in-place DP. For synchronous backups, the new state-value function is just the old one plugged back into the equation in each sweep. For in-place DP, each iteration sweeps through all the states to update the value at each state one step at a time while also using the updated value function in each step. I implemented both for comparison purposes to see how fast they converge.

  - The iterative process ends when the maximum error across all states is below a threshold value (`theta`).
    
  - I also had to make an additional function to **flatten** the MDP to an MRP so that I could use the vector equations. When doing so, the transition probability matrix goes from `(s,a,s') -> (s,s')` by averaging over all actions under the policy (which gives the probability of each action at a state) while the reward matrix goes from `(s,a) -> (s,)`.

- **Policy Iteration -** this function then uses the policy evaluation function to find the optimal policy by evaluating a policy, greedily finding the best deterministic policy by giving the action that results in the highest `Q` a probability of 1, and repeating until you reach the optimal policy (stops changing).

  - Since we already have `V`, we can compute `Q` using the Bellman expectation equation -> `R + gamma * P @ V`. We use the unflattened matrices here because `Q` is a function of `(s,a)`.

- **Value Iteration -** a better way of solving for the optimal policy by recursively applying the Bellman optimality equation to find the optimal value function and then greedily finding the deterministic optimal policy using that function. I once again implemented it using both synchronous backups and in-place DP for comparison.

With all the algorithms I learned for DP implemented, I wrote an experiment to ensure everything works as intended and see the optimal policy. I used an open 5x5 grid where the goal is at the bottom right corner (4,4), the box is at (2,1), and the agent starts at (0,0). These are some of the tests I ran:

- **Environment Behaviour vs Model -** I first wanted to ensure the environment worked as intended so I generated 20 random state-action pairs and sampled 5000 steps for each, counting the frequency of each state it transitioned to. I then compared these frequencies to my state-transition matrix and the total variation distance was ~0 (0.0065) meaning that my model represented the environment as expected.
  
- **Policy Evaluation Behaviour -** to qualitatively test whether my policy evaluation was accurate, I used a uniform random policy (equal probability of taking each action), computed the value function, and plotted 3 different heat maps with the box in different positions. The color of the cell represents the value of being in that state. As you can see from the figure below, it seems to be evaluating appropriately.

  - The red arrows represent the deterministic policy. Since we are feeding a uniform random policy here, it just takes the first action at every state (uses argmax on policy at each state).
    
  - Being closer to the box on the side where you can still push towards the goal has the highest value while being on the opposite side where actions can lead to a push away from the goal have the lowest value.

<div align="center">
 <img width="800" height="250" alt="image" src="https://github.com/user-attachments/assets/649b3272-385a-4014-80ae-29c5c90ba301" />
</div>

- **Convergence -** just ran policy and value iteration on the same layout to find optimal policies and plotted log error against sweeps. Policy iteration start with the uniform random policy while value iteration started started with a zero value function. The error decreased by ~gamma (0.99) each sweep, as expected by the Bellman equations. Policy iteration took roughly 4x as many total sweeps to reach the optimal policy (5504 vs 1376).

  - In addition to this, I also checked convergence on synchronous backups vs in-place DP for both of these algorithms. As expected, in-place was better for both: convergence fell to 4775 sweeps for policy iteration and 1306 for value iteration.
  
<div align="center">
<img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/90eff11a-8c96-49c5-8841-86fc008757fe" />
</div>

- **Optimal Policy -** I used a similar heat map to also plot the optimal policy which seems to be taking the right actions in each state meaning the algorithms were successful in solving the MDP.

<div align="center">
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/b7e2f472-7b91-47ed-b869-591d787ae639" />
</div>

- **Unrecoverable States -** this layout has certain states from where it becomes impossible to reach the goal. e.g. if the box ends up in the other corners, it is stuck because the agent can only push the box. In such states, the agent will keep receiving a reward of -1 no matter what action is taken. The value function computes return at a given state which in this case is the geometric series: -(1 + γ + γ² + …). This means that after the value in such states approaches -100 but won't end up at -100 because we iterate until the error is below `1e-6`. Therefore I was able to identify all unrecoverable states by identifying states that had a rounded value of -99.9999 (less than 6 decimal places). I plotted all 216 of these states and they all seem to make sense.

<div align="center">
 <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/5f2c6a7f-4e18-4df8-bd6f-7957a4d48943" />
</div>

### Model-Free Prediction
In progress...

