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
To introduce more complexity and make the problem more realistic, I took away the state transition probabilities matrix and rewards matrix. It isn't realistic to know these values for every state and action possible, particularly in situations where there can be infinite states. This part mainly covers how this impacts prediction (policy evaluation). 

To build up this problem, we first need a stochastic policy to evaluate. The reason it should be stochastic is because our value function will now be learned through experience and the performance of the methods used can vary based on these experiences. A deterministic policy would result in the same actions being sampled in each state. While the slip from the environment can still result in small variation, it won't be enough to capture the differences across methods. In addition to this, a stochastic policy also allows us to visit more states, so there are more states to measure value on. 

- **Soft Policy -** the stochastic policy was constructed by taking the optimal policy from the result of DP and introducing noise: optimal action gets probability 0.85 while other 3 actions get 0.05.

Since we are now collecting experience with this policy, we need a framework for stepping through the environment. To do this, I created an `Episode` data class that stores all the states visited, actions taken, and rewards received in a list, as well as whether the episode terminated (reached goal) or truncated (timed out before reaching goal). I also created a function to compute discounted returns for a given episode which returns an array storing the total discounted return at every time step in the episode.

With the problem now set up, I implemented the algorithms I learned for model-free prediction as separate functions that all run the policy for `n_episodes` episodes and return the value function as well as the number of episodes that were truncated.

- **Monte-Carlo -** simply uses a running average of return to update the value function after every episode for every state visited. If `first_visit` is enabled, value of a state is only updated once per episode.

  - `alpha` controls how important new episodes are: closer to 1 pulls the value at that state closer to the estimate from that episode. If you want it to be an exact average (all episodes matter equally) then setting `running_average` holds a count for the total number of times each state has been visited across all episodes and so it divides the estimate by that count.
 
- **TD(0) -** essentially the same as MC but instead of using total return at every state for each episode, it uses **bootstrapping** by estimating the total return using the immediate reward and value at the next state. As you keep running episodes, the value function gets better and so the estimate for return (target) gets better, recursively pulling the function towards the true value.
  
  - Programmatically, we are now applying updates within the episode after every step rather than after the episode when all returns are available (which MC does)
  
  - The target is more biased (initialized as zeros) but with lower variance in the earlier episodes than MC because it only uses one step in the target estimation instead of the whole episode (each step introduces noise).
 
- **n-step TD -** builds on TD(0) by extending the target from just using immediate reward and next state's value to using an n-step return and the value of the state that comes after those n-steps to estimate the total return. This helps form a middle ground between TD(0) and MC to weigh the tradeoff between bias and variance.

  - Similar to TD(0), it updates `V` on each step in every episode but only starts updating after n-steps have been taken. A double-ended queue was used to facilitate this.
 
- **Forward TD(λ) -** takes the n-step TD a step further by setting the target to the weighted average n-step return over all n. This requires having the full episode available which makes it impractical because MC is simpler and uses the actual return instead of estimating it. It was just implemented to demonstrate the idea.

- **Backward TD(λ) -** this method takes a slightly different approach to account for credit assignment: the impact previous steps have on value doesn't just come from **recency** but also **frequency**. This method now tracks an **eligibility trace** `E` and updates the value at _all_ states in every step.

  - The eligibility trace of a state shrinks by `gamma * lambda` on every step and if you are currently in that state, it also increments by 1 (simultanously accounting for frequency and recency).
  
  - The value update target itself is the same as in TD(0): immediate reward + next state's value
    
  - The value at each state is updated by the TD error, proportional to the eligibility trace at that state and `alpha`
  
    -  e.g. if a state was visited at t = 0 and never again, it still gets updated on every step after but the amount it is updated shrinks every step

These methods account for the main algorithms I learned for when the number of episodes can be controlled. If you are in a situation where there is a limited number of episodes available, you can run TD in a loop over the same episodes until you reach within a certain relative error level. I demonstrated this for the TD(0) method in `batch_td_0`, which takes a list of episodes and returns the value function. 

  - I also learned that this should converge to the the solution of the maximum likelihood Markov model. To demonstrate this, I wrote `certainty_equivalence` which counts the frequency of all states transitions, total rewards received at each state/action, and visits to each state/action across all steps from all episodes and uses this count to compute the transition probabilities matrix and reward matrix (average reward). This models the maximum likelihood Markov model. It also creates a policy by dividing visits to each state/action by total actions actions taken at that state to implicitly represent the policy used in these episodes. Together, using `policy_evaluation` from DP, it gives us the value function, which should be the same as what `batch_td_0` gives us.

With all these algorithms in place, I wanted to actually test them out and see how they compare, while also playing around with the parameters (α, λ, n) to see what works best. These are some of the experiments I ran and their results:

- **Learning Curves -** using policy evaluation results from DP as ground truth for the value function of the soft policy, the root mean square error for the Monte-Carlo, TD(0), and backward TD(0.8) algorithms were computed every 10 episodes. This was done for 1000 episodes using 5 different seeds for the environment and action sampler.
  
  - Each line shows the average RMS error across the 5 seeds and the shaded region around the line shows the 95% confidence interval of the mean
    
    - The 95% CI is computed by sampling 1000 means across different combinations of the 5 seeds used and calculating the 2.5th and 97.5th percentiles
      
    - This shows how much the mean RMS error changes depending on the seeds you choose for sampling actions or stepping through the environment.
      
      - There's a little bit of overlap around episode 300 between MC and TD(0.8) which means that MC may actually have a lower mean RMS error for some seed combinations under these configurations.
        
  - Note that the curves below were generated after finding the best values for `alpha` from the other experiments: TD(0) (α = 0.5), MC (α = 0.35), TD(0.8) (α = 0.35).
    
  - From the graph, you can see that in general, MC has the best performance, TD(0) has the worst performance, and TD(0.8) is in-between.
    
    - On the other hand, TD has a much smoother curve compared to MC which occasionally has big spikes. This is because MC uses the full return on each update so an episode where the environment slipped into bad states can pull the value function down (error goes up) whereas TD only uses 1 or a few immediate steps in it's estimate of the return so a single episode can't make as much of a difference.
   
<div align="center">
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/973198a0-98e3-4a8d-ad08-bce703046074" />
</div>

- **Step Size -** for the two algorithms that mainly depend on `alpha` to control performance (TD(0) & MC), similar to the learning curves, the mean RMS error with the DP value function across 5 different seeds was plotted against different step sizes. As expected, both curves are u-shaped which makes sense because smaller step sizes don't update the value function fast enough after each episode and bigger step sizes tend to pull the value function towards the latest episode without letting the value settle.

  - In general, TD(0) could not outperform MC at any step size but comes relatively close near it's optimal step size (0.5). This is a reasonable tradeoff for not having complete episodes.
  
  - The curve for TD(0) shoots up much more on the higher step size than the curve for MC, which also makes sense because the updates to V(s) depend on V(s') so a bad update to one state immediately feeds back into updates to previous states which can blow up when step size is high. Conversely, MC updates depend on the full return so one bad update with a high step size just introduces noise once, it doesn't feed into the target for other states. 

<div align="center">
<img width="450" height="275" alt="image" src="https://github.com/user-attachments/assets/940cbb07-975b-46d8-b8c6-7114709565ed" />
</div>

- **n-step TD Heat Map -** performance of n-step TD can be controlled by both `alpha` and `n` so I ran n-step TD for 1000 episodes over 5 different seeds across different step size and n combinations and plotted a heat map showing the resulting RMS error for each combination. The configuration that resulted in the lowest RMS error was **n = 4-8** and **α = 0.35**. This fits the general idea I learned that a ~middle step-size is typically the best (similar to what was discussed with the u-shaped curve above) while a middle n balances the bias of TD(0) against the variance of MC.

  - Another observation I made was that step size typically matters more than n anyways because the heat map shows that you can still get a relatively low RMS error with a "bad" n by just choosing the right step size.
 
<div align="center">
<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/58ba7200-c6fd-4cdb-8b8f-dbb5f9406e9e" />
</div>

- **TD(λ) Heat Map -** I created a similar heat map for TD(λ) which depends on λ and step-size. In general, it gives the same idea as the last plot, where middle values are better for λ to balance the trade-off. Forward-view is slightly better since it updates offline with a fixed V, so bad updates can’t feed back within an episode. The more practical implementation, with backward view, is still close at it's optimal configurations but the RMS error blows up in the bottom right corner (higher step-size/λ). This is the same problem discussed earlier (bad update feeding back into other targets) but made much worse because instead of just multiplying TD error by a high `alpha`, it is also multiplied by `E`. When λ ~= 1, `E` barely shrinks and frequent visits (which happen a lot in a grid environment) can easily bring this value above 1 for a lot of states, resulting in amplified bad updates, which we know spread to other states based on the mechanics of the algorithm.

<div align="center">
<img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/29d9e179-1b4b-4295-bc35-681661b4f971" />
</div>

- **Batch Comparison -** to compare the batch computations for TD(0) and MC, a simple bar graph for the RMS error after 50 episodes was plotted, showing that batch TD leads to a lower RMS error but the brief overlap in 95% CI means that there may be some seeds where MC could be better. In general though, this confirms the idea I learned that batch TD is more efficient for Markov environments.

  - In addition to this, I also plotted the RMS error and 95% CI for the certainty equivalence, which is essentially identical. This confirms the idea that batch TD(0) converges to the solution of the maximum likelihood Markov model.
 
<div align="center">
<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/0e39904d-8fc4-4a08-bfda-8df3bc809f0c" />
</div>

### Model-Free Control




   
