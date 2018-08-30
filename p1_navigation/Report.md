[//]: # (Image References)

[image1]: https://github.com/josjo80/DeepRL/blob/master/p1_navigation/download.png "Training Score"

# Project 1: Navigation

### Implementation

For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.  The goal of the game environment was to collect as many bananas as possible in a certain period of time.  For a full description of the game please refer to the ReadMe file.  I used the Double Deep Q Network (DDQN) architecture to solve this problem.  The Deep Q networks consisted of 3 hidden layers of 256, 128, and 64 units each.  The input layer was 37 units since that is the length of the state space.  The output was 4 units since there are 4 potential actions.  I chose to use 3 layers and the number of units after experimenting with these parameters and found that they resulted in acceptable performance.  


### Agent Architecture

I chose to use the DDQN architecture because it generally learns fasterand more efficiently.  Within this architecture, I used a batch size of 32, a learning rate of 5e-4, and a gamma of 0.99.  I noticed that slower learning rates or larger batch sizes led to slower overall learning.  The DQN algorithm suffers from overestimating the Q-values due to using the max operation to choose the best action within the TD target, (rewards + gamma*Q(s',max(Q(s',a'))).  Because the Q-value can be noisy early on in learning, this tends to lead to overestimation of the Q-values.  DDQN resolves this by using the local Q-value network to choose the action as before, but then calculate the action-value using the target Q-value network which updates at a slower rate, Q(s',argmax(Q(s',a', theta_local), theta_i_target)).  Because the target network isn't updating every timestep, it's estimate of the action-value is more stable.  Also, by using the target network to evaluate the local network's choise, overestimated action-values from the local network get updated with a relatively lower,or muted, value. In the long run this prevents the network from propogating incidental high rewards that could have been achieved by chance and don't reflect long term returns.

### Training Results

![Training Score][image1]

The above is the agent's training score and shows that it was able to achieve an average score of 13 after approximately 500 episodes.

### Improvements

Further exploration of the hyperparameters could be performed to determine an optimal network architecture and see if the agent can learn more efficiently.  Additionally, other learning algorithms could be explored such as prioritized experience replay and dueling DQN networks.  Finally, it would be interesting to try learning directly from the pixels as opposed to the derived state space given.

### References
https://github.com/dxyang/DQN_pytorch