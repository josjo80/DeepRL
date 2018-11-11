[//]: # (Image References)

[image1]: https://github.com/josjo80/DeepRL/blob/master/p3_collab_compet/5th_success.png "Training Score"

# Project 3: Multi-Agent Collaboration/Competition in Tennis

### Implementation

For this project, I trained multiple agents to control their tennis racket to volley a ball.  The goal of the game was to keep the ball in play for as long as possible.  The agents get +0.1 points for hitting the ball and -0.01 if the ball hits the ground or goes out of bounds.  For a full description of the game please refer to the ReadMe file.  I used the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) architecture to solve this problem.  The actor networks consisted of 2 hidden layers of 128 and 64 units each with an output layer of 2 for each action.  The output values represent the continuous action space, the horizontal and vertical distance to move in the environment (i.e. towards the net is positive horizontal action for each agent).  The critic networks were able to observe all agents' states and actions and therefore consisted of 2 hidden layers.  The input layer was 48 units since that is twice the length of the state space.  The first hidden layer was 128 units.  The second hidden layer input was 128 + 4 units in oder to pass the actions from both agents into that layer. The output of that layer was 64 units.  The output of the last layer was 1 unit since it estimates the value of the 4 actions collectively.


### Agent Architecture

I chose to use the MADDPG architecture because it can be used with multiple agents in both a collaborative and competitive scenario along with the fact that it can output a continuous action space.  The architecture is similar to DDPG but for each agent's critic network, it sees the entire states and actions of all agents. Within this architecture, I updated the main script to handle interacting with the environment, adding all the agents' experiences to a replay buffer and then after 2 timesteps I called self.update(...) 1 time.  I used a batch size of 512, an actor learning rate of 1e-3, a critic learning rate of 1e-3, a gamma of 0.99, and a tau of 1e-1.  I had trouble initially with getting the agents to learn with a batch size of 1024 and a tau of 1e-3.  While the success from one training run to the next varied greatly in the number of episodes, the batch size and tau made the biggest difference.  This seems to indicate that by increasing tau, this increased the update of target weights and improved convergence.  Lowering the batch size effectively gives less precise gradient steps but can also help with getting out of local minima.  This appears to have helped in this case.

### Training Results

![Training Score][image1]

It took 703 episodes to reach the goal of 0.5 average max score.  It took several attemps to dial in the batch size and tau, as described above, to get the training stable and converge relatively fast.

### Improvements

Further exploration of the hyperparameters could be performed to determine an optimal network architecture and see if the agent can learn more efficiently.  Additionally, other learning algorithms could be explored such as PPO.  I would also like to try out differences in the frequency and quantity of training batches.

### References
Udacity DRLND slack channel #office_hours and the Study Group for DRLND.  I also used the baseline implementation from the Udacity MADDPG workspace.