[//]: # (Image References)

[image1]: https://github.com/josjo80/DeepRL/blob/master/p1_navigation/download.png "Training Score"

# Project 1: Navigation

### Implementation

For this project, I trained an agent to control a robotic arm to touch a moving target.  The goal of the game environment was to keep the arm end effector on the target for as long as possible and to collect an average of 30 points over 100 episodes.  For a full description of the game please refer to the ReadMe file.  I used the Deep Deterministic Policy Gradient (DDPG) architecture to solve this problem.  The actor netowrks consisted of 2 hidden layers of 128 and 64 units each.  And the critic networks consisted of 3 hidden layers.  The input layer was 33 units since that is the length of the state space.  The first hidden layer was 128 units.  The second hidden layer input was 128 + 4 units in oder to pass the actions into that layer and the output of that layer was 64 units.  The output of the last layer was 4 units since there are 4 potential actions.  


### Agent Architecture

I chose to use the DDPG architecture because it can output a continuous action space.  Within this architecture, I updated the main script to handle 20 agents learning simultaneously.  I added all the agents' experiences to a replay buffer and then after 20 timesteps I called self.learn(...) 10 times within my ddpg_agent.py script.  I used a batch size of 128, an actor learning rate of 1e-4, a critic learning rate of 1e-3 and a gamma of 0.99.  I had trouble initially with getting the agents to learn because I did not understand how to implement updating the actor/critic networks across all agents after 20 timesteps.  After fixing this code in the ddpg_agent.py script I still had trouble due to the fact that my critic had an output size of 1, which meant that it was giving a single Q value across all actions.  I bumped this up to 4 and got much better results.  I also increased the max_t limit to 1000 to gather more experience per agent per episode and this seemed to help as well.

### Training Results

The average score across all 20 agents over 100 episodes reached as high as 11.

### Improvements

Further exploration of the hyperparameters could be performed to determine an optimal network architecture and see if the agent can learn more efficiently.  Additionally, other learning algorithms could be explored such as PPO and A2C networks.

### References
N/A