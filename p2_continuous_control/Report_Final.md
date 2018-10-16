[//]: # (Image References)

[image1]: https://github.com/josjo80/DeepRL/blob/master/p1_navigation/download.png "Training Score"

# Project 1: Navigation

### Implementation

For this project, I trained an agent to control a robotic arm to touch a moving target.  The goal of the game environment was to keep the arm end effector on the target for as long as possible and to collect an average of 30 points over 100 episodes.  For a full description of the game please refer to the ReadMe file.  I used the Deep Deterministic Policy Gradient (DDPG) architecture to solve this problem.  The actor networks consisted of 2 hidden layers of 128 and 64 units each with an output layer of 4 for each action.  The output values represent the continuous action space, the torques, of the two joints of the arm.  Each joint has two values.  And the critic networks consisted of 2 hidden layers.  The input layer was 33 units since that is the length of the state space.  The first hidden layer was 128 units.  The second hidden layer input was 128 + 4 units in oder to pass the actions into that layer and the output of that layer was 64 units.  The output of the last layer was 1 unit since it estimates the value of the 4 actions collectively.


### Agent Architecture

I chose to use the DDPG architecture because it can output a continuous action space.  Within this architecture, I updated the main script to handle 20 agents learning simultaneously.  I added all the agents' experiences to a replay buffer and then after 2 timesteps I called self.learn(...) 1 time within my ddpg_agent.py script.  I used a batch size of 1024, an actor learning rate of 1e-3, a critic learning rate of 1e-3 and a gamma of 0.99.  I had trouble initially with getting the agents to learn because I did not understand how to implement updating the actor/critic networks across all agents after a set number of timesteps.  After fixing this code in the ddpg_agent.py script I still had trouble due to the fact that the replay buffer had not been implemented correctly.  I instantiated the buffer in the main script to give all agents access to the collective experience.  I also found a performance boost by using a single network for the actor and a single network for the critic.  This sped up the training time over using an actor and critic network for each agent, ie 20 total. 

### Training Results

The average score across all 20 agents over 100 episodes reached the goal of 30 with episodic scores as high as about 40.

### Improvements

Further exploration of the hyperparameters could be performed to determine an optimal network architecture and see if the agent can learn more efficiently.  Additionally, other learning algorithms could be explored such as PPO and A2C networks.  I would also like to try out differences in the frequency and quantity of training batches.  Rather than 1 training per 2 timestep, I could increase this to multiple trainings per 2 timesteps.

### References
Udacity DRLND slack channel #office_hours and the Study Group for DRLND.  Special thanks to @DavidC mentor for his help.