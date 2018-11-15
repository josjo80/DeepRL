import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, random_seed, 
                 discount_factor=0.99, 
                 tau=1e-2, 
                 update_every = 2,
                 noise_start=1.0,
                 noise_decay=0.99, 
                 num_agents=2,
                 action_size=2, 
                 state_size=24, 
                 buffer_size=1e6, 
                 batch_size=512, 
                 lr_actor=1e-3, 
                 lr_critic=1e-3, 
                 weight_decay=0):
        #super(MADDPG, self).__init__()

        self.discount_factor = discount_factor #Gamma
        self.tau = tau                         #Soft update rate of target parameters
        self.iter = 0
        self.ii = 0                            #timestep in environment over all episodes
        self.update_every = update_every       #Interval over which to perform a learning step
        self.noise = noise_start               #Starting noise amplitude
        self.noise_decay = noise_decay         #Noise decay rate
        self.num_agents = num_agents           #Number of agents
        self.action_size = action_size         #Action size
        self.state_size = state_size           #State size
        self.buffer_size = buffer_size         #Replay Buffer size
        self.batch_size = batch_size           #Mini-batch size for training
        self.lr_actor = lr_actor               #Learning rate for the actor
        self.lr_critic = lr_critic             #Learning rate for the critic
        self.weight_decay = weight_decay       #L2 weight decay of optimizer
        
        #Create Replay Buffer
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, random_seed)

        
        # critic input = obs_full = 24*2=48
        self.maddpg_agent = [Agent(self.state_size, self.action_size, self.state_size*2, random_seed, self.tau, 
                                   lr_actor=self.lr_actor, lr_critic=self.lr_critic, weight_decay=self.weight_decay), 
                             Agent(self.state_size, self.action_size, self.state_size*2, random_seed, self.tau,
                                  lr_actor=self.lr_actor, lr_critic=self.lr_critic, weight_decay=self.weight_decay)]
        
        
    def act(self, obs_all_agents, add_noise=True):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs.reshape(1,-1), noise_weight=self.noise, add_noise=add_noise) \
                   for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        self.noise *= self.noise_decay
        actions = np.array(actions).reshape(1,-1)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        states = states.reshape(1,-1)
        next_states = next_states.reshape(1,-1)
        
        self.memory.add(states, actions, rewards, next_states, dones)
        
        self.ii = (self.ii + 1) % self.update_every
        if self.ii == 0 and len(self.memory) > self.batch_size:
            experiences = [self.memory.sample() for i in range(self.num_agents)]
            self.learn(experiences, self.discount_factor)
        
    def learn(self, experiences, gamma):
        # Get predicted pred_actions and next-state actions from each target model
        actions_next_all = []
        actions_all = []
        for i in range(self.num_agents):
            states, _, _, next_states, _ = experiences[i]
            i = torch.tensor([i]).to(device)
            next_states_each = next_states.reshape(-1,2,24).index_select(1, i).squeeze(1)
            #Get the predicted nexst actions from the actor's target network a' = pi(s': theta_target)
            a_next_each = self.maddpg_agent[i].actor_target(next_states_each)
            actions_next_all.append(a_next_each)
            #Get predicted actions from the actor's local network
            states_each = states.reshape(-1,2,24).index_select(1,i).squeeze(1)
            a_each = self.maddpg_agent[i].actor_local(states_each)
            actions_all.append(a_each)
        
        for i, agent in enumerate(self.maddpg_agent):
            agent.learn(i, experiences[i], self.discount_factor, actions_next_all, actions_all)
        
            


class Agent():
    """DDPG Agent"""
    
    def __init__(self, state_size, 
                 action_size, 
                 full_obs_size, 
                 random_seed, 
                 tau, 
                 lr_actor,
                 lr_critic,
                 weight_decay):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network) Use action_size*2 since there are 2 agents
        self.critic_local = Critic(full_obs_size, action_size*2, random_seed).to(device)
        self.critic_target = Critic(full_obs_size, action_size*2, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        
    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()*noise_weight
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, i, experiences, gamma, actions_next_all, actions_all):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        #i = agent number
        i = torch.tensor([i]).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next_all = torch.cat(actions_next_all, dim=1).to(device)
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next_all)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards.index_select(1, i) + (gamma * Q_targets_next * (1 - dones.index_select(1, i)))
        # Calc current critic Q value of current state, action
        Q_expected = self.critic_local(states, actions)
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        #Select actions from current agent
        #Detach actions from other agents to help speed up training
        actions_all = [actions if i_d == i else actions.detach() for i_d, actions in enumerate(actions_all)]
        actions_all = torch.cat(actions_all, dim=1).to(device)
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_all).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=int(buffer_size))  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)