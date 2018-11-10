# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent3 import Agent
import numpy as np
import torch
from utilities import soft_update
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, random_seed, 
                 discount_factor=0.99, 
                 tau=1e-2, 
                 noise_start=1.0,
                 noise_decay=0.99, 
                 state_size=24, 
                 action_size=2, 
                 lr_actor=1e-3, 
                 lr_critic=1e-3):
        
        super(MADDPG, self).__init__()
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.noise = noise_start
        self.noise_decay = noise_decay
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic

        # critic input = obs_full + actions = 24*2=48
        self.maddpg_agent = [Agent(self.state_size, self.action_size, self.state_size*2, \
                                   self.lr_actor, self.lr_critic, random_seed), 
                             Agent(self.state_size, self.action_size, self.state_size*2, \
                                   self.lr_actor, self.lr_critic, random_seed)]
        
        

    def act(self, obs_all_agents, add_noise=True):
        """get actions from all agents in the MADDPG object"""
        #actions = [print(obs, obs.reshape(1,-1).shape, type(obs)) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        actions = [agent.act(obs.reshape(1,-1), noise_weight=self.noise, add_noise=add_noise) \
                   for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        self.noise *= self.noise_decay
        actions = np.array(actions).reshape(1,-1)
        return actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        states, actions, rewards, next_states, dones = samples
        
        agent = self.maddpg_agent[agent_number]
        

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        actions_next = []
        actions_all = []
        for a_i in range(2):
            #Get next actions for all next states from actor target network
            a_i = torch.tensor([a_i]).to(device)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, a_i).squeeze(1)
            a_next = self.maddpg_agent[a_i].actor_target(next_state)
            actions_next.append(a_next)
            #Get all actions from local actor to feed into the local critic below
            states_each = states.reshape(-1,2,24).index_select(1,a_i).squeeze(1)
            a_each = self.maddpg_agent[a_i].actor_local(states_each)
            actions_all.append(a_each)
            
        actions_next = torch.cat(actions_next, dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.critic_target(next_states, actions_next)
        
        agent_num = torch.tensor([agent_number]).to(device)
        y = rewards.index_select(1, agent_num) + self.discount_factor * q_next * (1 - dones.index_select(1, agent_num))
        
        q = agent.critic_local(states, actions)
        
        agent.critic_optimizer.zero_grad()
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ actions if i == agent_number \
                   else actions.detach() for i, actions in enumerate(actions_all) ]
        
        q_input = torch.cat(q_input, dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic_local(states, q_input).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)
            
            
            