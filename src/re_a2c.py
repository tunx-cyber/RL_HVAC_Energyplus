import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Config import config
import EnergyplusEnv
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from datetime import datetime
import Config
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 40)
        self.fc2 = nn.Linear(40, 32)
        self.fc3 = nn.Linear(32,action_dim)
        
    def forward(self, x) -> Categorical:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        distribution = Categorical(x)
        return distribution

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 56)
        self.fc2 = nn.Linear(56, 20)
        self.fc3 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class A2C:
    def __init__(self, state_dim, action_dim,
                 lr, gamma):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
    
    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def update(self, rewards, next_state, masks, log_probs, values):
        rewards = torch.FloatTensor(rewards).to(device)
        masks = torch.FloatTensor(masks).to(device)
        
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = self.critic(next_state)
        returns = self.compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optim.step()
        self.critic_optim.step()
        pass
    def save(self):
        torch.save(self.actor.state_dict(), "actor_a2c.pth")
        torch.save(self.critic.state_dict(), "critic_a2c.pth")

    def train(self, eps, env:EnergyplusEnv.EnergyPlusEnvironment):
        max_reward = -200
        x,y = [], []
        for i in range(eps):
            state = env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            done = False
            while not done:
                state = torch.FloatTensor(state).to(device)
                dist, value = self.actor(state), self.critic(state)

                action = dist.sample()
                next_state, reward, done = env.step(action)

                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(1-done)

                state = next_state
            
            total_reward = np.sum(rewards)
            if total_reward > max_reward:
                max_reward = total_reward
                print(f"model save at {datetime.now()}")
                self.save()

            self.update(rewards, next_state, masks, log_probs, values)

            x.append(i)
            y.append(total_reward)
        
        return x, y
    
    def test(self, env:EnergyplusEnv.EnergyPlusEnvironment):
        self.actor.load_state_dict(torch.load("actor_a2c.pth"))
        self.critic.load_state_dict(torch.load("critic_a2c.pth"))
        
        state = env.reset("test")
        done = False
        while not done:
            state = torch.FloatTensor(state).to(device)
            dist, value = self.actor(state), self.critic(state)

            action = dist.sample()
            next_state, reward, done = env.step(action)
            state = next_state
        
        print(f"a2c result: {env.total_reward, env.total_energy, env.total_temp_penalty}")

    

if __name__ == '__main__':
    cfg = Config.config()
    env = EnergyplusEnv.EnergyPlusEnvironment(cfg)
    agent = A2C(env.observation_space_size, env.action_space_size, 0.001, 0.99)
    start = datetime.now()
    x,y = agent.train(eps=200,env=env)
    end = datetime.now()
    print(f"trainning time is {end-start}")
    plt.plot(x,y,color='b')
    
    plt.show()
    agent.test(env)


