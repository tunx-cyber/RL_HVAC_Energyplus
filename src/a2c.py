import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Config import config
import EnergyplusEnv
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义A2C代理
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        
    def select_action(self, state):
        state_ = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.actor(state_)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()
    
    def update(self, states, actions, rewards, next_states, dones): # TODO: add comment for RL
        # transform into tensor
        nxt_state = torch.FloatTensor(next_states[-1]).to(device)
        states_ = torch.FloatTensor(states).to(device)
        actions_ = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_ = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones_ = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # 计算Advantage
        values = self.critic(states_)
        next_value = self.critic(nxt_state)
        returns = self.discount_with_dones(next_value, rewards_, dones_)
        advantages = torch.cat(returns).detach() - values
        
        # 更新Actor网络
        self.optimizer_actor.zero_grad()
        action_probs = self.actor(states_)
        dist = Categorical(action_probs) # distribution of probablity
        log_probs = dist.log_prob(actions_.squeeze(1))
        actor_loss = -(log_probs * advantages.detach()).mean()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # 更新Critic网络
        self.optimizer_critic.zero_grad()
        critic_loss = advantages.pow(2).mean()
        critic_loss.backward()
        self.optimizer_critic.step()
    
    def discount_with_dones(self, next_value, rewards, dones):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1.0 - dones[step])
            returns.insert(0, R)
        return returns
    
    def save(self):
        torch.save(self.actor.state_dict(), "actor.pth")
        torch.save(self.critic.state_dict(), "critic.pth")

# 训练A2C代理
def train_a2c():

    # basic setting
    cfg = config()
    env = EnergyplusEnv.EnergyPlusEnvironment(cfg=cfg)
    # 定义环境和代理参数
    state_dim = env.observation_space_size  # 状态维度
    action_dim = env.action_space_size  # 动作维度
    lr = 0.001  # 学习率
    gamma = 0.99  # 折扣因子
    num_episodes = 100  # 训练的总回合数

    agent = A2CAgent(state_dim, action_dim, lr=lr, gamma=gamma)

    max_reward = float("-inf")
    x = []
    y = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        states, actions, rewards, next_states, dones = [], [], [], [], []

        # 从7:15 到 21:00
        # 29 到 84的时间步 在energyplus.py里面改
        while not done:
            action = agent.select_action(state) # 这里改为多个agent选择
                                                # 相应的所有的next_state action要单独一个agent，不过reward和done是通用的，需要适当修改
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state

        agent.update(states, actions, rewards, next_states, dones)
        
        total_reward = np.sum(rewards)
        if episode % 10 == 0:
            print("Episode: {}, Reward: {}".format(episode, total_reward))
        
        x.append(episode)
        y.append(total_reward)
        if total_reward > max_reward:
            agent.save()
            max_reward = total_reward

    return x, y


if __name__ == "__main__":
    x,y = train_a2c()

    plt.plot(x,y,color = 'r')
    plt.show()
