import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Config import config
from EnergyplusEnv import EnergyPlusEnvironment
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义A2C代理
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.device = torch.device("cpu")
        self.real_init()
        

    def real_init(self):
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 计算Advantage
        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # 更新Actor网络
        self.optimizer_actor.zero_grad()
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions.squeeze(1))
        actor_loss = -(log_probs * advantages.detach()).mean()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # 更新Critic网络
        self.optimizer_critic.zero_grad()
        critic_loss = advantages.pow(2).mean()
        critic_loss.backward()
        self.optimizer_critic.step()


def train_a3c(global_agent : A2CAgent,
              rank,res, file_suffix):
    torch.manual_seed(rank)
    cfg = config()
    env = EnergyPlusEnvironment(cfg=cfg)
    state_dim = env.observation_space_size  # 状态维度
    action_dim = env.action_space_size  # 动作维度
    lr = 0.001  # 学习率
    gamma = 0.99  # 折扣因子
    num_episodes = 100  # 训练的总回合数
    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim,lr=lr,gamma=gamma)
    agent.actor.load_state_dict(global_agent.actor.state_dict())
    agent.critic.load_state_dict(global_agent.critic.state_dict())
    agent.optimizer_actor = optim.Adam(global_agent.actor.parameters(), lr)
    agent.optimizer_critic = optim.Adam(global_agent.critic.parameters(), lr)
    for episode in range(num_episodes):
        state = env.reset(file_suffix)
        done = False

        states, actions, rewards, next_states, dones = [], [], [], [], []
        x, y =[], []
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state

            agent.update(states, actions, rewards, next_states, dones)
        x.append(episode)
        y.append(np.sum(rewards))
        if episode % 100 == 0:
            global_agent.actor.load_state_dict(agent.actor.state_dict())
            global_agent.critic.load_state_dict(agent.critic.state_dict())
            global_agent.optimizer_actor.load_state_dict(agent.optimizer_actor.state_dict())
            global_agent.optimizer_critic.load_state_dict(agent.optimizer_critic.state_dict())
            
            print(f"Episode {episode}, Rank {rank}, Reward: {reward}")
    res.append([x,y])

def main():
    cfg = config()
    env = EnergyPlusEnvironment(cfg=cfg)
    # 定义环境和代理参数
    state_dim = env.observation_space_size  # 状态维度
    action_dim = env.action_space_size  # 动作维度
    lr = 0.001  # 学习率
    gamma = 0.99  # 折扣因子
    global_agent = A2CAgent(state_dim,action_dim,lr,gamma)
    # global_agent.actor.share_memory()
    # global_agent.critic.share_memory()

    processes = []
    res = []
    for rank in range(6):# 解决访问冲突问题，修改生成文件的文件名
        p = mp.Process(target=train_a3c, args=(global_agent,rank,res,str(rank)))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    import matplotlib.pyplot as plt
    for rank in range(len(res)):
            
        plt.plot(res[rank][0],res[rank][1],color = 'r')
        plt.show()
if __name__ == '__main__':
    main()
