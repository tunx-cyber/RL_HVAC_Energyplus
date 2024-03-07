import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
# 定义 Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# 定义 Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class A2C_MA:
    def __init__(self, num_agnets, state_dim, action_dim) -> None:
        self.num_agnets = num_agnets
        self.actors = [Actor(state_dim=state_dim, action_dim=action_dim) for _ in range(num_agnets)]
        '''
        只是用一个critic
        1. 参数共享：显著降低模型的计算复杂度和内存占用。
        2. 信息共享：不同智能体共享一个 Critic 网络可以促使智能体之间共享信息
                    智能体可以从其他智能体的经验中学习，从而提高整体性能。
        3. 技巧转移：共享 Critic 网络允许智能体之间进行技巧转移。
                    如果一个智能体在学习中发现了一种有效的策略或动作选择，其他智能体可以通过共享 Critic 网络获得这些知识，并更快地收敛到更好的策略。
        4. 信息冲突：在共享 Critic 网络中，不同智能体的经验可能会相互干扰，导致信息冲突
        5. 训练不稳定：由于共享 Critic 网络，不同智能体的训练样本可能会高度相关，这可能导致训练的不稳定性
        6. 难以学习个体策略：共享 Critic 网络可能会限制智能体学习个体策略的能力。由于 Critic 网络是共享的，它可能更倾向于学习全局性的策略，而忽视个体智能体的差异性和特殊需求。
        '''
        self.critic = Critic(state_dim) 
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_action(self, agent_index, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()
    
    def update(self, agent_index, states, actions, rewards, next_states, dones, gamma=0.99):
        state = torch.tensor(states[agent_index], dtype=torch.float32).to(device=self.device)
        action = torch.tensor(actions[agent_index], dtype=torch.long).unsqueeze(1).to(device=self.device)
        reward = torch.tensor(rewards[agent_index], dtype=torch.float32).to(device=self.device)
        next_state = torch.tensor(next_states[agent_index], dtype=torch.float32).to(device=self.device)
        done = torch.tensor(dones[agent_index], dtype=torch.float32).to(device=self.device)
        
        # 计算 Advantage
        # unsqueeze 方法用于在指定位置增加维度。它在指定位置插入一个维度大小为 1 的维度。
        # squeeze 方法用于压缩维度为 1 的维度。它会移除维度大小为 1 的维度，使张量更紧凑。
        value = self.critic(state.unsqueeze(0)).squeeze(0)
        next_value = self.critic(next_state.unsqueeze(0)).squeeze(0)
        advantage = reward + gamma * (1 - done) * next_value - value
        
        # 更新 Critic 网络
        critic_loss = nn.MSELoss()(value, reward + gamma * (1 - done) * next_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新 Actor 网络
        action_probs = self.actor(state.unsqueeze(0)).squeeze(0)
        log_probs = torch.log(action_probs.gather(1, action))
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self):
        state_dict_dict = {
            f'agent_{i+1}': x.state_dict() for i, x in enumerate(self.actors)
        }
        torch.save(state_dict_dict, "ma_actors.pth")
        torch.save(self.critic.state_dict(), "ma_critic.pth")
    
    def load(self):
        state_dict_dict = torch.load("ma_actors.pth")
        for i, agent_model in enumerate(self.actors):
            agent_model.load_state_dict(state_dict_dict[f'agent_{i+1}'])
        self.critic.load_state_dict(torch.load("ma_critic.pth"))

        
def train_multi_agent(env, ma_agent: A2C_MA,num_episodes):
    max_reward = float("-inf")
    episodes = []
    rewards_ = []
    

    for i in range(num_episodes):
        state = env.reset()
        done = False

        size = ma_agent.num_agnets
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        

        while not done:
            ma_actions = []
            for agent_index in range(size):
                ma_action = ma_agent.get_action(agent_index, state=state)
                ma_actions.append(ma_action)

            next_state, reward, done = env.step(ma_actions)
            states.append(state)
            actions.append(ma_actions)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
        
        #TODO: update到底是每次step后就执行还是一个回合后就执行
        for idx in range(size):
            ma_agent.update(idx, states, actions, reward, next_states, dones, 0.99)
        
        total_reward = np.sum(rewards)
        episodes.append(i)
        rewards_.append(total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            ma_agent.save()
    
    # show the result
    plt.plot(episodes, rewards_, color = 'r')
    plt.show()



