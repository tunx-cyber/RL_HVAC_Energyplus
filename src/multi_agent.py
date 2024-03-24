import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from multi_agent_env import Multi_Agent_Env
import Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datetime import datetime
# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(16,action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=-1)
        return x

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class A2C_MA:
    def __init__(self, num_agnets, state_dim, action_dim) -> None:
        self.num_agnets = num_agnets
        self.gamma = 0.99
        self.actors = [Actor(state_dim=state_dim, action_dim=action_dim).to(device=device) for _ in range(num_agnets)]
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
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizers =[ optim.Adam(self.actors[i].parameters(), lr=0.002) for i in range(num_agnets)]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.01)
        
    
    def get_action(self, agent_index, state):

        state_ = torch.FloatTensor(state[agent_index]).unsqueeze(0).to(device)
        action_probs = self.actors[agent_index](state_)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def update(self, agent_index, states, actions, rewards, next_states, dones, gamma=0.99):
         
        nxt_state = torch.FloatTensor(next_states[-1][agent_index]).to(device=device)
        states_ = torch.FloatTensor(np.array(states)[:,agent_index]).to(device=device)
        actions_ = torch.LongTensor(np.array(actions)[:,agent_index]).unsqueeze(1).to(device=device)
        rewards_ = torch.FloatTensor(np.array(rewards)[:,agent_index]).unsqueeze(1).to(device=device)
        dones_ = torch.FloatTensor(dones).unsqueeze(1).to(device=device)
        
 
        # 计算 Advantage
        # unsqueeze 方法用于在指定位置增加维度。它在指定位置插入一个维度大小为 1 的维度。
        # squeeze 方法用于压缩维度为 1 的维度。它会移除维度大小为 1 的维度，使张量更紧凑。
        values = self.critic(states_)
        next_value = self.critic(nxt_state)
        returns = self.discount_with_dones(next_value, rewards_, dones_)
        advantages = torch.cat(returns).detach() - values
        
        # 更新Actor网络
        self.actor_optimizers[agent_index].zero_grad()
        action_probs = self.actors[agent_index](states_)
        dist = Categorical(action_probs) # distribution of probablity
        
        log_probs = dist.log_prob(actions_.squeeze(1))
        actor_loss = -(log_probs * advantages.detach()).mean()
        actor_loss.backward()
        self.actor_optimizers[agent_index].step()
        
        # 更新Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss = advantages.pow(2).mean()
        critic_loss.backward()
        self.critic_optimizer.step()


    def discount_with_dones(self, next_value, rewards, dones):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1.0 - dones[step])
            returns.insert(0, R)
        return returns
    
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

        
    def train_multi_agent(self, env, num_episodes):
        start = datetime.now()

        max_reward = float("-inf")
        episodes = []
        rewards_ = []
        

        for i in range(num_episodes):
            state = env.reset()
            done = False

            size = self.num_agnets
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            

            while not done:
                ma_actions = []
                for agent_index in range(size):
                    ma_action = self.get_action(agent_index, state=state)
                    ma_actions.append(ma_action)

                next_state, reward, done = env.step(ma_actions)
                states.append(state)
                actions.append(ma_actions) # 二维数组
                rewards.append(reward) # 返回的reward是一个数组
                next_states.append(next_state)
                dones.append(done)

                state = next_state
            
            #update到底是每次step后就执行还是一个回合后就执行？
            #如果一个回合有限step可以在一个回合后执行，如果无限，则可以每次step更新
            for idx in range(size):
                self.update(idx, states, actions, rewards, next_states, dones, 0.99)
            
            total_reward = np.sum(rewards)
            episodes.append(i)
            rewards_.append(total_reward)

            if total_reward > max_reward:
                max_reward = total_reward
                self.save()
            
            print(f"training process: {(i/num_episodes):.2%}")
        end = datetime.now()
        run_time = end - start
        print("running time: %s "%run_time)
        # show the result
        plt.plot(episodes, rewards_, color = 'r')
        plt.xlabel("iterations")
        plt.ylabel("reward")
        plt.title("Multi-Agent training process")
        plt.show()
    def test(self):
        cfg = Config.config()
        env = Multi_Agent_Env(cfg=cfg)
        self.load()
        state = env.reset()
        done = False
        size = self.num_agnets

        while not done:
            ma_actions = []
            for agent_index in range(size):
                ma_action = self.get_action(agent_index, state=state)
                ma_actions.append(ma_action)
            next_state, reward, done = env.step(ma_actions)
            state = next_state
        env.energyplus.stop()
        print(env.total_reward, env.total_energy, env.total_temp_penalty)

if __name__ == "__main__":
    cfg = Config.config()
    env = Multi_Agent_Env(cfg=cfg)
    agents = A2C_MA(5, 5, env.action_space_size)
    agents.train_multi_agent(env, 200)

