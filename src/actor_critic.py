import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# 定义A2C算法
class A2C:
    def __init__(self, env : gym.Env):
        self.env = env

        self.state_dim = env.observation_space.shape[0]
        # self.state_dim = env.observation_space_size
        self.action_dim = env.action_space.n

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits = self.actor(state)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算TD误差
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_targets = rewards + 0.99 * next_values * (1 - dones)
        td_errors = td_targets - values

        # 更新Critic网络参数
        critic_loss = td_errors.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络参数
        logits = self.actor(states)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        log_probs = m.log_prob(actions)
        actor_loss = -(log_probs * td_errors.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, num_episodes):
        x = []
        y = []
        for episode in range(num_episodes):
            # first observation
            state = self.env.reset()
            done = False
            episode_reward = 0

            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_reward += reward

            self.update(states, actions, rewards, next_states, dones)
            x.append(episode)
            y.append(episode_reward)
            print(f"Episode {episode + 1}, Reward: {episode_reward}")
        return x,y

# 创建OpenAI Gym环境
env = gym.make('CartPole-v1')

# 创建A2C实例并训练
agent = A2C(env)
x,y = agent.train(num_episodes=100)
import matplotlib.pyplot as plt
plt.plot(x,y,color = 'r')
plt.show()