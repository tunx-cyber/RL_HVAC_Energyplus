import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# 定义A2C算法
class A2C:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.model = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        logits, _ = self.model(state)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def update(self, states, actions, rewards, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        logits, values = self.model(states)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)

        returns = []
        advantage = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                advantage = 0

            else:
                delta = rewards[t] + 0.99 * values[t + 1] - values[t]
                advantage = delta + 0.99 * 0.95 * advantage

            returns.insert(0, advantage + values[t])

        returns = torch.cat(returns)
        log_probs = m.log_prob(actions)
        actor_loss = -(log_probs * returns).mean()
        critic_loss = F.smooth_l1_loss(values.squeeze(), returns)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            states, actions, rewards, dones = [], [], [], []

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                state = next_state
                episode_reward += reward

            self.update(states, actions, rewards, dones)

            print(f"Episode {episode + 1}, Reward: {episode_reward}")

# 创建OpenAI Gym环境
env = gym.make('CartPole-v1')

# 创建A2C实例并训练
agent = A2C(env)
agent.train(num_episodes=100)