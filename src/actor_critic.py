import Config
import actor
import critic
import torch
import numpy as np
from torch.nn import functional as F
class actor_critic:
    def __init__(self, n_states, n_actions, cfg:Config.config):
        self.actor = actor.actor(n_states,n_actions)
        self.critic = critic.critic(n_states)
       # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor.lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic.lr)
        self.gamma = cfg.gamma

    def take_action(self, state):
        state = torch.tensor(state[np.newaxism :])
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample().item()
        return action
    
    # 模型更新
    def update(self, transition_dict):
        # 训练集
        state = torch.tensor(transition_dict['state'], dtype=torch.float)
        action = torch.tensor(transition_dict['action'])
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float)
        done = torch.tensor(transition_dict['done'], dtype=torch.float)

        # 预测的当前时刻的state_value
        td_value = self.critic(state)
        # 目标的当前时刻的state_value
        td_target = reward + self.gamma * self.critic(next_state) * (1-int(done))
        # 时序差分的误差计算，目标的state_value与预测的state_value之差
        td_error = td_target - td_value
        
        # 对每个状态对应的动作价值用log函数
        # log_probs = torch.log(self.actor(state).gather(1, action))

        # 优化器梯度清0
        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        self.critic_optimizer.zero_grad()  # 价值网络的优化器

        # 策略梯度损失
        actor_loss = -torch.log(self.actor(state)) * td_error.detach()
        # 值函数损失，预测值和目标值之间
        critic_loss =  td_error.pow(2)

        # 反向传播
        actor_loss.backward()
        critic_loss.backward()
        # 参数更新
        self.actor_optimizer.step()
        self.critic_optimizer.step()
    