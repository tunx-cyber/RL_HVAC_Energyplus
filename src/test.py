# load model
import a2c
import torch

cfg = a2c.config()
env = a2c.EnergyPlusEnvironment(cfg=cfg)
# 定义环境和代理参数
state_dim = env.observation_space_size  # 状态维度
action_dim = env.action_space_size  # 动作维度
lr = 0.001  # 学习率
gamma = 0.99  # 折扣因子
num_episodes = 200  # 训练的总回合数

agent = a2c.A2CAgent(state_dim, action_dim, lr=lr, gamma=gamma)

agent.actor.load_state_dict(torch.load("actor.pth"))
agent.critic.load_state_dict(torch.load("critic.pth"))

# run model
state = env.reset("test")
done = False

states, actions, rewards, next_states, dones = [], [], [], [], []

while not done:
    action = agent.select_action(state) # 这里改为多个agent选择
                                        # 相应的所有的next_state action要单独一个agent，不过reward和done是通用的，需要适当修改
    next_state, reward, done = env.step(action)

    state = next_state

# get result
print(env.total_reward,env.total_energy, env.temp_penalty)

# TODO multi agent
# TODO problem in solution space
# 