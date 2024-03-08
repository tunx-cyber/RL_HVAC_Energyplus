# load model
import a2c
import torch

class A2C_Test:
    def run(self):
        cfg = a2c.config()
        env = a2c.EnergyplusEnv.EnergyPlusEnvironment(cfg=cfg)
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
            action = agent.select_action(state) 
            next_state, reward, done = env.step(action)
            state = next_state

        # get result
        return env.total_reward, env.total_energy, env.temp_penalty

# TODO multi agent
# TODO problem in solution space
# TODO problem in a2c agent
# TODO when there is nobody and it is cooling day.
# TODO share paramters or attention machanism or 异形agent
    
a2c_test = A2C_Test()
a2c_res = a2c_test.run()

import RuleBased
rule_test = RuleBased.RuleBased()
rule_test.run()
rule_test.ep.energyplus_exec_thread.join()


print("ac2 result: ",a2c_res)
print("rule based result: ", rule_test.get_result())

