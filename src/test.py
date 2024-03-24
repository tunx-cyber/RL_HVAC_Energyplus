# # load model
# import a2c
# import torch

# class A2C_Test:
#     def run(self):
#         cfg = a2c.config()
#         env = a2c.EnergyplusEnv.EnergyPlusEnvironment(cfg=cfg)
#         # 定义环境和代理参数
#         state_dim = env.observation_space_size  # 状态维度
#         action_dim = env.action_space_size  # 动作维度
#         lr = 0.001  # 学习率
#         gamma = 0.99  # 折扣因子

#         agent = a2c.A2CAgent(state_dim, action_dim, lr=lr, gamma=gamma)

#         agent.actor.load_state_dict(torch.load("actor.pth"))
#         agent.critic.load_state_dict(torch.load("critic.pth"))

#         # run model
#         state = env.reset("test")
#         done = False

#         while not done:
#             action = agent.select_action(state) 
#             next_state, reward, done = env.step(action)
#             state = next_state

#         # get result
#         return env.total_reward, env.total_energy, env.total_temp_penalty

    
# a2c_test = A2C_Test()
# a2c_res = a2c_test.run()

import re_a2c
cfg = re_a2c.Config.config()
env = re_a2c.EnergyplusEnv.EnergyPlusEnvironment(cfg)
actest = re_a2c.A2C(env.observation_space_size, env.action_space_size, 0.001, 0.99)
actest.test(env=env)
import RuleBased
rule_test = RuleBased.RuleBased()
rule_test.run()
rule_test.ep.energyplus_exec_thread.join()

print("rule based result: ", rule_test.get_result())
