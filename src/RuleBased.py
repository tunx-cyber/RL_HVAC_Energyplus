import Energyplus
from queue import Queue, Full, Empty
import numpy as np
import Config
class RuleBased:
    def __init__(self) -> None:
        self.last_obs = {}
        self.obs_queue : Queue = None
        self.act_queue : Queue = None

        self.ep : Energyplus.EnergyPlus = Energyplus.EnergyPlus(None, None)
        self.cfg = Config.config()
        self.observation_space_size = len(self.energyplus.variables) + len(self.energyplus.meters)
        # choose one value from every variable
        self.action_space_size = self.energyplus.action_space_size

        self.reward = 0
        self.energy_cost = 0
        self.temp_penalty = 0

    
    def run(self):
        if self.ep is not None:
            self.ep.stop()
        
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)

        self.ep = Energyplus.EnergyPlus(
            obs_queue = self.obs_queue,
            act_queue = self.act_queue
        )
        self.ep.start("rule_base")
        
        done = False
        while(not done):
            if self.ep.failed():
                raise RuntimeError(f"E+ failed {self.ep.sim_results['exit_code']}")
            
            if self.ep.simulation_complete:
                done = True
                obs = self.last_obs
            else:
                timeout = 2
                try:
                    self.act_queue.put(self.get_action(list(self.last_obs.values())), timeout=timeout)
                    self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                except(Full, Empty):
                    done = True
                    obs = self.last_obs
            

        obs = self.obs_queue.get()
        self.last_obs = obs

    def compute_reward(self):
        obs = self.last_obs

        temp_reward = 0

        temps = ["zone_air_temp_"+str(i+1) for i in range(5)]
        occups = ["people_"+str(i+1) for i in range(5)]

        temps_vals = []
        occups_vals = []

        for temp in temps:
            temps_vals.append(obs[temp])
        for occup in occups:
            occups_vals.append(obs[occup])

        for i in range(len(temps_vals)):
            if occups_vals[i] <= 0.001:
                temp_reward += 0
            elif self.cfg.T_MIN <= temps_vals[i] <= self.cfg.T_MAX:
                temp_reward += 0
            elif temps_vals[i] < self.cfg.T_MIN :
                temp_reward += temps_vals[i] - self.cfg.T_MIN
            else:
                temp_reward += self.cfg.T_MAX - temps_vals[i]


        self.energy_cost += obs["elec_cooling"] + obs["gas_heating"]

        self.temp_penalty += temp_reward
        
        elec_reward = - self.energy_cos/1000

        self.reward = temp_reward*1000 + elec_reward

    def get_action(self, obs_vec):

        action_val = []
        obs = self.last_obs
        temps = ["zone_air_temp_"+str(i+1) for i in range(5)]
        occups = ["people_"+str(i+1) for i in range(5)]

        temps_vals = []
        occups_vals = []

        for temp in temps:
            temps_vals.append(obs[temp])
        for occup in occups:
            occups_vals.append(obs[occup])
        
        # TODO 这里要对应温度值设定的规则 用idx
        for i in range(len(temps_vals)):
            if occups_vals[i] <= 0.001:
                action_val.append(temps_vals[i])
            elif self.cfg.T_MIN <= temps_vals[i] <= self.cfg.T_MAX:
                action_val.append(temps_vals[i])
            elif temps_vals[i] < self.cfg.T_MIN :
                action_val.append(temps_vals[i] + 0.5)
            else:
                action_val.append(temps_vals[i] - 0.5)
        
        return action_val

        

    def get_result(self):
        return self.reward, self.energy_cost, self.temp_penalty

rule_test = RuleBased()

rule_test.run()
print(rule_test.get_result())
