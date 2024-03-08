# energyplus library
import Energyplus
import math
from queue import Queue, Full, Empty
import numpy as np
import Config
# data = []
class RBEnergyPlus(Energyplus.EnergyPlus):
    def __init__(self,obs_queue: Queue = Queue(1), act_queue: Queue = Queue(1), action_space = None, get_action_func = None) -> None:

        super().__init__(obs_queue, act_queue, action_space, get_action_func)

    def _send_actions(self, state_argument):
        if self.simulation_complete or not self._init_callback(state_argument):
            return 
        if self.act_queue.empty():
            return
        actions = self.act_queue.get()

        for i in range(len(self.actuator_handles)):
            # Effective heating set-point higher than effective cooling set-point err
            self.dx.set_actuator_value(
                state=state_argument,
                actuator_handle=list(self.actuator_handles.values())[i],
                actuator_value=actions[i]
            )

class RuleBased:
    def __init__(self) -> None:
        self.last_obs = {}
        self.obs_queue : Queue = None
        self.act_queue : Queue = None

        self.ep : RBEnergyPlus = RBEnergyPlus(None, None)
        self.cfg = Config.config()

        self.reward = 0
        self.energy_cost = 0
        self.temp_penalty = 0

    
    def run(self):
        if self.ep is not None:
            self.ep.stop()
        
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)

        self.ep = RBEnergyPlus(
            obs_queue = self.obs_queue,
            act_queue = self.act_queue
        )
        self.ep.start("rule_base")
        
        done = False
        while not done:
            if self.ep.failed():
                raise RuntimeError(f"E+ failed {self.ep.sim_results['exit_code']}")
            
            if self.ep.simulation_complete:
                done = True
                obs = self.last_obs
            else:
                timeout = 2
                try:
                    self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                    self.compute_reward()
                    action = self.get_action()
                    self.act_queue.put(action, timeout=timeout)
                except(Full, Empty):
                    done = True
                    obs = self.last_obs

            

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
                temp_reward += 1
            elif temps_vals[i] < self.cfg.T_MIN :
                temp_reward += -1
            else:
                temp_reward += -1


        self.energy_cost += obs["elec_cooling"] / 3600000

        self.temp_penalty += temp_reward
        
        energy_reward = - self.energy_cost

        self.reward += temp_reward*0.1 + energy_reward

    def get_action(self):

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
        
        for i in range(len(temps_vals)):
            if occups_vals[i] <= 0.001:
                action_val.append(math.floor( temps_vals[i]) )
                action_val.append(math.floor( temps_vals[i]) )
            elif self.cfg.T_MIN <= temps_vals[i] <= self.cfg.T_MAX:
                action_val.append(math.floor( temps_vals[i]) )
                action_val.append(math.floor( temps_vals[i]) )
            elif temps_vals[i] < self.cfg.T_MIN :
                action_val.append(math.floor(temps_vals[i] + 1) )
                action_val.append(math.floor(temps_vals[i] + 1) )
            else:
                action_val.append(math.floor(temps_vals[i] - 1) )
                action_val.append(math.floor(temps_vals[i] - 1) )
        
        return action_val     

    def get_result(self):
        return self.reward, self.energy_cost, self.temp_penalty
