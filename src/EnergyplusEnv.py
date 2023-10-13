import Energyplus
from queue import Queue, Full, Empty
import numpy as np
'''
Variable air volume (VAV) is a type of heating, ventilating, and/or air-conditioning (HVAC) system. 
Unlike constant air volume (CAV) systems, which supply a constant airflow at a variable temperature, 
VAV systems vary the airflow at a constant or varying temperature.[1][2] 
The advantages of VAV systems over constant-volume systems include more precise temperature control, 
reduced compressor wear, lower energy consumption by system fans, less fan noise, 
and additional passive dehumidification.[3]
'''

class EnergyPlusEnvironment():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.episode = -1
        self.timestep = 0

        self.last_obs = {}
        self.obs_queue = None
        self.act_queue = None
        self.energyplus: Energyplus.EnergyPlus= None
        self.obs_space = []

    def reset(self):
        self.episode += 1
        self.last_obs = self.sample()

        if self.energyplus is not None:
            self.energyplus.stop()
        
        self.obs_queue = Queue(maxsize = 1)
        self.act_queue = Queue(maxsize = 1)

        self.energyplus = Energyplus.EnergyPlus(
            obs_queue = self.obs_queue,
            act_queue = self.act_queue
        )

        self.energyplus.start()

        obs = self.obs_queue.get()
        self.last_obs = obs
        return np.array(list(obs.values())), {}
        
    def step(self, action):
        self.timestep += 1
        done = False

        if self.energyplus.failed():
            raise RuntimeError(f"E+ failed {self.energyplus.sim_results['exit_code']}")
        
        if self.energyplus.simulation_complete:
            done = True
            obs  = self.last_obs
        else:
            timeout = 2
            try:
                # TODO: use network to choose action
                action = None
                self.act_queue.put(action,timeout=timeout)
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
            except(Full, Empty):
                done = True
                obs = self.last_obs
        
        reward = self.get_reward()
        obs_vec = np.array(list(obs.values()))
        return obs_vec, reward, done, False, {}

    def get_reward(self):
        pass

    def sample(self):
        # random sample
        pass

    def close(self):
        if self.energyplus is not None:
            self.energyplus.stop()

        

