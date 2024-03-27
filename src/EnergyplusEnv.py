import Energyplus
from queue import Queue, Full, Empty
import numpy as np
import Config
import itertools
import matplotlib.pyplot as plt
a = np.linspace(19,24,6)
actions = list(itertools.product(a, repeat=5))

def get_action_f(action_space, action_idx):
    real_act = []
    a = action_space[action_idx]
    for i in range(len(a)):
        real_act.append(a[i])
        real_act.append(a[i]-3)
    return real_act

class EnergyPlusEnvironment:
    def __init__(self, cfg: Config.config) -> None:
        self.cfg = cfg
        self.episode = -1
        self.timestep = 0

        self.last_obs = {}
        self.obs_queue : Queue= None # this queue and the energyplus's queue is the same obj
        self.act_queue : Queue= None # this queue and the energyplus's queue is the same obj
        self.energyplus: Energyplus.EnergyPlus= Energyplus.EnergyPlus(None,None,actions,get_action_f)

        # observation space is a two dimentional array
        # the firt is the action variable
        # the second is the action avaliable values
        self.observation_space_size = len(self.energyplus.variables) + len(self.energyplus.meters)
        self.action_space = actions
        # choose one value from every variable
        self.action_space_size = len(self.action_space)

        self.temps_name = ["zone_air_temp_"+str(i+1) for i in range(5)]
        self.occups_name = ["people_"+str(i+1) for i in range(5)]
        self.total_energy = 0
        self.total_temp_penalty = 0
        self.total_reward = 0

        #get the indoor/outdoor temperature series
        self.indoor_temps = []
        self.outdoor_temp = []
        #get the setpoint series
        self.setpoints = []
        #get the energy series
        self.energy = []
        #get the occupancy situation
        self.occup_count = []
    # return the first observation
    def reset(self, file_suffix = "defalut"):
        self.total_temp_penalty = 0
        self.total_energy = 0
        self.total_reward = 0
        self.indoor_temps.clear()
        self.outdoor_temp.clear()
        self.setpoints.clear()
        self.energy.clear()
        self.occup_count.clear()

        self.energyplus.stop()
        self.episode += 1
        # self.last_obs = self.sample()

        if self.energyplus is not None:
            self.energyplus.stop()
        
        self.obs_queue = Queue(maxsize = 1)
        self.act_queue = Queue(maxsize = 1)

        self.energyplus = Energyplus.EnergyPlus(
            obs_queue = self.obs_queue,
            act_queue = self.act_queue,
            action_space=self.action_space,
            get_action_func=get_action_f
        )

        self.energyplus.start(file_suffix)

        obs = self.obs_queue.get()
        self.last_obs = obs

        self.indoor_temps.append([obs[x] for x in self.temps_name])
        self.occup_count.append([obs[x] for x in self.occups_name])

        self.outdoor_temp.append(obs["outdoor_air_drybulb_temperature"])

        return np.array(list(obs.values()))
    
    # predict next observation
    def step(self, action):
        self.timestep += 1
        done = False

        if self.energyplus.failed():
            raise RuntimeError(f"E+ failed {self.energyplus.sim_results['exit_code']}")
        
        if self.energyplus.simulation_complete:
            done = True
            obs  = self.last_obs
        else:
            timeout = 3
            try:
                self.setpoints.append(get_action_f(actions,action))
                self.act_queue.put(action,timeout=timeout)
                
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                self.indoor_temps.append([obs[x] for x in self.temps_name])
                self.occup_count.append([obs[x] for x in self.occups_name])
                self.outdoor_temp.append(obs["outdoor_air_drybulb_temperature"])

            except(Full, Empty):
                done = True
                obs = self.last_obs
        
        reward = self.get_reward()
        obs_vec = np.array(list(obs.values()))

        return obs_vec, reward, done

    def get_reward(self):

        # according to the meters and variables to compute
        obs = self.last_obs
        
        # compute the temperature reward
        temp_reward = 0

        temps_vals = []
        occups_vals = []
        for temp in self.temps_name:
            temps_vals.append(obs[temp])
        for occup in self.occups_name:
            occups_vals.append(obs[occup])
        
        # TODO find a good function to evaluate the temperature reward
        # 520 oocups timesteps
        for i in range(len(temps_vals)):
            if occups_vals[i] < 1:
                temp_reward += 0
            elif self.cfg.T_MIN <= temps_vals[i] <= self.cfg.T_MAX:
                temp_reward += 0
            elif temps_vals[i] < self.cfg.T_MIN :
                temp_reward += -1
            else:
                temp_reward += -1
        
        # energy reward
        energy = obs["elec_cooling"] / 3600000
        self.energy.append(energy)
        energy_reward = - energy
        self.total_energy += energy

        # temperature reward
        self.total_temp_penalty += temp_reward

        # reward combination
        reward = temp_reward*0.1 + energy_reward*0.9
        
        self.total_reward += reward

        return reward
        

    def close(self):
        if self.energyplus is not None:
            self.energyplus.stop()
    
    def render(self):
        #get the indoor/outdoor temperature series
        zone_temp = []
        for i in range(5):
            zone_temp.append(np.array(self.indoor_temps)[:,i])
        
        # get occupancy
        zone_occupy = []
        for i in range(5):
            zone_occupy.append(np.array(self.occup_count)[:,i])
        #get the setpoint series
        sp_series = []
        for i in range(0,10,2):
            sp_series.append(np.array(self.setpoints)[:,i])
        #get the energy series
        x = range(len(self.setpoints))
        
        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("temperature (℃)")
            plt.plot(x,zone_temp[i],label=f"zone_{i+1}_temperature")
        plt.legend()
        plt.show()

        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("setpoint (℃)")
            plt.plot(x,sp_series[i],label=f"zone_{i+1}_setpoint")
        plt.legend()
        plt.show()
        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("occupancy")
            plt.plot(x,zone_occupy[i],label=f"zone_{i+1}_people_occupant_count ")
        plt.legend()
        plt.show()

        plt.plot(x,self.energy)
        plt.title("energy cost")
        plt.xlabel("timestep")
        plt.ylabel("energy cost (kwh)")
        plt.show()

        plt.plot(x, self.outdoor_temp)
        plt.title("outdoor temperature")
        plt.xlabel("timestep")
        plt.ylabel("temperature (℃)")
        plt.show()
