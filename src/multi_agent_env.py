from Energyplus import EnergyPlus
from Energyplus import Queue
import Config
from Energyplus import np
from Energyplus import Full, Empty
from matplotlib import pyplot as plt
a = np.linspace(19,24,12)
def get_action_fun(acts, idxs):
    real_act = []
    for i in idxs:
        real_act.append(acts[i]) 
        real_act.append(acts[i]-3) 
    return real_act

class MAEnergyPlus(EnergyPlus):
    def __init__(self,obs_queue: Queue = Queue(1), act_queue: Queue = Queue(1),
                 action_space = None, get_action_func =None) -> None:
        
        super().__init__(obs_queue, act_queue, action_space, get_action_func)
        
       
class Multi_Agent_Env:
    def __init__(self, cfg: Config.config) -> None:
        self.cfg = cfg

        self.timestep = 0

        self.last_obs = {}
        self.obs_queue : Queue= None # this queue and the energyplus's queue is the same obj
        self.act_queue : Queue= None # this queue and the energyplus's queue is the same obj
        self.energyplus: MAEnergyPlus= MAEnergyPlus(None,None,a,get_action_fun)

        # observation space is a two dimentional array
        # the firt is the action variable
        # the second is the action avaliable values
        self.observation_space_size = len(self.energyplus.variables) + len(self.energyplus.meters)
        # choose one value from every variable
        self.action_space_size = self.energyplus.action_space_size

        self.total_energy = 0
        self.total_temp_penalty = 0
        self.total_reward = 0

        self.temps_name = ["zone_air_temp_"+str(i+1) for i in range(5)]
        self.occups_name = ["people_"+str(i+1) for i in range(5)]

        #get the indoor/outdoor temperature series
        self.indoor_temps = []
        self.outdoor_temp = []
        #get the setpoint series
        self.setpoints = []
        #get the energy series
        self.energy = []

    # return a first observation
    def reset(self, file_suffix = "defalut"):
        self.energyplus.stop()
        self.total_temp_penalty = 0
        self.total_energy = 0
        self.total_reward = 0
        self.indoor_temps.clear()
        self.outdoor_temp.clear()
        self.setpoints.clear()
        self.energy.clear()

        if self.energyplus is not None:
            self.energyplus.stop()
        
        self.obs_queue = Queue(maxsize = 1)
        self.act_queue = Queue(maxsize = 1)

        self.energyplus = MAEnergyPlus(
            obs_queue = self.obs_queue,
            act_queue = self.act_queue,
            action_space=a,
            get_action_func=get_action_fun
        )

        self.energyplus.start(file_suffix)

        obs = self.obs_queue.get()
        obs_value = list(obs.values())
        agent_obs_vec = []
        for i in range(5):
            single_obs = [obs_value[i], obs_value[i+5], obs_value[10], obs_value[11], obs_value[12]]
            agent_obs_vec.append(single_obs)

        self.last_obs = obs
        self.indoor_temps.append([obs[x] for x in self.temps_name])
        self.outdoor_temp.append(obs["outdoor_air_drybulb_temperature"])
        return np.array(agent_obs_vec)
    
    # predict next observation
    # mutil agents' actions
    # return each agent's reward
    def step(self, actions):
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
                self.setpoints.append(get_action_fun(a,actions))
                self.act_queue.put(actions,timeout=timeout)

                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                self.indoor_temps.append([obs[x] for x in self.temps_name])
                self.outdoor_temp.append(obs["outdoor_air_drybulb_temperature"])

            except(Full, Empty):
                done = True
                obs = self.last_obs
        
        reward = self.get_reward() 

        obs_value = np.array(list(obs.values()))
        agent_obs_vec = []
        for i in range(5):
            single_obs = [obs_value[i], obs_value[i+5], obs_value[10], obs_value[11], obs_value[12]]
            agent_obs_vec.append(single_obs)
        return agent_obs_vec, reward, done

    def get_reward(self):

        # according to the meters and variables to compute
        obs = self.last_obs

        temps = ["zone_air_temp_"+str(i+1) for i in range(5)]
        occups = ["people_"+str(i+1) for i in range(5)]
        temps_vals = []
        occups_vals = []
        for temp in temps:
            temps_vals.append(obs[temp])
        for occup in occups:
            occups_vals.append(obs[occup])

        n = len(temps_vals)
        ma_rewards = []
        ma_temp_reward = []

        # TODO find a good function to evaluate the temperature reward
        for i in range(n):
            if occups_vals[i] <= 1:
                ma_temp_reward.append(0)

            elif self.cfg.T_MIN <= temps_vals[i] <= self.cfg.T_MAX:
                ma_temp_reward.append(0)

            elif temps_vals[i] < self.cfg.T_MIN :
                ma_temp_reward.append(-1)

            else:
                ma_temp_reward.append(-1)
        
        # energy reward
        energy = obs["elec_cooling"] / 3600000
        self.energy.append(energy)
        energy_reward = - energy
        
        for idx in range(n):
            ma_rewards.append(ma_temp_reward[idx] * 0.1 + 0.9*energy_reward / n )
        
        self.total_energy += energy

        self.total_temp_penalty += sum(ma_temp_reward)

        self.total_reward += sum(ma_temp_reward) * 0.1 + 0.9*energy_reward #0.1 0.9效果比较好

        return ma_rewards
    
    def render(self):
        #get the indoor/outdoor temperature series
        zone_temp = []
        for i in range(5):
            zone_temp.append(np.array(self.indoor_temps)[:,i])
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