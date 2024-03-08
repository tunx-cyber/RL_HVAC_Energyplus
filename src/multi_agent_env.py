from Energyplus import EnergyPlus
from Energyplus import Queue
import Config
from Energyplus import np
from Energyplus import Full, Empty
a = np.linspace(17,26,10)
def get_action_fun(acts, idxs):
    real_act = []
    for i in idxs:
        real_act.append(acts[i]) 
        real_act.append(acts[i]) 
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

    # return a first observation
    def reset(self, file_suffix = "defalut"):
        self.energyplus.stop()
        

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
        self.last_obs = obs
        return np.array(list(obs.values()))
    
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
                self.act_queue.put(actions,timeout=timeout)
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
            except(Full, Empty):
                done = True
                obs = self.last_obs
        
        reward = self.get_reward() 

        obs_vec = np.array(list(obs.values()))

        return obs_vec, reward, done

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
            if occups_vals[i] <= 0.001:
                ma_temp_reward.append(0)

            elif self.cfg.T_MIN <= temps_vals[i] <= self.cfg.T_MAX:
                ma_temp_reward.append(1)

            elif temps_vals[i] < self.cfg.T_MIN :
                ma_temp_reward.append(-1)

            else:
                ma_temp_reward.append(-1)
        
        # energy reward
        energy = obs["elec_cooling"] / 3600000
        energy_reward = - energy
        
        for idx in range(n):
            ma_rewards.append(ma_temp_reward[idx] * 0.1 + energy_reward / n )
        
        self.total_energy += energy

        self.total_temp_penalty += sum(ma_temp_reward)

        self.total_reward += self.total_temp_penalty * 0.1 + energy_reward

        return ma_rewards