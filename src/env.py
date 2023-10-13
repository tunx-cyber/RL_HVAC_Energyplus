import Config

# in zone i at slot t
indoor_temp = [[]]

# in zone i at slot t
occupants_num = [[]]

# at slot t
outdoor_temp = 1

# air supply rate of zone i at slot t
air_supply_rate_set = [i*0.1*450 for i in range(11)]

# thermal disturbance in zone i at slot t
thermal_disturbance = [[]]

# CO2 concentration in zone i at slot t
CO2 = [[]]

# CO2 mixed
CO2_mix = [[]]

# outside CO2
CO2_outside = [[]]

# the damper position of AHU
damper_position_set = [i*0.1 for i in range(11)]

# electricity price
price = []

# mixed air temperature
mixed_air_temp = []

# power consumption of coil
coil_pow_consume = []

class env:
    def __init__(self,cfg:Config.config):
        self.cfg = cfg
        pass

    def supply_fan_cost(self, curr_air_supply_rate_set, curr_price):
        sum = 0
        for i in range(self.cfg.N):
            sum += curr_air_supply_rate_set[i] ** 3
        
        return self.cfg.fan_consume_coefficient * sum * curr_price * self.cfg.time_slot
    
    def cooling_coil_cost(self, curr_air_supply_rate_set, curr_price, damper_pos,
        outdoor_temp, indoor_temp_set):
        sum = 0
        for i in range(self.cfg.N):
            sum += curr_air_supply_rate_set[i] * self.cfg.air_specific_heat *(
                damper_pos * indoor_temp_set[i] + (1 - damper_pos) * outdoor_temp - self.cfg.supply_air_temp
                ) / (self.cfg.cooling_coil_efficiency * self.cfg.COP)
        
        return sum * curr_price * self.cfg.time_slot
    

    def reward_part_1(self, supply_fan_cost_):
        return -supply_fan_cost_ / self.cfg.N
    
    def reward_part_2(self, curr_air_supply_rate, damper_pos, 
                      indoor_temp, outdoor_temp, curr_price):
        
        part_2 = -self.cfg.N / (self.cfg.N + 1) * curr_air_supply_rate * self.cfg.air_specific_heat / (
            self.cfg.cooling_coil_efficiency * self.cfg.COP) * \
        (damper_pos * indoor_temp + (1 - damper_pos) * outdoor_temp - self.cfg.supply_air_temp) * \
        curr_price * self.cfg.time_slot
        return part_2

    def reward_part_3(self, indoor_temp, occupant_num):
        part_3 = 0
        if self.cfg.T_MIN <= indoor_temp <= self.cfg.T_MAX and occupant_num > 0:
            part_3 = 0
        elif indoor_temp > self.cfg.T_MAX and occupant_num > 0:
            part_3 = -(indoor_temp - self.cfg.T_MAX)
        elif indoor_temp < self.cfg.T_MIN and occupant_num > 0:
            part_3 = -(self.cfg.T_MIN - indoor_temp)
        else:
            part_3 = 0
        return part_3

    def reward_part_4(self, occupant_num, curr_CO2):
        part_4 = 0
        if occupant_num ==0:
            return part_4
        elif curr_CO2 >= self.cfg.CO2:
            part_4 = -(curr_CO2 - self.cfg.CO2) * self.cfg.N / (self.cfg.N + 1)
        else:
            part_4 = 0
        return part_4
    
    # for one agent
    def reward(self,supply_fan_cost_, curr_air_supply_rate, 
               indoor_temp, outdoor_temp, damper_pos, curr_price,
               occupant_num, curr_CO2,):
        # energy consumption
        part_1 = self.reward_part_1(self,supply_fan_cost_)
        # energy consumption of the cooling coil
        part_2 = self.reward_part_2(curr_air_supply_rate, damper_pos, indoor_temp, outdoor_temp, curr_price)
        # comfortable temperature range at slot t
        part_3 = self.reward_part_3(indoor_temp, occupant_num)
        # CO2 concentration violation
        part_4 = self.reward_part_4(occupant_num, curr_CO2)

        alpha = 0.1
        beta = 0.2
        return alpha * part_1 + part_2 + part_3 + beta * part_4
    
    def reward_last(self, curr_air_supply_rate_set, curr_price, damper_pos,
        outdoor_temp, indoor_temp_set, CO2_set, occupant_set):
        part_1 = 0
        part_2 = self.coil_pow_consume(curr_air_supply_rate_set, curr_price, damper_pos,
        outdoor_temp, indoor_temp_set)
        part_2 = -part_2/(self.cfg.N + 1)
        part_3 = 0

        part_4 = 0
        for i in range(self.cfg.N):
            temp = 0
            if occupant_set[i] > 0 :
                if CO2_set[i] > self.cfg.CO2:
                    temp = -(CO2_set[i] - self.cfg.CO2)
            part_4 += temp
        part_4 = part_4 / (self.cfg.N + 1)

        alpha = 0.1
        beta = 0.2
        return alpha * part_1 + part_2 + part_3 + beta * part_4

    def step(self, action):
        next_state = 1
        reward = 1
        done = False
        return next_state, reward, done
    

    def reset(self):
        pass