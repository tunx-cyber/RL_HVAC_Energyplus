# parameters and constants
import torch
class config:
    def __init__(self) -> None:
        # total number of zones
        self.N = 4
        # total number of time slots
        self.L = 24*60/15
        # max acceptable indoor temperature
        self.T_MAX = 24
        # min acceptable indoor temperature (C)
        self.T_MIN = 19
        # the set of neighbors of zone i
        self.neighbors={}
        # acceptable CO2 concentration (ppm)
        self.CO2 = 1300
        # time slot length (min)
        self.time_slot = 15
        # air density (g/m^3)
        self.air_density = 100
        # the volume of zone i
        self.zone_volume = []
        # the CO2 generation rate per person (L/s)
        self.CO2_generation_rate = 10
        # number of discreate level ralated to m(i,t)
        self.M = 10
        # number of dicraete levels related to damper position in AHU
        self.Z = 1
        # Fan power consumption coefficient
        self.fan_consume_coefficient = 2e-6
        # the specific heat of air (J/g/。C)
        self.air_specific_heat = 40
        # the efficiency factor of the cooling coil
        self.cooling_coil_efficiency = 0.8879
        # coefficiency factor of cooling coil
        self.COP = 5.9153
        # supply air temperature of the VFD fan
        self.supply_air_temp = 21
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # discount factor
        self.gamma = 0.995
        # target network update rate
        self.epsilon = 0.001
        # batch size
        self.batch_size = 128
        # buffer size
        self.buffer_size = 4800000