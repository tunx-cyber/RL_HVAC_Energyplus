# energyplus library
import sys
sys.path.insert(0,r"C:\EnergyPlusV23-1-0")
from pyenergyplus.api import EnergyPlusAPI
from pyenergyplus.datatransfer import DataExchange

import numpy as np
import csv

import threading

from queue import Queue, Empty, Full
from typing import Dict, Any, Tuple, Optional, List

idf_file = "./resource/HVACTemplate-5ZoneVAVFanPowered.idf"
epw_file = "./resource/USA_CO_Golden-NREL.724666_TMY3.epw"
idd_file = r"C:\EnergyPlusV23-1-0\Energy+.idd"

# data = []
class EnergyPlus:
    '''
    obs_queue是存放观察值的
    act_queue是存放动作值的
    action_space是动作空间
    get_action_func就是如何根据神经网络或者其他规则获取action_apace里面的值
    '''
    def __init__(self,obs_queue: Queue = Queue(1), act_queue: Queue = Queue(1), action_space = None, get_action_func = None) -> None:

        # for RL
        self.obs_queue = obs_queue
        self.act_queue = act_queue

        # for energyplus
        self.energyplus_api = EnergyPlusAPI()
        self.dx: DataExchange = self.energyplus_api.exchange
        self.energyplus_exec_thread = None

        # energyplus running states
        self.energyplus_state = None
        self.initialized = False
        self.simulation_complete = False
        self.warmup_complete = False
        self.warmup_queue = Queue()
        self.progress_value: int = 0
        self.sim_results: Dict[str, Any] = {}

        # request variables to be available during runtime
        self.request_variable_complete = False

        # get the variable names csv
        self.has_csv = False
        
        # variables, meters, actuators
        # look up in the csv file that get_available_data_csv() generate
        # or look up the html file
        '''
        space1-1 都是idf文件里面自定义的名字
        html文件里面也有，可以一个一个试
        csv文件里面也有
        '''
        # variables
        self.variables = {
            "zone_air_temp_1" : ("Zone Air Temperature","SPACE1-1"),
            "zone_air_temp_2" : ("Zone Air Temperature","SPACE2-1"),
            "zone_air_temp_3" : ("Zone Air Temperature","SPACE3-1"),
            "zone_air_temp_4" : ("Zone Air Temperature","SPACE4-1"),
            "zone_air_temp_5" : ("Zone Air Temperature","SPACE5-1"),
            "people_1" : ("Zone People Occupant Count", "SPACE1-1"),
            "people_2" : ("Zone People Occupant Count", "SPACE2-1"),
            "people_3" : ("Zone People Occupant Count", "SPACE3-1"),
            "people_4" : ("Zone People Occupant Count", "SPACE4-1"),
            "people_5" : ("Zone People Occupant Count", "SPACE5-1"),
            'outdoor_air_drybulb_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
            # "damper_pos": ("Zone Air Terminal VAV Damper Position","SPACE5-1 VAV REHEAT")
        }
        # Heating Coil NaturalGas Energy
        # Cooling Coil Electricity Energy
        self.var_handles: Dict[str, int] = {}

        # meters
        self.meters = {
            # "transfer_cool_1" : "Cooling:EnergyTransfer:Zone:SPACE1-1",
            # "transfer_heat_1" : "Heating:EnergyTransfer:Zone:SPACE1-1",
            # "transfer_cool_2" : "Cooling:EnergyTransfer:Zone:SPACE2-1",
            # "transfer_heat_2" : "Heating:EnergyTransfer:Zone:SPACE2-1",
            # "transfer_cool_3" : "Cooling:EnergyTransfer:Zone:SPACE3-1",
            # "transfer_heat_3" : "Heating:EnergyTransfer:Zone:SPACE3-1",
            # "transfer_cool_4" : "Cooling:EnergyTransfer:Zone:SPACE4-1",
            # "transfer_heat_4" : "Heating:EnergyTransfer:Zone:SPACE4-1",
            # "transfer_cool_5" : "Cooling:EnergyTransfer:Zone:SPACE5-1",
            # "transfer_heat_5" : "Heating:EnergyTransfer:Zone:SPACE5-1",

            # https://unmethours.com/question/55005/hvac-energy-consumption/
            "elec_hvac": "Electricity:HVAC",
            # "elec_heating" : "Heating:Electricity",
            "elec_cooling" : "Cooling:Electricity",
            # "gas_heating" : "Heating:NaturalGas"
        }
        self.meter_handles: Dict[str, int] = {}

        # actuators
        self.actuators = {
            "cooling_1" :(
                "Zone Temperature Control",
                "Cooling Setpoint",
                "SPACE1-1"
            ),
            "heating_1" :(
                "Zone Temperature Control",
                "Heating Setpoint",
                "SPACE1-1"
            ),
            "cooling_2" :(
                "Zone Temperature Control",
                "Cooling Setpoint",
                "SPACE2-1"
            ),
            "heating_2" :(
                "Zone Temperature Control",
                "Heating Setpoint",
                "SPACE2-1"
            ),
            "cooling_3" :(
                "Zone Temperature Control",
                "Cooling Setpoint",
                "SPACE3-1"
            ),
            "heating_3" :(
                "Zone Temperature Control",
                "Heating Setpoint",
                "SPACE3-1"
            ),
            "cooling_4" :(
                "Zone Temperature Control",
                "Cooling Setpoint",
                "SPACE4-1"
            ),
            "heating_4" :(
                "Zone Temperature Control",
                "Heating Setpoint",
                "SPACE4-1"
            ),
            "cooling_5" :(
                "Zone Temperature Control",
                "Cooling Setpoint",
                "SPACE5-1"
            ),
            "heating_5" :(
                "Zone Temperature Control",
                "Heating Setpoint",
                "SPACE5-1"
            ),
            # "vol_heat_rate_1":(
            #     "Sizing:Zone",
            #     "Zone Design Heating Vol Flow",
            #     "SPACE1-1"
            # ),
            # "vol_cool_rate_1":(
            #     "Sizing:Zone",
            #     "Zone Design Cooling Vol Flow",
            #     "SPACE1-1"
            # ),
            # "vol_heat_rate_2":(
            #     "Sizing:Zone",
            #     "Zone Design Heating Vol Flow",
            #     "SPACE2-1"
            # ),
            # "vol_cool_rate_2":(
            #     "Sizing:Zone",
            #     "Zone Design Cooling Vol Flow",
            #     "SPACE2-1"
            # ),
            # "vol_heat_rate_3":(
            #     "Sizing:Zone",
            #     "Zone Design Heating Vol Flow",
            #     "SPACE3-1"
            # ),
            # "vol_cool_rate_3":(
            #     "Sizing:Zone",
            #     "Zone Design Cooling Vol Flow",
            #     "SPACE3-1"
            # ),
            # "vol_heat_rate_4":(
            #     "Sizing:Zone",
            #     "Zone Design Heating Vol Flow",
            #     "SPACE4-1"
            # ),
            # "vol_cool_rate_4":(
            #     "Sizing:Zone",
            #     "Zone Design Cooling Vol Flow",
            #     "SPACE4-1"
            # ),
            # "vol_heat_rate_5":(
            #     "Sizing:Zone",
            #     "Zone Design Heating Vol Flow",
            #     "SPACE5-1"
            # ),
            # "vol_cool_rate_5":(
            #     "Sizing:Zone",
            #     "Zone Design Cooling Vol Flow",
            #     "SPACE5-1"
            # ),           

        }
        self.actuator_handles: Dict[str, int] = {}

        self.action_space = action_space
        self.action_space_size = 0
        if self.action_space is not None :
            self.action_space_size = len(self.action_space)
        self.get_action_func = get_action_func

    
    def start(self, suffix = "defalut"):
        self.energyplus_state = self.energyplus_api.state_manager.new_state()
        runtime = self.energyplus_api.runtime

        # request the variable
        if not self.request_variable_complete:
            for key, var in self.variables.items():
                self.dx.request_variable(self.energyplus_state, var[0], var[1])
                self.request_variable_complete = True

        # register callback used to track simulation progress
        def report_progress(progress: int) -> None:
            self.progress_value = progress

        runtime.callback_progress(self.energyplus_state, report_progress)

        # register callback used to signal warmup complete
        def _warmup_complete(state: Any) -> None:
            self.warmup_complete = True
            self.warmup_queue.put(True)

        runtime.callback_after_new_environment_warmup_complete(self.energyplus_state, _warmup_complete)

        # register callback used to collect observations and send actions
        runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)

        # register callback used to send actions
        runtime.callback_after_predictor_after_hvac_managers(self.energyplus_state, self._send_actions)

        # run EnergyPlus in a non-blocking way
        def _run_energyplus(runtime, cmd_args, state, results):
            # print(f"running EnergyPlus with args: {cmd_args}")
            self.energyplus_api.runtime.set_console_output_status(state=state,print_output=False)
            # start simulation
            results["exit_code"] = runtime.run_energyplus(state, cmd_args)
            
        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=(
                self.energyplus_api.runtime,
                self.make_eplus_args(suffix),
                self.energyplus_state,
                self.sim_results
            )
        )
        self.energyplus_exec_thread.start()


    def stop(self) -> None:
        if self.energyplus_exec_thread:
            self.simulation_complete = True
            self._flush_queues()
            self.energyplus_exec_thread.join()
            self.energyplus_exec_thread = None
            self.energyplus_api.runtime.clear_callbacks()
            self.energyplus_api.state_manager.delete_state(self.energyplus_state)

    def _collect_obs(self, state_argument):
        if self.simulation_complete or not self._init_callback(state_argument):
            return

        self.next_obs = {
            **{
                key: self.dx.get_variable_value(state_argument, handle)
                for key, handle in self.var_handles.items()
            }
        }
        # add the meters such as electricity
        for key, handle in self.meter_handles.items():
            self.next_obs[key] = self.dx.get_meter_value(state_argument,handle)
        # if full, it will block the entire simulation
        # print(f"obs: {self.next_obs}")
        self.obs_queue.put(self.next_obs) 
        
    def _send_actions(self, state_argument):
        if self.simulation_complete or not self._init_callback(state_argument):
            return 
        if self.act_queue.empty():
            return
        
        # softmax后是给的是一个投票，是index
        action_idx = self.act_queue.get()
        # print(action_idx)
        actions = self.get_action_func(self.action_space, action_idx)

        for i in range(len(self.actuator_handles)):
            # Effective heating set-point higher than effective cooling set-point err
            self.dx.set_actuator_value(
                state=state_argument,
                actuator_handle=list(self.actuator_handles.values())[i],
                actuator_value=actions[i]
            )
            
        
    def _flush_queues(self):
        for q in [self.obs_queue, self.act_queue]:
            while not q.empty():
                q.get()


    def make_eplus_args(self, suffix = "default"):
        args = [
            "-i",
            idd_file,
            "-w",
            epw_file,
            "-d",
            "res",
            "-p",
            suffix,
            "-x",
            "-r",
            idf_file,
        ]
        return args
    
    def _init_callback(self, state_argument) -> bool:
        """initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.initialized = self._init_handles(state_argument) \
            and not self.dx.warmup_flag(state_argument)
        return self.initialized

    def _init_handles(self, state_argument):
        """initialize sensors/actuators handles to interact with during simulation"""
        if not self.initialized:
            if not self.dx.api_data_fully_ready(state_argument):
                return False
        
            # store the handles so that we do not need get the hand every callback
            self.var_handles = {
                key: self.dx.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: self.dx.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: self.dx.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }

            for handles in [
                self.var_handles,
                self.meter_handles,
                self.actuator_handles
            ]:
                if any([v == -1 for v in handles.values()]):
                    print("Error! there is -1 in handle! check the variable names in the var.csv")

                    print("variables:")
                    for k in self.var_handles :
                        print(self.var_handles[k])

                    print("meters:")
                    for k in self.meter_handles :
                        print(self.meter_handles[k])

                    print("actuators")
                    for k in self.actuator_handles:
                        print(k)
                    
                    self.get_available_data_csv(state_argument)
                    exit(1)

            self.initialized = True

        return True
    
    # get the name and key for handles
    def get_available_data_csv(self, state):
        if self.has_csv:
            return
        else:
            available_data = self.dx.list_available_api_data_csv(self.energyplus_state).decode("utf-8")
            lines = available_data.split('\n')
            with open("var.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for line in lines:
                    fields = line.split(',')
                    writer.writerow(fields)

            self.has_csv = True
    
    def failed(self) -> bool:
        return self.sim_results.get("exit_code", -1) > 0
