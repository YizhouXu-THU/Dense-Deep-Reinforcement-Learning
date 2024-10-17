import os
import sys
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import traci
from sumolib import checkBinary

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class SUMO_env_replay:
    def __init__(self, sumo_cfg, gui=True, seed=42):
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        sumoCmd = [sumoBinary, '--start', '-c', sumo_cfg]
        sumoCmd += ['--seed', str(seed)]
        sumoCmd += ['--collision.action', 'none']
        traci.start(sumoCmd)
        
        self.delta_t = traci.simulation.getDeltaT()
        
        for light in traci.trafficlight.getIDList():
            traci.trafficlight.setProgram(light, 'off')
    
    def reset(self, scenario_data):
        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.remove(veh_id)
        
        self.scenario_data = scenario_data
        self.veh_list = list(scenario_data.keys())
        self.veh_num = len(self.veh_list)
        self.step_num = len(scenario_data[self.veh_list[0]]['trajs_mask'])
        
        for veh_id, veh_info in scenario_data.items():
            if veh_info['trajs_mask'][0] > 0:
                x, y = veh_info['trajs'][0, 0], veh_info['trajs'][0, 1]
                heading = veh_info['trajs'][0, 3]
                
                traci.vehicle.add(veh_id, routeID='', typeID=veh_info['vType'], departSpeed=0)
                traci.vehicle.setLaneChangeMode(veh_id, 0b000000000000)
                traci.vehicle.setSpeedMode(veh_id, 0b100000)
                traci.vehicle.moveToXY(veh_id, edgeID='', lane=-1, x=x, y=y, angle=heading, keepRoute=2)
                
                if veh_id == 'CAV':
                    traci.vehicle.setColor(veh_id, (255, 0, 0))
                elif veh_info['is_pov'][0] > 0:
                    traci.vehicle.setColor(veh_id, (0, 0, 255))
                else:
                    traci.vehicle.setColor(veh_id, (0, 255, 0))
        
        if 'CAV' in scenario_data:
            traci.gui.trackVehicle('View #0', 'CAV')
            traci.gui.setZoom('View #0', 500)
        
        traci.simulationStep()
    
    def step(self, t):
        for veh_id, veh_info in self.scenario_data.items():
            if veh_id in traci.vehicle.getIDList():
                if veh_info['trajs_mask'][t] > 0:
                    x, y = veh_info['trajs'][t, 0], veh_info['trajs'][t, 1]
                    heading = veh_info['trajs'][t, 3]
                    traci.vehicle.moveToXY(veh_id, edgeID='', lane=-1, x=x, y=y, angle=heading, keepRoute=2)
                    
                    if veh_id == 'CAV':
                        traci.vehicle.setColor(veh_id, (255, 0, 0))
                    elif veh_info['is_pov'][t] > 0:
                        traci.vehicle.setColor(veh_id, (0, 0, 255))
                    else:
                        traci.vehicle.setColor(veh_id, (0, 255, 0))
                else:
                    traci.vehicle.remove(veh_id)
        traci.simulationStep()
    
    def close(self):
        traci.close()

data_path = './data_analysis/raw_data/Experiment-0927_2024-09-27/traj_data/singapore-onenorth/'
scenario_list = os.listdir(data_path)
map_name = data_path.split('/')[-2]
map_path = os.path.join('./maps', map_name)

env = SUMO_env_replay(sumo_cfg=os.path.join(map_path, map_name+'.sumocfg'), gui=True)
for scenario in scenario_list:
    print(scenario)
    scenario_data = pickle.load(open(os.path.join(data_path, scenario), 'rb'))
    env.reset(scenario_data)
    for t in range(1, env.step_num):
        env.step(t)
env.close()
