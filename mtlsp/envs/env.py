import os
import pickle
import numpy as np
import conf.conf as conf
from mtlsp.vehicle.vehicle import VehicleList, Vehicle
from abc import abstractmethod
from mtlsp.controller.vehicle_controller.controller import Controller
from mtlsp.simulator import Simulator
from mtlsp.network.trafficnet import TrafficNet


class BaseEnv(object):
    def __init__(self, global_controller_dict, independent_controller_dict, info_extractor):
        self.episode_info = {"id": 0, "start_time": None, "end_time": None}
        self.vehicle_list = VehicleList({})
        self.departed_vehicle_id_list = []
        self.arrived_vehicle_id_list = []
        self.simulator: Simulator = None
        self.net: TrafficNet = None
        self.global_controller_dict = global_controller_dict
        self.global_controller_instance_list = []
        for veh_type in self.global_controller_dict:    # "BV", "CAV"
            self.global_controller_instance_list.append(self.global_controller_dict[veh_type](self, veh_type))
        self.independent_controller_dict = independent_controller_dict
        self.info_extractor = info_extractor(self)
    
    def initialize(self):
        self.episode_info = {"id": self.simulator.episode, "start_time": self.simulator.get_time(), "end_time": None}
        # xyz 0930
        self.simulator._delete_all_vehicles_in_sumo()
        self.traj_data = {}
        self.vehicle_list = VehicleList({})
        self.departed_vehicle_id_list = []
        self.arrived_vehicle_id_list = []

    def __getattrib__(self, item):
        print(item)
        if item == 'step':
            print('step called')
            self.step()

    def _maintain_all_vehicles(self):
        """Maintain the vehicle list based on the departed vehicle list and arrived vehicle list.
        """        
        self.departed_vehicle_id_list = self.simulator.get_departed_vehID_list()
        self.arrived_vehicle_id_list = self.simulator.get_arrived_vehID_list()
        self._add_vehicles(self.departed_vehicle_id_list)
        self._delete_vehicles(self.arrived_vehicle_id_list)
    
    def _add_vehicles(self, veh_id_list):
        """Add vehicles from veh_id_list.

        Args:
            veh_id_list (list(str)): List of vehicle IDs needed to be inserted.

        Raises:
            ValueError: If one vehicle is neither "BV" nor "AV", it should not enter the network.
        """        
        for veh_id in veh_id_list:
            veh = Vehicle(id=veh_id, controller=Controller(), simulator=self.simulator)
            if veh.type in self.independent_controller_dict:
                veh.install_controller(self.independent_controller_dict[veh.type]())
            self.vehicle_list.add_vehicles(veh)

    def _delete_vehicles(self,veh_id_list):
        """Delete vehicles in veh_id_list.

        Args:
            veh_id_list (list(str)): List of vehicle IDs needed to be deleted.

        Raises:
            ValueError: If the vehicle is neither "BV" nor "AV", it shouldn't enter the network.
        """        
        for veh_id in veh_id_list:
            self.vehicle_list.pop(veh_id, None)

    def _check_vehicle_list(self):
        """Check the vehicle lists after the simulation step to maintain them again.
        """        
        realtime_vehID_set = set(self.simulator.get_vehID_list())
        vehID_set = set(self.vehicle_list.keys())
        if vehID_set != realtime_vehID_set:
            # print('Warning: the vehicle list is not up-to-date, so update it!')
            for vehID in realtime_vehID_set:
                if vehID not in vehID_set:
                    self._add_vehicles([vehID])
            for vehID in vehID_set:
                if vehID not in realtime_vehID_set:
                    self._delete_vehicles([vehID])

    # @profile
    # @abstractmethod
    def step(self):
        """Maintain vehicle list and make the simulation step forwards.
        """        
        self._maintain_all_vehicles() # maintain both the bvlist and avlist
        self._step()

    # @profile       
    @abstractmethod
    def _step(self):
        """Method that child class MUST implement to specify all actions needed in one step.
        """
        control_info_list = []
        for global_controller in self.global_controller_instance_list:
            control_info = global_controller.step()
            control_info_list.append(control_info)
        self.info_extractor.get_snapshot_info(control_info_list)
        return control_info_list

    # @profile
    def terminate_check(self):
        reason, stop, additional_info = self._terminate_check()
        if stop:
            self.episode_info["end_time"] = self.simulator.get_time() - self.simulator.step_size
            self.info_extractor.get_terminate_info(stop, reason, additional_info)
        return stop, reason, additional_info

    def _terminate_check(self):
        reason = None
        stop = False
        additional_info = None
        if self.simulator.get_vehicle_min_expected_number() == 0:
            reason = "All Vehicles Left"
            stop = True
        return reason, stop, additional_info
    
    # xyz 0927
    def append_traj_data(self):
        for veh_id in self.traj_data.keys():
            if veh_id not in self.vehicle_list:
                self.traj_data[veh_id]["trajs"].append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                self.traj_data[veh_id]["trajs_mask"].append(0.0)
                self.traj_data[veh_id]["is_pov"].append(0.0)
            else:
                info = self.vehicle_list[veh_id].observation.information["Ego"]
                x, y, z = info["position3D"]
                heading = info["heading"]   # sumo heading definition: 0-north, 90-east, 180-south, 270-west
                vel = info["velocity"]
                acc = info["acceleration"]
                self.traj_data[veh_id]["trajs"].append(np.array([x, y, z, heading, vel, acc]))
                self.traj_data[veh_id]["trajs_mask"].append(1.0)
                self.traj_data[veh_id]["is_pov"].append(float(self.vehicle_list[veh_id].is_pov))
        # TODO: 这里假设在每个episode内车辆只会减少不会中途增加，如果中途增加车辆需要添加相应代码
    
    def save_traj_data(self):
        for veh_id in self.traj_data.keys():
            self.traj_data[veh_id]["vType"] = self.vehicle_list[veh_id].vType
            self.traj_data[veh_id]["trajs"] = np.array(self.traj_data[veh_id]["trajs"])
            self.traj_data[veh_id]["trajs_mask"] = np.array(self.traj_data[veh_id]["trajs_mask"])
            self.traj_data[veh_id]["is_pov"] = np.array(self.traj_data[veh_id]["is_pov"])
            if np.all(self.traj_data[veh_id]["trajs_mask"] == 0):
                del self.traj_data[veh_id]
        # save_path = os.path.join(conf.experiment_config["traj_data_save_path"], conf.experiment_config["experiment_name"]+'-'+conf.experiment_config["map"])
        save_path = os.path.join(self.simulator.experiment_path, "traj_data", conf.experiment_config["map"])
        name = "episode" + str(self.episode_info["id"]) + ".pkl"
        os.makedirs(save_path, exist_ok=True)
        pickle.dump(self.traj_data, open(os.path.join(save_path, name), "wb"))
