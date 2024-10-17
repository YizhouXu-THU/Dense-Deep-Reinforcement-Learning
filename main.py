import sys
import os
from mtlsp.simulator import Simulator
from envs.nde import NDE
from envs.nade import NADE
from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
from conf import conf
# libsumo_flag = (not conf.simulation_config["gui_flag"])
# if libsumo_flag:
#     import libsumo as traci
# else:
#     import traci
from functools import partial
# import time
# import shutil
import utils
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='NADE', metavar='N',
                    help='simulation running mode, (NDE or D2RL or NADE)')
parser.add_argument('--experiment_name', type=str, default='1010', metavar='N',
                    help='specify experiment name (debug by default)')
parser.add_argument('--worker_id', type=int, default=0, metavar='N',
                    help='specify the worker id (for multi-process purposes)')
args = parser.parse_args()

conf.experiment_config["mode"] = args.mode
conf.experiment_config["experiment_name"] = args.experiment_name
if args.mode == "NDE":
    conf.simulation_config["epsilon_setting"] = "fixed" # not use d2rl-based agent
elif args.mode == "NADE":
    conf.simulation_config["epsilon_setting"] = "fixed" # not use d2rl-based agent
elif args.mode == "D2RL":
    conf.simulation_config["epsilon_setting"] = "drl" # use d2rl-based agent

print(f"Using mode {conf.experiment_config['mode']}, epsilon_setting {conf.simulation_config['epsilon_setting']}")
# If running D2RL experiments, then load the D2RL agent
d2rl_agent_path = "./checkpoints/2lane_400m/model.pt"
if conf.simulation_config["epsilon_setting"] == "drl":
    try:
        conf.discriminator_agent = conf.load_discriminator_agent(checkpoint_path=d2rl_agent_path)
    except:
        print("Time out, shutting down")
        sys.exit(0)


def run_nade_experiment(episode, experiment_path):
# def run_nade_experiment(episode, env: NADE, sim: Simulator):
    # Specify the running experiment
    if args.mode == "NDE":
        env = NDE(cav_model=conf.experiment_config["AV_model"])
    elif args.mode == "NADE" or args.mode == "D2RL":
        env = NADE(BVController=TreeSearchNADEBackgroundController, cav_model=conf.experiment_config["AV_model"])
    
    # Specify the sumo map file and sumo configuration file
    # sumo_net_file_path = './maps/2LaneHighway/2LaneHighway.net.xml'
    # sumo_route_file_path = './maps/2LaneHighway/2LaneHighwayHighSpeed.rou.xml'
    # sumo_config_file_path = './maps/2LaneHighway/2LaneHighwayHighSpeed.sumocfg'
    # sumo_net_file_path = './maps/town04_straight/Town04.net.xml'
    # sumo_route_file_path = './maps/town04_straight/Town04.rou.xml,./maps/town04_straight/carlavtypes.rou.xml'
    # sumo_config_file_path = './maps/town04_straight/Town04.sumocfg'
    # sumo_net_file_path = './maps/singapore-onenorth/singapore-onenorth.net.xml'
    # sumo_route_file_path = './maps/singapore-onenorth/singapore-onenorth-d2rl-example.rou.xml'
    # sumo_config_file_path = './maps/singapore-onenorth/singapore-onenorth.sumocfg'
    map = "singapore-onenorth"
    conf.experiment_config["map"] = map
    sumo_net_file_path = os.path.join('./maps', map, map+'.net.xml')
    sumo_route_file_path = os.path.join('./maps', map, map+'-d2rl-example.rou.xml')
    sumo_config_file_path = os.path.join('./maps', map, map+'.sumocfg')
    
    # Set up the simulator
    sim = Simulator(
        sumo_net_file_path=sumo_net_file_path,
        sumo_route_file_path=sumo_route_file_path,
        sumo_config_file_path=sumo_config_file_path,
        num_tries=50,
        step_size=conf.simulation_config["step_size"],
        action_step_size=conf.simulation_config["action_step_size"],
        lc_duration=1,
        track_cav=conf.simulation_config["gui_flag"],
        sublane_flag=True,
        gui_flag=conf.simulation_config["gui_flag"],
        # output=["fcd"],
        output=[],
        experiment_path=experiment_path
    )
    sim.bind_env(env)
    # Begin the experiment running
    sim.run(episode)
    # Return the experiment result: if no crash happens, then return 0, else, return the collision with the relative importance sampling weight
    return env.info_extractor.weight_result

def run_experiments(run_experiment=run_nade_experiment):
    # get the total running episode number and the saving experiment path
    episode_num, experiment_path = utils.get_conf()
    # get the episode starting id (default 0, can be specified using --worker_id)
    start_num = int(args.worker_id) * episode_num
    
    # # xyz 0930
    # # Specify the running experiment
    # env = NADE(BVController=TreeSearchNADEBackgroundController, cav_model=conf.experiment_config["AV_model"])
    
    # # Specify the sumo map file and sumo configuration file
    # # sumo_net_file_path = './maps/2LaneHighway/2LaneHighway.net.xml'
    # # sumo_route_file_path = './maps/2LaneHighway/2LaneHighwayHighSpeed.rou.xml'
    # # sumo_config_file_path = './maps/2LaneHighway/2LaneHighwayHighSpeed.sumocfg'
    # # sumo_net_file_path = './maps/town04_straight/Town04.net.xml'
    # # sumo_route_file_path = './maps/town04_straight/Town04.rou.xml,./maps/town04_straight/carlavtypes.rou.xml'
    # # sumo_config_file_path = './maps/town04_straight/Town04.sumocfg'
    # # sumo_net_file_path = './maps/singapore-onenorth/singapore-onenorth.net.xml'
    # # sumo_route_file_path = './maps/singapore-onenorth/singapore-onenorth-d2rl-example.rou.xml'
    # # sumo_config_file_path = './maps/singapore-onenorth/singapore-onenorth.sumocfg'
    # map = "singapore-onenorth"
    # conf.experiment_config["map"] = map
    # sumo_net_file_path = os.path.join('./maps', map, map+'.net.xml')
    # sumo_route_file_path = os.path.join('./maps', map, map+'-d2rl-example.rou.xml')
    # sumo_config_file_path = os.path.join('./maps', map, map+'.sumocfg')
    
    # # Set up the simulator
    # sim = Simulator(
    #     sumo_net_file_path=sumo_net_file_path,
    #     sumo_route_file_path=sumo_route_file_path,
    #     sumo_config_file_path=sumo_config_file_path,
    #     num_tries=50,
    #     step_size=conf.simulation_config["step_size"],
    #     action_step_size=conf.simulation_config["action_step_size"],
    #     lc_duration=1,
    #     track_cav=conf.simulation_config["gui_flag"],
    #     sublane_flag=True,
    #     gui_flag=conf.simulation_config["gui_flag"],
    #     # output=["fcd"],
    #     output=[],
    #     experiment_path=experiment_path
    # )
    # sim.bind_env(env)
    
    # define the run single experiment function
    run_experiment_ = partial(run_experiment, experiment_path=experiment_path)
    # run_experiment_ = partial(run_experiment, env=env, sim=sim)
    # Run {episode_num} episodes
    weight_result = []
    run_experiments_number = 0
    crash_number = 0
    for i in range(start_num, start_num+episode_num):
        print("episode:", i, end="  ")
        # try:
        if True:
            weight_tmp = run_experiment_(i)
            # if args.mode == "NDE":
            #     if weight_tmp > 0:
            #         crash_number += 1
            #     run_experiments_number += 1
            #     expected_run_experiments_number = i - start_num + 1
            #     # Save the result every 50 episodes to reduce the disk usage (for higher computational efficiency)
            #     if (i-start_num) % 50 == 0:
            #         np.save(experiment_path + "/weight" + str(args.worker_id) + ".npy", np.array([crash_number, run_experiments_number, expected_run_experiments_number]))
            # else:
            #     weight_result.append(weight_tmp)
            #     if (i-start_num) % 50 == 0:
            #         np.save(experiment_path + "/weight" + str(args.worker_id) + ".npy", np.array(weight_result))
            #     run_experiments_number += 1
            #     expected_run_experiments_number = i - start_num + 1
        # except Exception as e:
        #     print(e, f"Error happens at worker {args.worker_id} episode {i}")
        #     traci.close()
        #     continue
    # sim.stop()

if __name__ == "__main__":
    run_experiments(run_nade_experiment)
