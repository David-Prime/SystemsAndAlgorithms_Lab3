# Environment

## KP-test version 2022-12-09 with working video

"""
Installing all required drivers for the RL-algoritm by "!pip - install commands".
"""

##!pip install highway-env ##KP
#!pip install --user git+https://github.com/eleurent/highway-env
##KP - install newer version than published - clone from that
#!pip install git+https://github.com/USERNAME/highway-env

"""
Loading all required libraries to run the RL-algorithm for the ego-car and the rest of the simulation
and rearning the parameters.
"""
import gym
import highway_env

# Agent
##KP !pip install git+https://github.com/eleurent/rl-agents
#!pip install --user git+https://github.com/eleurent/rl-agents
#!pip install moviepy ##KP - used for playing videos
#from moviepy.editor import *  #KP - gives error

# Visualisation utils
import sys
#%load_ext tensorboard
#!pip install tensorboardx gym pyvirtualdisplay
#!apt-get install -y xvfb python-opengl ffmpeg      #gets an error
#!git clone https://github.com/eleurent/highway-env.git
sys.path.insert(0, '/content/highway-env/scripts/')
# from utils import record_videos,show_videos
#!pip install imageio-ffmpeg #KP - mpeg4 encoder needed for play
import imageio_ffmpeg
#from moviepy.video.io.ImageSequenceClip import ImageSequenceClip ##KP

from typing import Dict, Text

import numpy as np
import gym
import highway_env
from matplotlib import pyplot as plt
# %matplotlib inline

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray

"""
The HighwayEnv class is creating an enviroment for the simulation and also the cars within it (agents) as well as the ego-car.
all methods included in this class control the different parameters in the "config() -method" and the calculations of them.
These paramaters are the one to change to identify the optimum values that could bring a good learning rate and learned result, 
in other terms a high performance of the RL-model for autonomous driving.
"""
class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.
    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 0,
            "controlled_vehicles": 1,
            "initial_lane_id": 0,
            "duration": 5,  # [s]
            "ego_spacing": 0.5,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.6,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": -100.0,   # The reward received at each lane change action.
            "reward_speed_range": [0, 1],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config
    
    # Method to reset the enviroment of road and vehicles after crash or mime limit reached of episode
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    # Method to create the enviroment of road
    def _create_road(self) -> None:
        #Create a road composed of straight adjacent lanes.
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
    # Method to create the enviroment of cars
    def _create_vehicles(self) -> None:
        #Create some new random vehicles of a given type, and add them on the road.
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
    
    # Method to calculate the reward of the previous driving episode
    def _reward(self, action: Action) -> float:
        
        #The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        #:param action: the last action performed
        #:return: the corresponding reward
        
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }
    # Method to handle disruptions like a car crash or time limit reached
    def _is_terminated(self) -> bool:
        #The episode is over if the ego vehicle crashed or the time is out.
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road or
                self.time >= self.config["duration"])

    def _is_truncated(self) -> bool:
        return False
"""
A similar class as above but with limited amount of parameters for the environment,
gives a more easy to run simulation/parameter training with the reward algorithm of RL.
"""
class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
        """
    # The method to control the different parameters to adjust the reward and learning performance
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 50,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 0.1,
        })
        return cfg

    # Method to create the cars/agents in the environment
    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False



env = gym.make('highway-v0')
env.reset()
for _ in range(3):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, terminated, truncated, info = env.step(action)    #obs, reward, done, info = env.step(action)
    env.render()

# plt.imshow(env.render(mode="rgb_array"))
# plt.show()

"""
Loading the libraries for environment and the validation of the performance of RL-algorithm.
"""

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

# Get the environment and agent configurations from the rl-agents repository
#!git clone https://github.com/eleurent/rl-agents.git
# %cd /content/rl-agents/scripts/


"""
Loads the parameters from the config files (.json) and adding the paths to these files.
"""
env_config = 'rl-agents/scripts/configs/HighwayEnv/env_easy.json'   #env_config = 'configs/HighwayEnv/env.json'
agent_config = 'rl-agents/scripts/configs/HighwayEnv/agents/DQNAgent/dqn.json'


"""
env = load_environment(env_config)#env = load_environment(env_config)
import pprint
from matplotlib import pyplot as plt
env.config["lane_change_reward"] = 1
env.config["vehicles_count"] = 50
env.config['reward_speed_range'] = [0, 100]
env.config["lanes_count"] = 2
env.config["initial_lane_id"] = 0
pprint.pprint(env.config)
pprint.pprint(agent_config)
"""

# Gives a test of the simulation and a picture of this
env.reset()
plt.imshow(env.render(mode="rgb_array"))
plt.show()


"""
Loads the pasameters from functions to set up the agents (cars on the road),
and sets up the number of episodes to run before the validation of the RL-model
"""
agent = load_agent(agent_config, env)#agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=60, display_env=True)
print(f"Ready to train {agent} on {env}")



# %tensorboard --logdir "{evaluation.directory}"
# Initializing the training of the RL-model and the evaluation of it
evaluation.train()



"""
Loads the simulation environment and sterts to record each episode of the simulation of the validation
"""
env = load_environment(AbstractEnv)# env = load_environment(env_config)
env.configure({"offscreen_rendering": True})
env.render(mode="rgb_array")
env = record_videos(env)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=30, recover=True)
evaluation.test()
show_videos(evaluation.run_directory)

