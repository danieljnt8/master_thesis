import os

# import couple of libs some will be useful
import gym
import numpy as np
from collections import deque
import random
import re
import os
import sys
import time
import json
import itertools
from datasets import Dataset
from _code.const import PATH_MODEL_SB,PATH_DATA_INTERACTIONS
from citylearn.agents.rbc import BasicRBC as BRBC
# import stable_baselines3
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.utils import set_random_seed

from citylearn.citylearn import CityLearnEnv
from utils.rewards import CustomReward

import functools

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    #building_info = env.get_building_information()
    #building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
              #  "building_info": building_info,
                "observation": observations }
    return obs_dict


### PART of ENVIRONMENT

index_commun = [0, 2, 19, 4, 8, 24]
index_particular = [20, 21, 22, 23]

normalization_value_commun = [12, 24, 2, 100, 100, 1]
normalization_value_particular = [5, 5, 5, 5]

len_tot_index = len(index_commun) + len(index_particular) * 5

## env wrapper for stable baselines
class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self, env):
        self.env = env
        self.dataset= []
        # get the number of buildings
        self.num_buildings = len(env.action_space)

        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * self.num_buildings), high=np.array([1] * self.num_buildings), dtype=np.float32)

        # define the observation space
        self.observation_space = gym.spaces.Box(low=np.array([0] * len_tot_index), high=np.array([1] * len_tot_index), dtype=np.float32)

        # TO THINK : normalize the observation space
        self.current_obs = None
    def reset(self):
        obs_dict = env_reset(self.env)
        obs = self.env.reset()

        observation = self.get_observation(obs)
        
        self.current_obs = observation
        self.interactions = []

        return observation

    def get_observation(self, obs):
        """
        We retrieve new observation from the building observation to get a proper array of observation
        Basicly the observation array will be something like obs[0][index_commun] + obs[i][index_particular] for i in range(5)

        The first element of the new observation will be "commun observation" among all building like month / hour / carbon intensity / outdoor_dry_bulb_temperature_predicted_6h ...
        The next element of the new observation will be the concatenation of certain observation specific to buildings non_shiftable_load / solar_generation / ...  
        """
        
        # we get the observation commun for each building (index_commun)
        observation_commun = [obs[0][i]/n for i, n in zip(index_commun, normalization_value_commun)]
        observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in obs]

        observation_particular = list(itertools.chain(*observation_particular))
        # we concatenate the observation
        observation = observation_commun + observation_particular

        return observation

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        # reprocessing action
        action = [[act] for act in action]
        #print(action)
        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)
        
        observation = self.get_observation(obs)
        
        
        self.interactions.append({
            "observations": self.current_obs,
            "next_observations": self.get_observation(obs),  # Assuming next observation is same as current for simplicity
            "actions": action,
            "rewards": sum(reward),
            "dones": done,
            "info": info
        })

        self.dataset.append({
            "observations": self.current_obs,
            "next_observations": self.get_observation(obs),  # Assuming next observation is same as current for simplicity
            "actions": action,
            "rewards": sum(reward),
            "dones": done,
            "info": info
        })
        
        self.current_obs = observation
        
        

        return observation, sum(reward), done, info
        
    def render(self, mode='human'):
        return self.env.render(mode)

def train_agent(schema, timesteps, seed, additional= None, saved_path = PATH_MODEL_SB, model_str = "PPO"):
    env = CityLearnEnv(schema=schema)
    #reward_func = CustomReward(env)
    #env.reward_function = reward_func
    reward_name = "base"
    #env = EnvCityGym(env)
    

    if model_str == "PPO":
        env = EnvCityGym(env)
        model =  PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps)
        data = env.dataset  #for stablebaselines
        model.save(model_path)

        model_path = PATH_MODEL_SB + "/" + model_str + "/" + "model_{}_timesteps_{}_rf_{}_seed_{}".format(model_str,timesteps,reward_name,seed)
    else:
        model_str = "BasicRBC"
        model = BRBC(env)

        model.learn(episodes=11)
        data = model.dataset #for RBC Agent

    
    data_path = PATH_DATA_INTERACTIONS+ "/" + model_str + "/" + "model_{}_timesteps_{}_rf_{}_seed_{}.pkl".format(model_str,timesteps,reward_name,seed)
    #model_path = PATH_MODEL_SB + "/" + model_str + "/" + "model_{}_timesteps_{}_rf_{}_seed_{}".format(model_str,timesteps,reward_name,seed)
    Dataset.from_dict({k: [s[k] for s in data] for k in data[0].keys()}).save_to_disk(data_path)
    
if __name__ == "__main__":

    schema = "citylearn_challenge_2022_phase_2"

    train_agent(schema,timesteps = 100000,seed =572,model_str= "BasicRBC")
