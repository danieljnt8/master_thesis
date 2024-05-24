
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
from train_DT import TrainableDT
import gym

from citylearn.citylearn import CityLearnEnv
import itertools
from trajectory.utils.CityStableEnv import *

# Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    state_preds, action_preds, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]


def eval(run_path = None, model_path=None,schema = "citylearn_challenge_2022_phase_2",device ="cpu",TARGET_RETURN=-3000,nb_iter = 719):
    env = CityLearnEnv(schema=schema)
    env = EnvCityGym(env)



    state_mean = np.load(run_path+"state_mean.npy")
    state_std = np.load(run_path+"state_std.npy")

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    state = np.array(state)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
    scale = 1.0

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    for t in range(nb_iter):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        action = get_action(
            model,
            (states - state_mean) / state_std,
            actions,
            rewards,
            target_return,
            timesteps,
        )
        #print(action)
        
        actions[-1] = action
        action = action.detach().cpu().numpy()
        
        state, reward, done, _ = env.step(action)
        
        state = np.array(state)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        print(reward)

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        print(pred_return)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    df_evaluate = env.env.evaluate()
    df_evaluate.to_csv(run_path+"results.csv")

        

if __name__ == "__main__":
    run_path = "checkpoints/city_learn/DT_PPO_100k_cl_24/run/"
    model_path = "checkpoints/city_learn/DT_PPO_100k_cl_24/run/checkpoint-20"
    model = TrainableDT.from_pretrained(model_path,local_files_only = True)
    device = "cpu"

    eval(run_path,model_path,device=device)

