import os
import torch
import numpy as np
import wandb

import pickle
from tqdm.auto import trange, tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass
from datasets import load_from_disk
from omegaconf import OmegaConf

from citylearn.agents.rbc import HourRBC
from citylearn.agents.q_learning import TabularQLearning
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.wrappers import TabularQLearningWrapper

from stable_baselines3.a2c import A2C

from torch.utils.data import DataLoader
from trajectory.models.gpt import GPT, GPTTrainer

from trajectory.utils.common import pad_along_axis
from trajectory.utils.discretization import KBinsDiscretizer
from trajectory.utils.env import create_env

import matplotlib.pyplot as plt


def join_trajectory(states, actions, rewards, discount=0.99):
    traj_length = states.shape[0]
    # I can vectorize this for all dataset as once,
    # but better to be safe and do it once and slow and right (and cache it)
    
    if actions.ndim == 3 :
        actions = actions.reshape(actions.shape[0],actions.shape[1])
    
    if rewards.ndim == 1 :
        rewards = rewards.reshape(rewards.shape[0],1)
        
    print("Discount "+str(discount))
    discounts = (discount ** np.arange(traj_length))

    values = np.zeros_like(rewards)
    for t in range(traj_length):
        # discounted return-to-go from state s_t:
        # r_{t+1} + y * r_{t+2} + y^2 * r_{t+3} + ...
        # .T as rewards of shape [len, 1], see https://github.com/Howuhh/faster-trajectory-transformer/issues/9
        values[t] = (rewards[t + 1:].T * discounts[:-t - 1]).sum()
    print(states.shape)
    print(actions.shape)
    print(rewards.shape)
    print(values.shape)

    joined_transition = np.concatenate([states, actions, rewards, values], axis=-1)

    return joined_transition

def segment(states, actions, rewards, terminals):
    assert len(states) == len(terminals)
    
    trajectories = {}

    episode_num = 0
    for t in trange(len(terminals), desc="Segmenting"):
        if episode_num not in trajectories:
            trajectories[episode_num] = {
                "states": [],
                "actions": [],
                "rewards": []
            }
        
        trajectories[episode_num]["states"].append(states[t])
        trajectories[episode_num]["actions"].append(actions[t])
        trajectories[episode_num]["rewards"].append(rewards[t])

        if terminals[t]:
            # next episode
            episode_num = episode_num + 1

    trajectories_lens = [len(v["states"]) for k, v in trajectories.items()]

    for t in trajectories:
        trajectories[t]["states"] = np.stack(trajectories[t]["states"], axis=0)
        trajectories[t]["actions"] = np.stack(trajectories[t]["actions"], axis=0)
        trajectories[t]["rewards"] = np.stack(trajectories[t]["rewards"], axis=0)

    return trajectories, trajectories_lens

class DiscretizedDataset(Dataset):
    def __init__(self, dataset,env_name="city_learn", num_bins=100, seq_len=10, discount=0.99, strategy="uniform", cache_path="data"):
        self.seq_len = seq_len
        self.discount = discount
        self.num_bins = num_bins
        self.dataset = dataset
        self.env_name = env_name
        
        trajectories, traj_lengths = segment(self.dataset["observations"],self.dataset["actions"],self.dataset["rewards"],self.dataset["dones"])
        self.trajectories = trajectories
        self.traj_lengths = traj_lengths
        self.cache_path = cache_path
        self.cache_name = f"{env_name}_{num_bins}_{seq_len}_{strategy}_{discount}"
        
        self.joined_transitions = []
        for t in tqdm(trajectories, desc="Joining transitions"):
            self.joined_transitions.append(
                    join_trajectory(trajectories[t]["states"], trajectories[t]["actions"], trajectories[t]["rewards"],discount = self.discount)
                )
        """
        if cache_path is None or not os.path.exists(os.path.join(cache_path, self.cache_name)):
            self.joined_transitions = []
            for t in tqdm(trajectories, desc="Joining transitions"):
                self.joined_transitions.append(
                    join_trajectory(trajectories[t]["states"], trajectories[t]["actions"], trajectories[t]["rewards"],discount = self.discount)
                )

            os.makedirs(os.path.join(cache_path), exist_ok=True)
            # save cached version
            with open(os.path.join(cache_path, self.cache_name), "wb") as f:
                pickle.dump(self.joined_transitions, f)
        else:
            with open(os.path.join(cache_path, self.cache_name), "rb") as f:
                self.joined_transitions = pickle.load(f)
        """

        self.discretizer = KBinsDiscretizer(
            np.concatenate(self.joined_transitions, axis=0),
            num_bins=num_bins,
            strategy=strategy
        )

        # get valid indices for seq_len sampling
        indices = []
        for path_ind, length in enumerate(traj_lengths):
            end = length - 1
            for i in range(end):
                indices.append((path_ind, i, i + self.seq_len))
        self.indices = np.array(indices)

    def get_env_name(self):
        return self.env.name

    def get_discretizer(self):
        return self.discretizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #print(idx)
        traj_idx, start_idx, end_idx = self.indices[idx]
        
        joined = self.joined_transitions[traj_idx][start_idx:end_idx]
        

        loss_pad_mask = np.ones((self.seq_len, joined.shape[-1]))
        if joined.shape[0] < self.seq_len:
            # pad to seq_len if at the end of trajectory, mask for padding
            loss_pad_mask[joined.shape[0]:] = 0
            joined = pad_along_axis(joined, pad_to=self.seq_len, axis=0)

        joined_discrete = self.discretizer.encode(joined).reshape(-1).astype(np.longlong)
        loss_pad_mask = loss_pad_mask.reshape(-1)

        return joined_discrete[:-1], joined_discrete[1:], loss_pad_mask[:-1]
    
def train(config_path = "configs/medium/city_learn_traj.yaml",offline_data_path = None,device = "cpu"):
    config = OmegaConf.load(config_path)

    if offline_data_path is None :
        offline_data_path = config.trainer.offline_data_path
    if torch.cuda.is_available():
        device = "cuda:1"

  
    wandb.init(
            **config.wandb,
            config=dict(OmegaConf.to_container(config, resolve=True))
        )
    
    offline_data_path = offline_data_path
    dataset = load_from_disk(offline_data_path)

    datasets = DiscretizedDataset(dataset,discount = config.dataset.discount, seq_len = config.dataset.seq_len, strategy = config.dataset.strategy)
    dataloader = DataLoader(datasets,  batch_size=config.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    trainer_conf = config.trainer
    data_conf = config.dataset

    model = GPT(**config.model)
    model.to(device)
    

    num_epochs = config.trainer.num_epochs

    warmup_tokens = len(datasets) * data_conf.seq_len * config.model.transition_dim
    final_tokens = warmup_tokens * num_epochs

    trainer = GPTTrainer(
            final_tokens=final_tokens,
            warmup_tokens=warmup_tokens,
            action_weight=trainer_conf.action_weight,
            value_weight=trainer_conf.value_weight,
            reward_weight=trainer_conf.reward_weight,
            learning_rate=trainer_conf.lr,
            betas=trainer_conf.betas,
            weight_decay=trainer_conf.weight_decay,
            clip_grad=trainer_conf.clip_grad,
            eval_seed=trainer_conf.eval_seed,
            eval_every=trainer_conf.eval_every,
            eval_episodes=trainer_conf.eval_episodes,
            eval_temperature=trainer_conf.eval_temperature,
            eval_discount=trainer_conf.eval_discount,
            eval_plan_every=trainer_conf.eval_plan_every,
            eval_beam_width=trainer_conf.eval_beam_width,
            eval_beam_steps=trainer_conf.eval_beam_steps,
            eval_beam_context=trainer_conf.eval_beam_context,
            eval_sample_expand=trainer_conf.eval_sample_expand,
            eval_k_obs=trainer_conf.eval_k_obs,  # as in original implementation
            eval_k_reward=trainer_conf.eval_k_reward,
            eval_k_act=trainer_conf.eval_k_act,
            checkpoints_path=trainer_conf.checkpoints_path,
            save_every=1,
            device=device
        )
    
    trainer.train(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs
    )

if __name__ == "__main__":
    train(config_path = "configs/medium/city_learn_traj_v1_seq20_rf_firstCR.yaml")
