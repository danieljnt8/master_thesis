import os
import random
import warnings
from dataclasses import dataclass
import numpy as np
import torch
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
from datasets import load_from_disk
from tqdm.auto import trange, tqdm
import datasets

@dataclass
class DecisionTransformerCityLearnDataCollator:
    """
        Data collators are objects that will form a batch by using a list of dataset elements as input.
        Args:
            dataset ('List[dict]'):
                Offline dataset to train the model with.
            max_ep_len ('float'):
                Length of an episode in the dataset
            max_len ('float'):
                Subsets of the episode we use for training
            scale ('float'):
                normalization of rewards/returns
        """

    return_tensors: str = "pt"
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset

    def __init__(self, dataset, model_path, max_ep_len, max_len=24, scale=1000) -> None:
        self.dataset = dataset
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.max_ep_len = max_ep_len
        self.max_len = max_len
        self.scale = scale
        # calculate dataset stats for normalization of states
        states = []  # List of all states of all sequences e.g. [s1,s2,s3,s1,s2,s3,s1,s2,s3]
        traj_lens = []  # List of sequence length e.g. [3, 3, 3]
        for obs in dataset["observations"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        if self.max_ep_len > 4096:
            warnings.warn("max_ep_len over 4096. Error while training expected, please lower max_ep_len")

        np.save(f'{model_path}/state_mean.npy', self.state_mean)
        np.save(f'{model_path}//state_std.npy', self.state_std)

        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []  # mask?

        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)

            # get sequences from dataset
            s.append(np.array(feature["observations"][si: si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["actions"][si: si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si: si + self.max_len]).reshape(1, -1, 1))

            d.append(np.array(feature["dones"][si: si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                : s[-1].shape[1]  # TL: check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        # print("rtg", rtg)
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # return_priority = 0.1

        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        # return_preds = output[2]
        # return_targets = kwargs["rewards"]
        attention_mask = kwargs["attention_mask"]

        act_dim = action_preds.shape[2]
        # ret_dim = return_preds.shape[2]

        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # return_preds = return_preds.reshape(-1, ret_dim)[attention_mask.reshape(-1) > 0]
        # return_targets = return_targets.reshape(-1, ret_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)  # + return_priority * (return_preds - return_targets) ** 2)

        return {"loss": loss}  # TODO test other loss functions

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

def segment_v2(states, actions, rewards, terminals):
    assert len(states) == len(terminals)
    
    trajectories = []
    episode = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones":[]
    }

    for t in trange(len(terminals), desc="Segmenting"):
        episode["observations"].append(states[t])
        episode["actions"].append(actions[t])
        episode["rewards"].append(rewards[t])
        episode["dones"].append(terminals[t])

        if terminals[t]:
            # Convert lists to numpy arrays
            episode["observations"] = np.array(episode["observations"])
            episode["actions"] = np.array(episode["actions"])
            episode["rewards"] = np.array(episode["rewards"])
            episode["dones"] = np.array(episode["dones"])
            # Append the current episode to the trajectories list
            trajectories.append(episode)
            # Reset episode
            episode = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones":[]
            }

    # If there are any remaining observations, actions, and rewards in the current episode
    if episode["observations"]:
        episode["observations"] = np.array(episode["observations"])
        episode["actions"] = np.array(episode["actions"])
        episode["rewards"] = np.array(episode["rewards"])
        episode["dones"] = np.array(episode["dones"])
        trajectories.append(episode)
    
    trajectories_lens = [len(episode["observations"]) for episode in trajectories]

    return trajectories, trajectories_lens


if __name__ == "__main__":
    model_name = "DT_PPO_100k_conte"
    offline_data_path = "data_interactions/PPO/model_PPO_timesteps_100000_seed_572.pkl"
    max_ep_len= 4096
    max_len = 24
    scale = 1
    context_length = 24 #follow the max_len

    lr = 1e-4
    epochs = 20
    batch_size = 64
    weight_decay = 1e-4
    warmup_ratio = 0 

    model_name = "DT_PPO_100k_cl_{}".format(context_length)

    dataset = load_from_disk(offline_data_path)
    dataset,_ = segment_v2(dataset["observations"],dataset["actions"],dataset["rewards"],dataset["dones"])
    dataset = datasets.Dataset.from_dict({k: [s[k] for s in dataset] for k in dataset[0].keys()})

    model_path = f"checkpoints/city_learn/{model_name}/run"
    
    '''
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f'created path {model_path}')
    '''
    
    if os.path.exists(model_path):
        suffix = 2
        while True:
            new_path = f"{model_path}_{suffix}"
            if not os.path.exists(new_path):
                model_path = new_path
                break
            suffix += 1
    os.makedirs(model_path, exist_ok=True)

    collator =DecisionTransformerCityLearnDataCollator(dataset, model_path, max_ep_len, max_len, scale)
    config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim, max_length=context_length)   
    model = TrainableDT(config) 

    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=False,
        remove_unused_columns=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        max_grad_norm=1,
        logging_dir=f"logs/DT_logs/{model_name}",
        logging_steps=1,   
        save_steps=1,
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()