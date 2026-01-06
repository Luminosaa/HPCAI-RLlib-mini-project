import ray
import os
import torch
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

def env_creator(config):
    import gymnasium as gym
    return wrap_atari_for_new_api_stack(gym.make("ale_py:ALE/Frostbite-v5"), dim=84, framestack=4)

tune.register_env("frostbite", env_creator)

config = (
    PPOConfig()
    .environment(env="frostbite")
    .framework("torch")
    .env_runners(
        num_env_runners=24,
        num_cpus_per_env_runner=1,
    )
    .rl_module(model_config={"vf_share_layers": True}) 
    .training(
        train_batch_size=8192, 
        lr=2.5e-4,  
        gamma=0.99, 
    )
)


results = tune.run(
    "PPO",
    storage_path=os.path.abspath("./checkpoints"),    
    config=config.to_dict(),
    stop={"training_iteration": 100},
    checkpoint_freq=5, 
    checkpoint_at_end=True,
    resume="AUTO", # Resume from last checkpoint if available
    verbose=2,
    progress_reporter=tune.CLIReporter(
        metric_columns=["episode_reward_mean", "episode_len_mean", "timesteps_total"],
    ),
)

checkpoint_path = results.get_last_checkpoint().path
print(f"CHECKPOINT_SAVED_AT: {checkpoint_path}")
