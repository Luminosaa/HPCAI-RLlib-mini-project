import ray
import torch
import numpy as np
import gymnasium as gym

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray import tune


if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)


def env_creator(config=None):
    env = gym.make(
        "ale_py:ALE/Frostbite-v5",
        render_mode="human",   # enable rendering for test
    )
    env.metadata["render_fps"] = 20
    return wrap_atari_for_new_api_stack(env, dim=84, framestack=4)

tune.register_env("frostbite", env_creator)


config = (
    PPOConfig()
    .environment(env="frostbite")
    .framework("torch")
    .env_runners(num_env_runners=0)
    .rl_module(model_config={"vf_share_layers": True})
)


checkpoint_path = (
    "/home/thomas/HPCAI-RLlib-mini-project/checkpoints/"
    "PPO_2026-01-04_00-41-36/"
    "PPO_frostbite_be19d_00000_0_2026-01-04_00-41-36/"
    "checkpoint_000018/"
)

algo = config.build()
algo.restore(checkpoint_path)

print("Checkpoint restored successfully.")


rl_module = algo.get_module("default_policy")


env = env_creator()
obs, info = env.reset()
done = False

print("Starting visualization...")

try:
    while not done:
        obs_batch = torch.from_numpy(np.expand_dims(obs, axis=0)).float()

        with torch.no_grad():
            output = rl_module.forward_inference({"obs": obs_batch})

        logits = output["action_dist_inputs"]
        action = torch.argmax(logits, dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

finally:
    env.close()
    print("Environment closed.")

