import ray
import os
import torch
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack

ray.init(ignore_reinit_error=True)

def env_creator(config):
    import gymnasium as gym
    return wrap_atari_for_new_api_stack(gym.make("ale_py:ALE/Tetris-v5"), dim=84, framestack=4)

tune.register_env("tetris_fast", env_creator)

config = (
    PPOConfig()
    .environment(env="tetris_fast")
    .framework("torch")
    .env_runners(num_env_runners=1)
    .rl_module(model_config={"vf_share_layers": True}) # Default CNN is auto-selected
    .training(
        train_batch_size=2000, # Very small for fast testing
        lr=0.001,             
    )
)

# Run for only 2 iterations to save time
results = tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 2}, 
    checkpoint_at_end=True,
    verbose=1,
)

# Get the path to the saved checkpoint
checkpoint_path = results.get_last_checkpoint().path
print(f"CHECKPOINT_SAVED_AT: {checkpoint_path}")



import torch
import numpy as np
import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm

# 1. Load the trained algorithm
algo = Algorithm.from_checkpoint(checkpoint_path)

# 2. Get the RLModule (This was the missing part!)
rl_module = algo.get_module("default_policy")

# 3. Create the environment for rendering
env = gym.make("ale_py:ALE/Tetris-v5", render_mode="human")
# Slow down the FPS so it's watchable
env.metadata["render_fps"] = 20 
env = wrap_atari_for_new_api_stack(env, dim=84, framestack=4)

obs, info = env.reset()
done = False

print("Starting visualization...")
try:
    while not done:
        # Convert observation to a torch tensor batch: [1, 84, 84, 4]
        obs_batch = torch.from_numpy(np.array([obs])).float()
        
        # Use the rl_module we defined above
        output = rl_module.forward_inference({"obs": obs_batch})
        
        # Get the best action (argmax of logits)
        logits = output["action_dist_inputs"]
        action = torch.argmax(logits, dim=1).item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        done = terminated or truncated
finally:
    # Always close to prevent Jupyter from hanging/crashing
    env.close()
    print("Environment closed.")
print("Score:", info.get("score", 0))