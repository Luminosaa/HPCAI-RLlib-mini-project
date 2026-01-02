import gymnasium as gym


# This is the "Safe" way to test if your ROM is working

env = gym.make("ale_py:ALE/Tetris-v5", render_mode="human")

obs, info = env.reset()


for _ in range(1000):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:

        obs, info = env.reset()


env.close() 