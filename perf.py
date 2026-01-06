import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack


def env_creator(config):
	return wrap_atari_for_new_api_stack(
		gym.make("ale_py:ALE/Frostbite-v5"), dim=84, framestack=4
	)


def measure_once(num_env_runners, train_iters, train_batch_size):
	config = (
		PPOConfig()
		.environment(env="frostbite")
		.framework("torch")
		.env_runners(num_env_runners=num_env_runners, num_cpus_per_env_runner=1)
		.rl_module(model_config={"vf_share_layers": True})
		.training(train_batch_size=train_batch_size, lr=2.5e-4, gamma=0.99)
	)

	algo = config.build_algo()

	# Warmup iteration (not measured)
	_ = algo.train()

	secs_list = []
	rows = []
	for i in range(1, train_iters + 1):
		t0 = time.time()
		_ = algo.train()
		secs = time.time() - t0

		secs_list.append(secs)
		rows.append(
			{
				"num_env_runners": num_env_runners,
				"iteration": i,
				"seconds_per_iter": secs,
			}
		)
		print(f"Runners={num_env_runners} iter={i} secs={secs:.3f}")

	try:
		algo.stop()
	except Exception:
		pass
	return secs_list, rows


def sweep_and_save(runners, train_iters, train_batch_size, output_csv):
	header = [
		"num_env_runners",
		"iteration",
		"seconds_per_iter",
	]
	# fresh file
	if os.path.exists(output_csv):
		os.remove(output_csv)
	with open(output_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=header)
		writer.writeheader()

		avg_by_runners = {}
		for n in runners:
			secs_list, rows = measure_once(n, train_iters, train_batch_size)
			for r in rows:
				writer.writerow(r)


	return avg_by_runners


def main():
	# Keep Ray up across the entire sweep
	if not ray.is_initialized():
		ray.init(ignore_reinit_error=True)
	# Register env once per Ray runtime
	tune.register_env("frostbite", env_creator)

	runners = [1, 2, 4, 8, 16, 24]
	train_iters = 3  # measured iterations after 1 warmup
	train_batch_size = 5012
	output = "perf_results.csv"
	sweep_and_save(runners, train_iters, train_batch_size, output)
	print(f"\nSaved raw measurements to {output}")

	ray.shutdown()


if __name__ == "__main__":
	main()
