
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(csv_path):
	seconds = defaultdict(list)
	steps_per_sec = defaultdict(list)
	has_steps = False

	with open(csv_path, "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				runners = int(row.get("num_env_runners", ""))
			except Exception:
				continue
			try:
				sec = float(row.get("seconds_per_iter", ""))
			except Exception:
				sec = None
			if sec is not None:
				seconds[runners].append(sec)

			sps_val = row.get("steps_per_sec")
			if sps_val is not None and sps_val != "":
				try:
					sps = float(sps_val)
					steps_per_sec[runners].append(sps)
					has_steps = True
				except Exception:
					pass

	return seconds, steps_per_sec if has_steps else None


def compute_avg(data_dict):
	xs = sorted(data_dict.keys())
	ys = [sum(data_dict[n]) / len(data_dict[n]) if data_dict[n] else float("nan") for n in xs]
	return xs, ys


def plot(csv_path: str, out_path: str):
	seconds, steps = load_data(csv_path)

	# Determine layout based on available metrics
	if steps is not None and any(steps.values()):
		fig, axes = plt.subplots(1, 2, figsize=(10, 4))
		ax1, ax2 = axes
	else:
		fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
		ax2 = None

	# Plot seconds per iteration (average)
	xs_sec, ys_sec = compute_avg(seconds)
	ax1.plot(xs_sec, ys_sec, marker="o")
	ax1.set_title("Seconds/iter vs runners (avg)")
	ax1.set_xlabel("num_env_runners")
	ax1.set_ylabel("seconds per iteration")
	ax1.grid(True, alpha=0.3)

	# Plot steps/sec (average), if available
	if ax2 is not None and steps is not None:
		xs_sps, ys_sps = compute_avg(steps)
		ax2.plot(xs_sps, ys_sps, marker="o", color="tab:green")
		ax2.set_title("Steps/sec vs runners (avg)")
		ax2.set_xlabel("num_env_runners")
		ax2.set_ylabel("steps per second")
		ax2.grid(True, alpha=0.3)

	fig.tight_layout()
	fig.savefig(out_path)
	print(f"Saved plot to {out_path}")


def main():
	parser = argparse.ArgumentParser(description="Plot perf sweep CSV")
	parser.add_argument("--input", type=str, default="perf_results.csv")
	parser.add_argument("--output", type=str, default="perf.png")
	args = parser.parse_args()

	plot(args.input, args.output)


if __name__ == "__main__":
	main()
