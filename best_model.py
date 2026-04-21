import argparse
import os
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser(
    description="Print the .zip checkpoint with the highest rollout/ep_rew_mean "
                "from a trained model's tensorboard log.")
parser.add_argument("model_dir", help="Path to a model folder (e.g. models/M18)")
parser.add_argument("--logs_dir", default="logs",
                    help="Root directory holding <name>_0/ tensorboard logs (default: logs)")
args = parser.parse_args()

model_dir = os.path.normpath(args.model_dir)
name = os.path.basename(model_dir)
log_dir = os.path.join(args.logs_dir, name + "_0")

if not os.path.isdir(log_dir):
    sys.exit(f"tensorboard log not found: {log_dir}")

acc = EventAccumulator(log_dir)
acc.Reload()
if "rollout/ep_rew_mean" not in acc.Tags()["scalars"]:
    sys.exit(f"no rollout/ep_rew_mean scalar in {log_dir}")

steps, rewards = zip(*[(e.step, e.value) for e in acc.Scalars("rollout/ep_rew_mean")])
steps, rewards = np.array(steps), np.array(rewards)

checkpoints = {int(f.split(".")[0]) for f in os.listdir(model_dir) if f.endswith(".zip")}
if not checkpoints:
    sys.exit(f"no .zip checkpoints in {model_dir}")

mask = np.isin(steps, list(checkpoints))
if not mask.any():
    sys.exit(f"no tensorboard steps match the checkpoints in {model_dir}")

steps, rewards = steps[mask], rewards[mask]
best = int(np.argmax(rewards))
best_path = os.path.join(model_dir, f"{steps[best]}.zip")

print(f"best: ep_rew_mean={rewards[best]:.3f} at step {steps[best]} "
      f"(out of {len(steps)} checkpoints)", file=sys.stderr)
print(best_path)
