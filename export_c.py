import argparse
import json
import os

import torch
import torch.nn as nn
from stable_baselines3 import PPO

from quad_race.environment import Flightplan
from quad_race.utils import generate_c_code


parser = argparse.ArgumentParser(description="Export a trained PPO policy to C.")
parser.add_argument("model_path", help="Path to the .zip model file")
parser.add_argument("--name", default=None,
                    help="Policy name; becomes nn_use_<name>() in the C file. "
                         "Defaults to the model's parent folder name.")
parser.add_argument("--output", default=None,
                    help="Path to the .c file. Defaults to <model_dir>/<name>.c.")
parser.add_argument("--options", default=None,
                    help="Path to options.json (for the flight plan). "
                         "Defaults to <model_dir>/options.json.")
args = parser.parse_args()

model_dir = os.path.dirname(os.path.abspath(args.model_path))
name = args.name or os.path.basename(model_dir)
out_path = args.output or os.path.join(model_dir, f"{name}.c")
options_path = args.options or os.path.join(model_dir, "options.json")

with open(options_path) as f:
    flight_plan_path = json.load(f)["EnvironmentConfig"]["flight_plan"]
flightplan = Flightplan(flight_plan_path)

model = PPO.load(args.model_path)
actor_layers = list(model.policy.mlp_extractor.policy_net.children()) + [model.policy.action_net]
network = nn.Sequential(*actor_layers)
std = torch.exp(model.policy.log_std.detach()).cpu().numpy()

generate_c_code(out_path, name, network, std, flightplan)
print(f"Wrote {out_path}")
