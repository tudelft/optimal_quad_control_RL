import argparse
import json
import os
import shutil

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from quad_race.environment import *


parser = argparse.ArgumentParser()
parser.add_argument('options_path', nargs='?', default='options.json')
args = parser.parse_args()

print("Training with option:", args.options_path)

with open(args.options_path, 'r') as f:
    params_config = json.load(f)
name = params_config["model_name"]
print("Training model with name:", name)

pretrained_path = params_config.get("TrainConfig", {}).get("pretrained_model", "")
if pretrained_path:
    print("Loading pretrained model:", pretrained_path)

# Default PPO Config
ppo_config = {
    "activation_fn": "relu",
    "pi": [64, 64, 64],
    "vf": [256, 256, 256],
    "log_std_init": 0,
    "n_steps": 2000,
    "batch_size": 5000,
    "n_epochs": 10,
    "gamma": 0.999,
    "ent_coef": 0,
}
ppo_config.update(params_config.get("PPOConfig", {}))
print("PPOConfig:", ppo_config)

activation_map = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}
act = ppo_config["activation_fn"].lower()
if act not in activation_map:
    raise ValueError(f"Unsupported activation function: {ppo_config['activation_fn']}. Please use 'relu' or 'tanh'.")
ppo_config["activation_fn"] = activation_map[act]

models_dir = 'models/'
log_dir = 'logs/'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

train_folder = os.path.join(models_dir, name)
log_folder = os.path.join(log_dir, name + '_0')
if os.path.exists(train_folder) or os.path.exists(log_folder):
    response = input(f"Folder {train_folder} already exists. Overwrite? (y/n): ")
    if response.lower() != 'y':
        print("Exiting without overwriting.")
        exit()
    print("Overwriting...")
    shutil.rmtree(train_folder, ignore_errors=True)
    shutil.rmtree(log_folder, ignore_errors=True)

os.makedirs(train_folder)
shutil.copy(args.options_path, os.path.join(train_folder, 'options.json'))

env = QuadRace()
env = VecMonitor(env)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        activation_fn=ppo_config["activation_fn"],
        net_arch=[dict(pi=ppo_config["pi"], vf=ppo_config["vf"])],
        log_std_init=ppo_config["log_std_init"],
    ),
    verbose=0,
    tensorboard_log=log_dir,
    n_steps=ppo_config["n_steps"],
    batch_size=ppo_config["batch_size"],
    n_epochs=ppo_config["n_epochs"],
    gamma=ppo_config["gamma"],
    ent_coef=ppo_config["ent_coef"],
)

if pretrained_path:
    print("Loading pretrained model from", pretrained_path)
    try:
        pretrained = PPO.load(pretrained_path)
    except Exception as e:
        print(f"WARNING: could not load pretrained_model '{pretrained_path}': {e}")
        response = input("Train from scratch instead? (y/n): ")
        if response.lower() != "y":
            print("Exiting.")
            exit()
        pretrained = None
    if pretrained is not None:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=pretrained.policy_kwargs,
            verbose=0,
            tensorboard_log=log_dir,
            n_steps=ppo_config["n_steps"],
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            gamma=ppo_config["gamma"],
            ent_coef=ppo_config["ent_coef"],
        )
        model.policy.load_state_dict(pretrained.policy.state_dict())

print("Logging to", log_dir)
print("Saving models to", models_dir)


def train(model, log_name, n=int(1e10)):
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps * env.num_envs * 10
    while model.num_timesteps < n:
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=log_name,
        )
        time_steps = model.num_timesteps
        model.save(models_dir + '/' + log_name + '/' + str(time_steps))
        print('Model saved at', models_dir + '/' + log_name + '/' + str(time_steps))


train(model, name)
