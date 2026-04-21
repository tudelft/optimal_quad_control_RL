import argparse

from stable_baselines3 import PPO

from quad_race.environment import QuadRace
from quad_race.utils import animate_policy, get_ep_info


parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='Path to the .zip model file')
args = parser.parse_args()

model = PPO.load(args.model_path)

env = QuadRace(
    num_envs=50,
    initialize_on_ground=True,
    initialize_uniform=False,
    initialize_at_random_gates=False,
    pause_if_collision=True,
    loop_gates=True,
    gate_size=0.65,
    pos_jumps_std=0.0,
    pos_jumps_prob=0.0,
    max_steps=3500,
    seed=1,
)

ep_info = get_ep_info(model, env, plotting=False)
animate_policy(model, env)
