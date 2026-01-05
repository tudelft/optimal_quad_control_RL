from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from quad_race_env import *
from randomization import *
from quadcopter_animation import animation


env = Quadcopter3DGates(
    num_envs=1,
    initialize_at_random_gates=False,
    randomization=randomization_5inch_30_percent,
    pause_if_collision=False
)

model_path = 'models/5inch_drone/run0_5inch_10_percent/1000000.zip'
model = PPO.load(model_path)

def animate_policy(model, env, reset_func=None, **kwargs):
    def run():
        actions, _ = model.predict(env.states, deterministic=True)
        states, rewards, dones, infos = env.step(actions)
        out = env.render()

        return out
    animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, fps=1/env.dt, **kwargs)


animate_policy(model, env)