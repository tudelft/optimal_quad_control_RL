# library imports
import os
import sys
from stable_baselines3 import PPO
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor

# custom imports
from quad_race_env import *
from randomization import *
from quadcopter_animation import animation

# SETUP LOGGING
session_name = 'perception_exp'
models_dir = 'models/'+session_name
log_dir = 'logs/'+session_name
video_log_dir = 'videos/'+session_name

env = Quadcopter3DGates(
    num_envs=100,
    randomization=randomization_dummy_30_percent,
    initialize_at_random_gates=False,
    initialize_on_ground=True,
)

test_env = Quadcopter3DGates(
    num_envs=1,
    randomization=randomization_dummy_30_percent,
    initialize_at_random_gates=False,
    initialize_on_ground=True
)

# Wrap the environment in a Monitor wrapper
env = VecMonitor(env)

# MODEL DEFINITION
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64,64,64], vf=[64,64,64])], log_std_init = 0)
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log=log_dir,
    n_steps=1000,
    batch_size=5000,
    n_epochs=10,
    gamma=0.999
)

print("Model created with policy architecture")
print("-----------------------------------")
print(model.policy)
print("-----------------------------------")

# path_overload = "models/ground_exp/test1/100000000.zip"
path_overload = "models/perception_exp/cool_split4/3000000.zip"
print("overloading weights from", path_overload)
model_old = PPO.load(path_overload)

model.policy.load_state_dict(model_old.policy.state_dict())
print("-----------------------------------")
print(model.policy)
print("-----------------------------------")
print("Logging to", log_dir)
print("Saving models to", models_dir)

# ANIMATION FUNCTION
def animate_policy(model, env, deterministic=False, log_times=False, print_vel=False, log=None, **kwargs):
    env.reset()
    def run():
        actions, _ = model.predict(env.states, deterministic=deterministic)
        
        # print('actions=', actions)
        # print('states=', env.states)
        # print('')

        states, rewards, dones, infos = env.step(actions)
        if log != None:
            log(states)
        if print_vel:
            # compute mean velocity
            vels = env.world_states[:,3:6]
            mean_vel = np.linalg.norm(vels, axis=1).mean()
            print(mean_vel)
        if log_times:
            if rewards[0] == 10:
                print(env.step_counts[0]*env.dt)
        
        return env.render()
    animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, **kwargs)
    
# animate untrained policy (use this to set the recording camera position)
animate_policy(model, test_env, cam_angle=0)
    
# TRAINING
# training loop saves model every 10 policy rollouts and saves a video animation
def train(model, log_name, n=int(1e9)):
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps*env.num_envs*10
    while model.num_timesteps < n:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        time_steps = model.num_timesteps
        # save model
        model.save(models_dir + '/' + log_name + '/' + str(time_steps))
        print('Model saved at', models_dir + '/' + log_name + '/' + str(time_steps))


# RUN TRAINING LOOP
name = 'cool_split5'

import shutil
if os.path.exists(log_dir + '/' + name + '_0'):
    print("Deleting logs...")
    shutil.rmtree(log_dir + '/' + name + '_0', ignore_errors=True)
if os.path.exists(models_dir + '/' + name):
    print("Deleting models...")
    shutil.rmtree(models_dir + '/' + name, ignore_errors=True)
if os.path.exists(video_log_dir + '/' + name):
    print("Deleting videos...")
    shutil.rmtree(video_log_dir + '/' + name, ignore_errors=True)

print("Training model", name)
train(model, name)