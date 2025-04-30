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

import argparse

parser = argparse.ArgumentParser(description="Training session configuration")

# Name of the training session
parser.add_argument('session_name', type=str, help='Name of the training session')

# Name of the model
parser.add_argument('name', type=str, help='Name of the model')

# Architecture of the policy (list of integers)
parser.add_argument('--pi', type=int, nargs='+', default=[64, 64], help='Architecture of the policy (e.g., --pi 64 64). Default is [64, 64]')

# Architecture of the value function (list of integers)
parser.add_argument('--vf', type=int, nargs='+', default=[64, 64], help='Architecture of the value function (e.g., --vf 64 64). Default is [64, 64]')

# State history input length (default is 0)
parser.add_argument('--state_history', type=int, default=0, help='State history input length (default is 0)')

# Action history input length (default is 0)
parser.add_argument('--action_history', type=int, default=0, help='Action history input length (default is 0)')

# History step size (default is 1)
parser.add_argument('--history_step_size', type=int, default=1, help='History step size (default is 1)')

# Param input (boolean, default is False)
parser.add_argument('--param_input', action='store_true', help='Use parameter input')

# Param input noise (default = 0.0)
parser.add_argument('--param_input_noise', type=float, default=0.0, help='Parameter input noise')

# Randomization (randomized, fixed_5inch, fixed_3inch)
parser.add_argument('--randomization', type=str, default='randomized', help='Randomization (randomized, fixed_5inch, fixed_3inch)')

# Parse the arguments
args = parser.parse_args()

# print summary of the arguments
print("Training session configuration:")
print(f"Session name: {args.session_name}")
print(f"Model name: {args.name}")
print(f"Policy architecture: {args.pi}")
print(f"Value function architecture: {args.vf}")
print(f"State history input length: {args.state_history}")
print(f"Action history input length: {args.action_history}")
print(f"History step size: {args.history_step_size}")
print(f"Parameter input: {args.param_input}")
print(f"Parameter input noise: {args.param_input_noise}")
print(f"Randomization: {args.randomization}")


# DEFINE RACE TRACK
r = 1.5
gate_pos = np.array([
    [ r,  -r, -1.5],
    [ 0,   0, -1.5],
    [-r,   r, -1.5],
    [ 0, 2*r, -1.5],
    [ r,   r, -1.5],
    [ 0,   0, -1.5],
    [-r,  -r, -1.5],
    [ 0,-2*r, -1.5]
])
gate_yaw = np.array([1,2,1,0,-1,-2,-1,0])*np.pi/2
start_pos = gate_pos[0] + np.array([0,-1.,0])

# SETUP LOGGING
models_dir = 'models/'+args.session_name
log_dir = 'logs/'+args.session_name
video_log_dir = 'videos/'+args.session_name

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(video_log_dir):
    os.makedirs(video_log_dir)

# Date and time string for unique folder names
datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# CREATE ENVIRONMENTS
if args.randomization == 'randomized':
    randomization = randomization_big
elif args.randomization == 'fixed_5inch':
    randomization = randomization_fixed_params_5inch
elif args.randomization == '5inch_10_percent':
    randomization = randomization_5inch_10_percent
elif args.randomization == '5inch_20_percent':
    randomization = randomization_5inch_20_percent
elif args.randomization == '5inch_30_percent':
    randomization = randomization_5inch_30_percent
elif args.randomization == 'fixed_3inch':
    randomization = randomization_fixed_params_3inch
elif args.randomization == '3inch_10_percent':
    randomization = randomization_3inch_10_percent
elif args.randomization == '3inch_20_percent':
    randomization = randomization_3inch_20_percent
elif args.randomization == '3inch_30_percent':
    randomization = randomization_3inch_30_percent
else:
    print("Randomization not recognized")
    # kill the process
    sys.exit()

env = Quadcopter3DGates(
    num_envs=100,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=randomization,
    gates_ahead=1, 
    num_state_history=args.state_history,
    num_action_history=args.action_history,
    history_step_size=args.history_step_size,
    param_input=args.param_input,
    param_input_noise=args.param_input_noise
)
test_env = Quadcopter3DGates(
    num_envs=1,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=randomization,
    gates_ahead=1,
    num_state_history=args.state_history,
    num_action_history=args.action_history,
    history_step_size=args.history_step_size,
    param_input=args.param_input,
    param_input_noise=args.param_input_noise
)

# Wrap the environment in a Monitor wrapper
env = VecMonitor(env)

# MODEL DEFINITION
# policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64,64], vf=[64,64])], log_std_init = 0)
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=args.pi, vf=args.vf)], log_std_init = 0)
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

print("Model created with policy architecture", args.pi, "and value function architecture", args.vf)
print("-----------------------------------")
print(model.policy)
print("-----------------------------------")
print("Logging to", log_dir)
print("Saving models to", models_dir)
print("Saving videos to", video_log_dir)

# ANIMATION FUNCTION
# def animate_policy(model, env, deterministic=False, log_times=False, print_vel=False, log=None, **kwargs):
#     env.reset()
#     def run():
#         actions, _ = model.predict(env.states, deterministic=deterministic)
        
#         # print('actions=', actions)
#         # print('states=', env.states)
#         # print('')

#         states, rewards, dones, infos = env.step(actions)
#         if log != None:
#             log(states)
#         if print_vel:
#             # compute mean velocity
#             vels = env.world_states[:,3:6]
#             mean_vel = np.linalg.norm(vels, axis=1).mean()
#             print(mean_vel)
#         if log_times:
#             if rewards[0] == 10:
#                 print(env.step_counts[0]*env.dt)
        
#         return env.render()
#     animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, **kwargs)
    
# animate untrained policy (use this to set the recording camera position)
# animate_policy(model, test_env)

# TESTING
test_env.reset()

# do 100 steps and print state and action
for i in range(100):
    print('step', i)
    num = test_env.num_state_history+1
    state_len = int(len(test_env.states[0])/num)
    for j in range(num):
        print('state', j, '=', test_env.states[0][j*state_len:(j+1)*state_len])
    actions, _ = model.predict(test_env.states, deterministic=True)
    states, rewards, dones, infos = test_env.step(actions)
    print('actions=', actions[0])
    print('')
    
# TRAINING
# training loop saves model every 10 policy rollouts and saves a video animation
def train(model, test_env, log_name, n=int(1e8)):
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps*env.num_envs*10
    while model.num_timesteps < n:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        time_steps = model.num_timesteps
        # save model
        model.save(models_dir + '/' + log_name + '/' + str(time_steps))
        print('Model saved at', models_dir + '/' + log_name + '/' + str(time_steps))
        # save policy animation
        # animate_policy(
        #     model,
        #     test_env,
        #     record_steps=1200,
        #     record_file=video_log_dir + '/' + log_name + '/' + str(time_steps) + '.mp4',
        #     show_window=False
        # )
        

# name = 'figure8_64_64_again!'
# import shutil
# shutil.rmtree(log_dir + '/' + name + '_0', ignore_errors=True)
# shutil.rmtree(models_dir + '/' + name, ignore_errors=True)
# shutil.rmtree(video_log_dir + '/' + name, ignore_errors=True)

# RUN TRAINING LOOP
name = args.name

# check if model already exists
# if os.path.exists(models_dir + '/' + name):
#     print(f"Model {name} already exists. Do you want to overwrite it (this will delete the existing model/logs/videos)? (y/n)")
    
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
train(model, test_env, name)