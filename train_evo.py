# library imports
import os
import sys
from time import time
from stable_baselines3 import PPO
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor

# custom imports
from evorl.rl.evolve_drone_env import *
from evorl.ctrl_allocation import ctrl_allocation_standard, ctrl_allocation_lambda, ctrl_allocation_alpha
from quadcopter_animation import animation

import argparse

parser = argparse.ArgumentParser(description="Training session configuration")

# Name of the training session
parser.add_argument('session_name', type=str, help='Name of the training session')

# Name of the model
parser.add_argument('name', type=str, help='Name of the model')

# Architecture of the policy (list of integers)
parser.add_argument('--pi', type=int, nargs='+', default=[64, 64, 64], help='Architecture of the policy (e.g., --pi 64 64 64). Default is [64, 64, 64]')

# Architecture of the value function (list of integers)
parser.add_argument('--vf', type=int, nargs='+', default=[64, 64, 64], help='Architecture of the value function (e.g., --vf 64 64 64). Default is [64, 64, 64]')

# State history input length (default is 0)
parser.add_argument('--state_history', type=int, default=0, help='State history input length (default is 0)')

# Action history input length (default is 0)
parser.add_argument('--action_history', type=int, default=0, help='Action history input length (default is 0)')

# History step size (default is 1)
parser.add_argument('--history_step_size', type=int, default=1, help='History step size (default is 1)')

# Transfer learning (default is None)
parser.add_argument('--transfer_learning', type=str, default=None, help='Transfer learning model (default is None)')

# Track radius (default is 1.5)
parser.add_argument('--r', type=float, default=1.5, help='Track radius (default is 1.5)')

# Train timestep
parser.add_argument('--timesteps', type=int, default=1e8, help='Training timestep (default is 1e8)')

# Drone type
parser.add_argument('--drone', type=str, default="standard", help='Drone type (default is standard)')


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
print(f"Transfer learning: {args.transfer_learning}")
# print(f"Parameter input: {args.param_input}")
# print(f"Parameter input noise: {args.param_input_noise}")
# print(f"Randomization: {args.randomization}")

# DEFINE DRONE CONTROL ALLOCATION
if args.drone == "standard":
    print("Training standard drone")
    Bf, Bm = ctrl_allocation_standard()
    
elif args.drone == "lambda":
    print("Training evolved drone")
    Bf, Bm = ctrl_allocation_lambda()
    
elif args.drone == "alpha":
    print("Training evolved drone")
    Bf, Bm = ctrl_allocation_alpha()
    
else:
    raise Exception("Error. Invalid drone type.")


# DEFINE RACE TRACK
r = args.r
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

env = Drone3DGates(
    num_envs=100,
    Bf=Bf,
    Bm=Bm,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    gates_ahead=1, 
    num_state_history=args.state_history,
    num_action_history=args.action_history,
    history_step_size=args.history_step_size,
)
test_env = Drone3DGates(
    num_envs=1,
    Bf=Bf,
    Bm=Bm,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    initialize_at_random_gates=False,
    gates_ahead=1,
    num_state_history=args.state_history,
    num_action_history=args.action_history,
    history_step_size=args.history_step_size,
)

# Wrap the environment in a Monitor wrapper
env = VecMonitor(env)

# MODEL DEFINITION
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

# model = PPO(
#     "MlpPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     verbose=0,
#     tensorboard_log=log_dir,
#     n_steps=500,
#     batch_size=5000,
#     n_epochs=10,
#     gamma=0.999
# )


# Load model for transfer learning
if args.transfer_learning is not None:
    model = PPO.load('models' + '/' + args.transfer_learning, env=env, tensorboard_log=log_dir)
    # model.num_timesteps = 0

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



# TRAINING
# training loop saves model every 10 policy rollouts and saves a video animation
def train(model, test_env, log_name, n=int(args.timesteps)):
    start_time = time()
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps*env.num_envs*10
    # Reset timesteps for 1st loop
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name=log_name)
    time_steps = model.num_timesteps
    # save model
    model.save(models_dir + '/' + log_name + '/' + str(time_steps))
    # print('Model saved at', models_dir + '/' + log_name + '/' + str(time_steps))
    print(f'Time elapsed: {(time() - start_time):.2f}s')
    print(f'Model saved at {models_dir}/{log_name}/{str(time_steps)}')
    while model.num_timesteps < n:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        time_steps = model.num_timesteps
        # save model
        model.save(models_dir + '/' + log_name + '/' + str(time_steps))
        # print('Model saved at', models_dir + '/' + log_name + '/' + str(time_steps))
        print(f'Time elapsed: {(time() - start_time):.2f}s')
        print(f'Model saved at {models_dir}/{log_name}/{str(time_steps)}')
        # save policy animation
        # animate_policy(
        #     model,
        #     test_env,
        #     record_steps=1200,    env.reset()
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
if os.path.exists(log_dir + '/' + name + '_1'):
    print("Deleting logs...")
    shutil.rmtree(log_dir + '/' + name + '_1', ignore_errors=True)
if os.path.exists(models_dir + '/' + name):
    print("Deleting models...")
    shutil.rmtree(models_dir + '/' + name, ignore_errors=True)
if os.path.exists(video_log_dir + '/' + name):
    print("Deleting videos...")
    shutil.rmtree(video_log_dir + '/' + name, ignore_errors=True)

print("Training model", name)
train(model, test_env, name)

# TESTING
test_env.reset()

# do 1000 steps and print state and action
for i in range(1000):
    num = test_env.num_state_history+1
    state_len = int(len(test_env.states[0])/num)
    actions, _ = model.predict(test_env.states, deterministic=True)
    states, rewards, dones, infos = test_env.step(actions)
    
    if (i+1)%500 == 0:
        print('step', i)
        # for j in range(num):
            # print('state', j, '=', test_env.states[0][j*state_len:(j+1)*state_len])
        # print('actions=', actions[0])
        print('rewards=', rewards)
        print('Num gates passed=', infos)
        print('')


