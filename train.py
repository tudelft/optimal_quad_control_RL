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
models_dir = 'models/DR_E2E'
log_dir = 'logs/DR_E2E'
video_log_dir = 'videos/DR_E2E'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(video_log_dir):
    os.makedirs(video_log_dir)

# Date and time string for unique folder names
datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# CREATE ENVIRONMENTS
env = Quadcopter3DGates(
    num_envs=100,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=randomization_big,
    gates_ahead=1
)
test_env = Quadcopter3DGates(
    num_envs=10,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    start_pos=start_pos,
    randomization=randomization_big,
    gates_ahead=1,
    # pause_if_collision=True
)

# Wrap the environment in a Monitor wrapper
env = VecMonitor(env)

# MODEL DEFINITION
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[64,64], vf=[64,64])], log_std_init = 0)
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

# ANIMATION FUNCTION
# def animate_policy(model, env, deterministic=False, log_times=False, print_vel=False, log=None, **kwargs):
#     env.reset()
#     def run():
#         actions, _ = model.predict(env.states, deterministic=deterministic)

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
def train(model, test_env, log_name, n=int(1e8)):
    # save every 10 policy rollouts
    TIMESTEPS = model.n_steps*env.num_envs*10
    for i in range(0,n):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name)
        time_steps = model.num_timesteps
        # save model
        model.save(models_dir + '/' + log_name + '/' + str(time_steps))
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
# get name from the passed arguments
if len(sys.argv) > 1:
    name = sys.argv[1]

    # delete
    import shutil
    shutil.rmtree(log_dir + '/' + name + '_0', ignore_errors=True)
    shutil.rmtree(models_dir + '/' + name, ignore_errors=True)
    shutil.rmtree(video_log_dir + '/' + name, ignore_errors=True)
    
    train(model, test_env, name)
else:
    print('Please provide a name for the training run.')
    sys.exit(1)