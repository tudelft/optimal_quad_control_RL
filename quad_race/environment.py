import torch
import stable_baselines3
import sys
import numpy as np
from sympy import *
import jax
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
import json


print("python version:", sys.version)
print("stable_baselines3 version:", stable_baselines3.__version__)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")
jax_device = jax.devices('cpu')[0]  # Cuda is slower jax_device = jax.devices('cuda')[0]
print("jax_device:", jax_device)

# set torch default device
torch.set_default_device(device)

# Equations of motion 3D quadcopter from https://arxiv.org/pdf/2304.13460.pdf
state = symbols('x y z v_x v_y v_z qw qx qy qz p q r w1 w2 w3 w4')
x,y,z,vx,vy,vz,qw,qx,qy,qz,p,q,r,w1,w2,w3,w4 = state
control = symbols('U_1 U_2 U_3 U_4')    # normalized motor commands between [-1,1]
u1,u2,u3,u4 = control

g = 9.81
params = symbols('k_x, k_x2, k_y, k_y2, k_w, k_angle, k_hor, k_v2, k_p1, k_p2, k_p3, k_p4, Jx, k_q1, k_q2, k_q3, k_q4, Jy, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, Jz, tau, k, w_min, w_max')
# k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max = params
k_x, k_x2, k_y, k_y2, k_w, k_angle, k_hor, k_v2, k_p1, k_p2, k_p3, k_p4, Jx, k_q1, k_q2, k_q3, k_q4, Jy, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, Jz, tau, k, w_min, w_max = params

# Rotation matrix 
# Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
# Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
# Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
# R = Rz*Ry*Rx
quat = Quaternion(qw, qx, qy, qz)
quat_inv = Quaternion(qw, -qx, -qy, -qz)

# Body velocity
# vbx, vby, vbz = R.T@Matrix([vx,vy,vz])
vbx, vby, vbz = Quaternion.rotate_point([vx,vy,vz], quat_inv)

# euler angles
# phi, theta, psi = quat.to_euler('xyz')
phi = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
theta = asin(2*(qw*qy - qz*qx))
psi = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

# normalized motor speeds to rad/s
w_min_n = 0.
w_max_n = 3000.
W1 = (w1+1)/2*(w_max_n-w_min_n) + w_min_n
W2 = (w2+1)/2*(w_max_n-w_min_n) + w_min_n
W3 = (w3+1)/2*(w_max_n-w_min_n) + w_min_n
W4 = (w4+1)/2*(w_max_n-w_min_n) + w_min_n

# motor commands scaled to [0,1]
U1 = (u1+1)/2
U2 = (u2+1)/2
U3 = (u3+1)/2
U4 = (u4+1)/2

# first order delay:
# the steadystate rpm motor response to the motor command U is described by:
# Wc = (w_max-w_min)*sqrt(k U**2 + (1-k)*U) + w_min
Wc1 = (w_max-w_min)*sqrt(k*U1**2 + (1-k)*U1) + w_min
Wc2 = (w_max-w_min)*sqrt(k*U2**2 + (1-k)*U2) + w_min
Wc3 = (w_max-w_min)*sqrt(k*U3**2 + (1-k)*U3) + w_min
Wc4 = (w_max-w_min)*sqrt(k*U4**2 + (1-k)*U4) + w_min

# rad/s
d_W1 = (Wc1 - W1)/tau
d_W2 = (Wc2 - W2)/tau
d_W3 = (Wc3 - W3)/tau
d_W4 = (Wc4 - W4)/tau

# normalized motor speeds d/dt[W - w_min_n)/(w_max_n-w_min_n)*2 - 1]
d_w1 = d_W1/(w_max_n-w_min_n)*2
d_w2 = d_W2/(w_max_n-w_min_n)*2
d_w3 = d_W3/(w_max_n-w_min_n)*2
d_w4 = d_W4/(w_max_n-w_min_n)*2

# Thrust and Drag
omega2 = W1**2 + W2**2 + W3**2 + W4**2
omega_average = (W1 + W2 + W3 + W4)/4
radius = 0.12954*0.75/2
denominator = radius*omega_average
v_horizontal = sqrt(vbx**2 + vby**2)
angle_of_attack = atan2(vbz, denominator)
mu_x = vbx/(radius*omega_average)
mu_y = vby/(radius*omega_average)
mu_xx_yy = atan2(v_horizontal**2, denominator**2)
v2 = vbz*abs(vbz)

T = -k_w*(1+angle_of_attack*k_angle + k_hor*mu_xx_yy)*omega2 - k_v2*v2
Dx = -k_x*vbx*(W1+W2+W3+W4)-k_x2*vbx*abs(vbx)
Dy = -k_y*vby*(W1+W2+W3+W4)-k_y2*vby*abs(vby)

# Moments
Mx = -k_p1*W1**2 - k_p2*W2**2 + k_p3*W3**2 + k_p4*W4**2 + Jx*q*r
My = -k_q1*W1**2 + k_q2*W2**2 - k_q3*W3**2 + k_q4*W4**2 + Jy*p*r
Mz = -k_r1*W1 + k_r2*W2 + k_r3*W3 - k_r4*W4 - k_r5*d_W1 + k_r6*d_W2 + k_r7*d_W3 - k_r8*d_W4 + Jz*p*q

# Dynamics
d_x = vx
d_y = vy
d_z = vz

# d_vx, d_vy, d_vz = Matrix([0,0,g]) + R@Matrix([Dx, Dy,T])
d_vx, d_vy, d_vz = Matrix([0,0,g]) + Matrix(Quaternion.rotate_point([Dx, Dy,T], quat))

# d_phi   = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
# d_theta = q*cos(phi) - r*sin(phi)
# d_psi   = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)
d_quat = 0.5*quat*Quaternion(0,p,q,r)
d_qw = d_quat.a
d_qx = d_quat.b
d_qy = d_quat.c
d_qz = d_quat.d

d_p     = Mx
d_q     = My
d_r     = Mz

# State space model
f = [d_x, d_y, d_z, d_vx, d_vy, d_vz, d_qw, d_qx, d_qy, d_qz, d_p, d_q, d_r, d_w1, d_w2, d_w3, d_w4]

# lambdify
f_func = jax.jit(lambdify((Array(state), Array(control), Array(params)), Array(f), 'jax'), device=jax_device)

# PERCEPTION REWARD
# function for calculating perception angle
cam_angle = symbols('cam_angle')
# body x-axis
x_body_axis = Matrix([1,0,0])
# optical axis
optical_axis = Matrix([cos(cam_angle), 0, -sin(cam_angle)]) # checked!
# gate pos in world frame
gx, gy, gz = symbols('gx gy gz')
# gate pos projected in the xy body frame
# gate_pos_B = R.T@(Matrix([gx,gy,gz]) - Matrix([x,y,z])) # checked
gate_pos_B = Matrix(Quaternion.rotate_point(Matrix([gx,gy,gz]) - Matrix([x,y,z]), quat_inv))
# gate_pos_B[2] = 0
# get angle between optical_axis and gate_pos_B
perc_angle = acos(optical_axis.dot(gate_pos_B)/(optical_axis.norm()*(gate_pos_B.norm()))) # checked (but only use when 0.1m away from the gates)
# lambdify
get_perc_angle_func = lambda a: lambdify((Array(state), Array([gx,gy,gz])), perc_angle.subs(cam_angle, a), 'jax')


# GET ACCELEROMETER
get_accelerometer = jax.jit(lambdify((Array(state), Array(params)), Array([Dx, Dy, T]), 'jax'), device=jax_device)

# GET EULER
get_euler = jax.jit(lambdify((Array(state),), Array([phi, theta, psi]), 'jax'), device=jax_device)
get_yaw = jax.jit(lambdify((Array(state),), Array([psi]), 'jax'), device=jax_device)
phi_, theta_, psi_= symbols('phi theta psi')
quat_ = Quaternion.from_euler([phi_, theta_, psi_], 'xyz')
qw_, qx_, qy_, qz_ = quat_.a, quat_.b, quat_.c, quat_.d
get_quat_from_euler = jax.jit(lambdify((Array([phi_, theta_, psi_]),), Array([qw_, qx_, qy_, qz_]), 'jax'), device=jax_device)

# GET ROTATION MATRIX
R = Matrix([
    [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
    [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
    [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]
])
# get first and last columns of R
get_R6 = jax.jit(lambdify((Array(state),), Array([R[0,0], R[1,0], R[2,0], R[0,2], R[1,2], R[2,2]]), 'jax'), device=jax_device)

# PARAMETER ENCODING (used for parameter input)
# normalize thrust and moment constants by scaling with w_max
k_wn = k_w*(w_max**2)
k_pn = (k_p1 + k_p2 + k_p3 + k_p4)/4 * (w_max**2)
k_qn = (k_q1 + k_q2 + k_q3 + k_q4)/4 * (w_max**2)
k_rn = (k_r1 + k_r2 + k_r3 + k_r4)/4 * (w_max)
k_rdn = (k_r5 + k_r6 + k_r7 + k_r8)/4 * (w_max)

# normalize to [-1,1] based on min and max expected values
normalize = lambda x, x_min, x_max: 2*(x - x_min)/(x_max - x_min) - 1
k_w_encoding    = normalize(k_wn,       1.0e+01,    3.0e+01)
k_p_encoding    = normalize(k_pn,       2.0e+02,    8.0e+02)
k_q_encoding    = normalize(k_qn,       2.0e+02,    8.0e+02)
k_r_encoding    = normalize(k_rn,       2.0e+01,    8.0e+01)
k_rd_encoding   = normalize(k_rdn,      2.0e+00,    8.0e+00)
k_encoding      = normalize(k,          0.,         1.)
tau_encoding    = normalize(tau,        0.01,       0.1)
w_min_encoding  = normalize(w_min,      0,          500)
w_max_encoding  = normalize(w_max,      3000,       5000)

# lambdify
param_encoding = jax.jit(lambdify((Array(params),), Array([k_w_encoding, k_p_encoding, k_q_encoding,
                                                           k_r_encoding, k_rd_encoding, k_encoding, tau_encoding,
                                                           w_min_encoding, w_max_encoding]), 'jax'), device=jax_device)

def load_config(source = 'options.json'):
    global params_config

    with open(source, 'r') as file:
        params_config = json.load(file)

load_config()

def randomization_A2RL_april_fine_tuned_json(num):    
    # Randomize parameters
    params = {}
    for key, (value, percentage) in params_config['ModelParams'].items():
        lower_bound = value * (1 - percentage / 100)
        upper_bound = value * (1 + percentage / 100)
        if key == 'k':  # Special handling for 'k'
            lower_bound = max(0, lower_bound)
            upper_bound = min(1, upper_bound)
        params[key] = np.random.uniform(lower_bound, upper_bound, size=num)
    
    return params

# Extracting values from the loaded configuration
env_config = params_config['EnvironmentConfig']

cam_angle_degrees_json = env_config['cam_angle_degrees']
cam_angle_rad_json = cam_angle_degrees_json * np.pi / 180
speed_limit_json = env_config['speed_limit']
gate_size_json = env_config['gate_size']

num_envs_json = env_config['num_envs']
pause_if_collision_json = env_config['pause_if_collision']
pos_noise_json = env_config['pos_noise']
vel_noise_json = env_config['vel_noise']
pos_jumps_std_json = env_config['pos_jumps_std']
vel_offset_std_json = env_config['vel_offset_std']
initialize_on_ground_json = env_config['initialize_on_ground']
initialize_at_random_gates_json = env_config['initialize_at_random_gates']
initialize_uniform_json = env_config['initialize_uniform']
loop_gates_json = env_config['loop_gates']

progress_reward_json = env_config.get('progress_reward', 1.0)
gate_reward_json = env_config.get('gate_reward', 1.0)
angular_rate_penalty_json = env_config.get('angular_rate_penalty', 0.001)
gate_offset_penalty_json = env_config.get('gate_offset_penalty', 1.0)
perception_penalty_json = env_config.get('perception_penalty', 0.0)
motor_penalty_json = env_config.get('motor_penalty', 0.0)
motor_penalty_threshold_json = env_config.get('motor_penalty_threshold', 0.0)
low_action_penalty_json = env_config.get('low_action_penalty', 0.0)
crash_penalty_json = env_config.get('crash_penalty', 10.0)

ground_height_json = env_config.get('ground_height', 0.0)
v_ground_json = env_config.get('v_ground', 2.0)
gate_thickness_json = env_config.get('gate_thickness', 1.0)


class Flightplan:
    def __init__(self, flight_plan_json_file=env_config['flight_plan']):

        # load the flight plan
        with open(flight_plan_json_file, 'r') as f:
            flight_plan_json = json.load(f)

        start = np.array(flight_plan_json['start'])
        gates = np.array(flight_plan_json['gates'])
        bounds_xy = flight_plan_json['bounds_xy']

        if not isinstance(bounds_xy, list):
            self.bounds_xy = None
        else:
            self.bounds_xy = np.array(bounds_xy)


        # degrees to radians
        start[3] *= np.pi/180
        gates[:,3] *= np.pi/180

        self.name = flight_plan_json['name']
        self.num_gates_recovery = flight_plan_json['num_gates_recovery']
        self.gate_pos = gates[:,0:3].astype(np.float32)
        self.gate_yaw = gates[:,3].astype(np.float32)
        self.start_pos = start[0:3].astype(np.float32)
        self.num_gates = self.gate_pos.shape[0]
        self.fake_gates = [] # .astype(np.float32)


    def __repr__(self):
        return f"Flightplan:\n\t(gate_pos={self.gate_pos}\n\gate_yaw={self.gate_yaw}\n\start_pos={self.start_pos})\n\tfake_gates={self.fake_gates}\n\tbounds_xy={self.bounds_xy}"
    
flightplan_json = Flightplan(env_config['flight_plan'])


class QuadRace(VecEnv):
    def __init__(self,
                 speed_limit=speed_limit_json,
                 cam_angle=cam_angle_rad_json,
                 randomization=randomization_A2RL_april_fine_tuned_json,
                 pos_jumps_std=pos_jumps_std_json,
                 pos_jumps_prob=0.0,
                 vel_offset_std=vel_offset_std_json,
                 progress_reward=progress_reward_json,
                 gate_reward=gate_reward_json,
                 angular_rate_penalty=angular_rate_penalty_json,
                 gate_offset_penalty=gate_offset_penalty_json,
                 perception_penalty=perception_penalty_json,
                 motor_penalty=motor_penalty_json,
                 motor_penalty_threshold=motor_penalty_threshold_json,
                 low_action_penalty=low_action_penalty_json,
                 crash_penalty=crash_penalty_json,
                 initialize_uniform=initialize_uniform_json,
                 initialize_on_ground=initialize_on_ground_json,
                 initialize_at_random_gates=initialize_at_random_gates_json,
                 loop_gates=loop_gates_json,
                 gate_size=gate_size_json,
                 gate_thickness=gate_thickness_json,
                 flightplan=flightplan_json,
                 ground_height=ground_height_json,
                 v_ground=v_ground_json,
                 gates_ahead=1,
                 motor_limit=1.0,
                 num_state_history=0,
                 num_action_history=0,
                 history_step_size=1,
                 param_input=False,
                 param_input_noise=0.,
                 seed=None,
                 disable_collision=False,
                 pause_if_collision=False,
                 disable_gate_collision=False,
                 pos_noise=0.0,
                 vel_noise=0.0,
                 R6_input=False,
                 max_steps=2000,
                 num_envs=100,
                 dt = 0.01,
                 ):
        # speed limit
        self.speed_limit = speed_limit

        # Reward coefficients
        self.progress_reward = progress_reward
        self.gate_reward = gate_reward
        self.angular_rate_penalty = angular_rate_penalty
        self.gate_offset_penalty = gate_offset_penalty
        self.perception_penalty = perception_penalty
        self.motor_penalty = motor_penalty
        self.motor_penalty_threshold = motor_penalty_threshold
        self.low_action_penalty = low_action_penalty
        self.crash_penalty = crash_penalty

        # loop gates
        self.loop_gates = loop_gates
        
        # pos and vel noise
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.pos_jumps_std = pos_jumps_std
        self.pos_jumps_prob = pos_jumps_prob
        self.vel_offset_std = vel_offset_std
        
        # cam angle
        self.cam_angle = cam_angle
        self.perc_angle_func = jax.jit(get_perc_angle_func(self.cam_angle), device=jax_device)
        
        # set seed
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # Define the race track
        self.start_pos = flightplan.start_pos
        self.gate_pos = flightplan.gate_pos
        self.gate_yaw = flightplan.gate_yaw
        self.num_gates = flightplan.num_gates
        self.gates_ahead = gates_ahead
        # Define bounds
        self.bounds_xy = flightplan.bounds_xy
        self.gate_size = np.ones(self.num_gates)*gate_size
        self.gate_thickness = gate_thickness
        self.ground_height = ground_height
        self.v_ground = v_ground

        self.disable_collision = disable_collision
        self.disable_gate_collision = disable_gate_collision
        # fake gates
        self.fake_gates = flightplan.fake_gates
        
        # Pause if collision
        self.pause_if_collision = pause_if_collision

        # Domain randomization
        def rand_f(n):
            param_dict = randomization(n)
            return np.array([param_dict[p.name] for p in params], dtype=np.float32).T
        
        self.randomization = rand_f
        self.params = self.randomization(num_envs)
        
        # Motor limit
        self.motor_limit = motor_limit

        # Initialization
        self.initialize_at_random_gates = initialize_at_random_gates
        self.initialize_on_ground = initialize_on_ground
        self.initialize_uniform = initialize_uniform
        
        # these 3 initialization options are mutually exclusive
        if self.initialize_on_ground and self.initialize_at_random_gates:
            raise ValueError('initialize_on_ground and initialize_at_random_gates are mutually exclusive')
        if self.initialize_on_ground and self.initialize_uniform:
            raise ValueError('initialize_on_ground and initialize_uniform are mutually exclusive')
        if self.initialize_at_random_gates and self.initialize_uniform:
            raise ValueError('initialize_at_random_gates and initialize_uniform are mutually exclusive')

        # state, action history
        self.num_state_history = num_state_history
        self.num_action_history = num_action_history
        self.history_step_size = history_step_size
        
        # param input
        self.param_input = param_input
        self.param_input_noise = param_input_noise
        # compute encoding with noise on the params
        if self.param_input:
            params_with_noise = self.params*np.random.uniform(1-self.param_input_noise, 1+self.param_input_noise, size=self.params.shape)
            self.param_encoding = param_encoding(params_with_noise.T).T
             
        # R6 input
        self.R6_input = R6_input
                   
        # Calculate relative gates
        # pos,yaw of gate i in reference frame of gate i-1 (assumes a looped track)
        self.gate_pos_rel = np.zeros((self.num_gates,3), dtype=np.float32)
        self.gate_yaw_rel = np.zeros(self.num_gates, dtype=np.float32)
        for i in range(0,self.num_gates):
            self.gate_pos_rel[i] = self.gate_pos[i] - self.gate_pos[i-1]
            # Rotation matrix
            R = np.array([
                [np.cos(self.gate_yaw[i-1]), np.sin(self.gate_yaw[i-1])],
                [-np.sin(self.gate_yaw[i-1]), np.cos(self.gate_yaw[i-1])]
            ], dtype=np.float32)
            self.gate_pos_rel[i,0:2] = R@self.gate_pos_rel[i,0:2]
            self.gate_yaw_rel[i] = self.gate_yaw[i] - self.gate_yaw[i-1]
            # wrap yaw
            self.gate_yaw_rel[i] %= 2*np.pi
            if self.gate_yaw_rel[i] > np.pi:
                self.gate_yaw_rel[i] -= 2*np.pi
            elif self.gate_yaw_rel[i] < -np.pi:
                self.gate_yaw_rel[i] += 2*np.pi

        # Define the target gate for each environment
        self.target_gates = np.zeros(num_envs, dtype=np.int32)

        # action space: [cmd1, cmd2, cmd3, cmd4]
        # U = (u+1)/2 --> u = 2U-1
        u_lim = 2*self.motor_limit-1
        action_space = spaces.Box(low=-1, high=u_lim, shape=(4,))

        # observation space: pos[G], vel[G], att[quatB->G], rates[B], rpms, future_gates[G], future_gate_dirs[G]
        # [G] = reference frame aligned with target gate
        # [B] = body frame
        self.state_len = 16+4*self.gates_ahead+4*self.num_action_history+9*self.param_input+3*self.R6_input
        self.obs_len = self.state_len*(1+self.num_state_history)
        observation_space = spaces.Box(
            low  = np.array([-np.inf]*self.obs_len, dtype=np.float32),
            high = np.array([ np.inf]*self.obs_len, dtype=np.float32)
        )

        # Initialize the VecEnv
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # world state: pos[W], vel[W], att[eulerB->W], rates[B], rpms
        self.world_states = np.zeros((num_envs,17), dtype=np.float32)
        # observation state
        self.states = np.zeros((num_envs,self.obs_len), dtype=np.float32)
        # state history tracking
        num_hist = 1+self.num_state_history
        self.state_hist = np.zeros((num_envs,num_hist,self.state_len), dtype=np.float32)
        # action history tracking
        self.action_hist = np.zeros((num_envs,num_hist,4), dtype=np.float32)

        # Define any other environment-specific parameters
        self.max_steps = max_steps      # Maximum number of steps in an episode
        self.dt = np.float32(dt) # Time step duration

        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.actions = np.zeros((num_envs,4), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs,4), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=np.bool_)
        self.final_gate_passed = np.zeros(num_envs, dtype=np.bool_)

        self.accelerometer = np.zeros((self.num_envs,3), dtype=np.float32)
        self.gyro = np.zeros((self.num_envs,3), dtype=np.float32)
        
        # velocity offset 
        self.vel_offset = np.zeros((self.num_envs,3), dtype=np.float32)
        
        self.pause = False
    
    def reset_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def update_states(self):
        # Transform pos and vel in gate frame
        gate_pos = self.gate_pos[self.target_gates%self.num_gates]
        gate_yaw = self.gate_yaw[self.target_gates%self.num_gates]

        # Rotation matrix from world frame to gate frame
        R = np.array([
            [np.cos(gate_yaw), np.sin(gate_yaw)],
            [-np.sin(gate_yaw), np.cos(gate_yaw)]
        ], dtype=np.float32).transpose((2,1,0))

        # new state array to prevent the weird bug related to indexing ([:] syntax)
        new_states = np.zeros((self.num_envs,self.state_len), dtype=np.float32)

        # Update positions
        pos_W = self.world_states[:,0:3]
        pos_G = (pos_W[:,np.newaxis,0:2] - gate_pos[:,np.newaxis,0:2]) @ R
        new_states[:,0:2] = pos_G[:,0,:]
        new_states[:,2] = pos_W[:,2] - gate_pos[:,2]
        # add noise to position gaussian noise
        if self.pos_noise > 0:
            new_states[:,0:3] += np.random.normal(0,self.pos_noise,size=(self.num_envs,3))
        

        # Update velocities
        vel_W = self.world_states[:,3:6]
        vel_G = (vel_W[:,np.newaxis,0:2]) @ R
        new_states[:,3:5] = vel_G[:,0,:]
        new_states[:,5] = vel_W[:,2]
        # add noise to velocity gaussian noise
        if self.vel_noise > 0:
            new_states[:,3:6] += np.random.normal(0,self.vel_noise,size=(self.num_envs,3))
        # add offset to velocity
        if (self.vel_offset > 0).any():
            new_states[:,3:6] += self.vel_offset

        # Update attitude
        if self.R6_input:
            R6 = np.array(jax.device_get(get_R6(self.world_states.T).block_until_ready()), dtype=np.float32).T
            new_states[:,6:12] = R6

            nx_xy = R6[:, np.newaxis, 0:2] @ R
            nz_xy = R6[:, np.newaxis, 3:5] @ R
            new_states[:,6:8]  = nx_xy[:,0,:]
            new_states[:,9:11] = nz_xy[:,0,:]
        else:
            phi_, theta_, psi_ = np.array(jax.device_get(get_euler(self.world_states.T).block_until_ready()), dtype=np.float32)
            psi_ -= gate_yaw
            # make sure euler angles are in the range [-pi,pi]
            phi_ = (phi_ + np.pi) % (2 * np.pi) - np.pi
            psi_ = (psi_ + np.pi) % (2 * np.pi) - np.pi
            theta_ = (theta_ + np.pi) % (2 * np.pi) - np.pi
            new_states[:,6] = phi_
            new_states[:,7] = theta_
            new_states[:,8] = psi_

        shift = 3*self.R6_input
        
        # Update rates
        new_states[:,9+shift:12+shift] = self.world_states[:,10:13]

        # Update rpms
        new_states[:,12+shift:16+shift] = self.world_states[:,13:17]

        # Update future gates relative to current gate ([0,0,0,0] for out of bounds)
        for i in range(self.gates_ahead):
            indices = (self.target_gates+i+1)
            if self.loop_gates:
                # loop when out of bounds
                indices = indices % self.num_gates
            valid = indices < self.num_gates
            new_states[valid,16+4*i+shift:16+4*i+3+shift] = self.gate_pos_rel[indices[valid]]
            new_states[valid,16+4*i+3+shift] = self.gate_yaw_rel[indices[valid]]

        # update action history
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:,0] = self.actions
        
        for i in range(self.num_action_history):
            new_states[:,16+4*self.gates_ahead+4*i+shift:16+4*self.gates_ahead+4*i+4+shift] = self.action_hist[:,(i+1)*self.history_step_size-1]
        
        # update param encoding
        if self.param_input:
            new_states[:,16+4*self.gates_ahead+4*self.num_action_history+shift:] = self.param_encoding
            
        # update state history
        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:,0] = new_states

        # stack history up to self.num_state_history
        self.states = self.state_hist[:,0:(self.num_state_history+1)*self.history_step_size:self.history_step_size].reshape((self.num_envs,-1))

    def reset_(self, dones):
        num_reset = dones.sum()
        
        if self.initialize_at_random_gates:
            # set target gates to random gates
            self.target_gates[dones] = np.random.randint(0,self.num_gates, size=num_reset)
            # set position to 1m in front of the target gate
            # gate_pos + [cos(gate_yaw), sin(gate_yaw), 0]
            pos = self.gate_pos[self.target_gates[dones]%self.num_gates]
            yaw = self.gate_yaw[self.target_gates[dones]%self.num_gates]
            
            pos = pos - 2*np.array([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], dtype=np.float32).T
            x0, y0, z0 = pos.T
        elif self.initialize_uniform:
            if self.bounds_xy is None:
                raise ValueError('Cannot initialize uniform when bounds_xy is None')
            # x,y randomly distributed in the bounds
            x0 = np.random.uniform(self.bounds_xy[0,0],self.bounds_xy[0,1], size=(num_reset,))
            y0 = np.random.uniform(self.bounds_xy[1,0],self.bounds_xy[1,1], size=(num_reset,))
            z0 = np.random.uniform(-5,0, size=(num_reset,))
            
            # for every gate we calculate the distance to the drone + whether the drone is behind the gate
            dist_to_gate = np.zeros((num_reset,self.num_gates))
            behind_gate = np.zeros((num_reset,self.num_gates), dtype=np.bool_)
            for i in range(self.num_gates):
                pos = self.gate_pos[i]
                yaw = self.gate_yaw[i]
                dist_to_gate[:,i] = np.linalg.norm(np.stack([x0-pos[0], y0-pos[1]], axis=1), axis=1)
                behind_gate[:,i] = ((np.cos(yaw)*(x0-pos[0]) + np.sin(yaw)*(y0-pos[1])) < 0)
            
            # find closest gate in front of the drone
            closest_gate = np.zeros(num_reset, dtype=np.int32)
            for i in range(num_reset):
                # if drone is not behind any gate, pick the closest gate
                if not behind_gate[i].any():
                    closest_gate[i] = np.argmin(dist_to_gate[i])
                else:
                    # if drone is behind a gate, pick the from behind gates
                    idx = np.argmin(dist_to_gate[i][behind_gate[i]])
                    closest_gate[i] = np.where(behind_gate[i])[0][idx]
            self.target_gates[dones] = closest_gate
        else:
            # set target gates to 0
            self.target_gates[dones] = np.zeros(num_reset, dtype=np.int32)
            # use start_pos
            x0 = 0*np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[0]
            y0 = 0*np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[1]
            z0 = 0*np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[2]
                  
        if self.initialize_on_ground:
            x0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[0]
            y0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[1]
            z0 = np.zeros(num_reset)
            
            vx0, vy0, vz0 = np.zeros((3,num_reset))
            phi0, theta0 = np.zeros((2,num_reset))
            psi0 = np.random.uniform(-np.pi/4,np.pi/4, size=(num_reset,)) + self.gate_yaw[self.target_gates[dones]%self.num_gates]
            p0, q0, r0 = np.zeros((3,num_reset))
            w10, w20, w30, w40 = -np.ones((4,num_reset))
        else:
            vx0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
            vy0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
            vz0 = np.random.uniform(-0.5,0.5, size=(num_reset,))
            
            phi0   = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
            theta0 = np.random.uniform(-np.pi/9,np.pi/9, size=(num_reset,))
            psi0   = np.random.uniform(-np.pi,np.pi, size=(num_reset,))
            
            p0 = np.random.uniform(-0.1,0.1, size=(num_reset,))
            q0 = np.random.uniform(-0.1,0.1, size=(num_reset,))
            r0 = np.random.uniform(-0.1,0.1, size=(num_reset,))
            
            w10 = np.random.uniform(-1,1, size=(num_reset,))
            w20 = np.random.uniform(-1,1, size=(num_reset,))
            w30 = np.random.uniform(-1,1, size=(num_reset,))
            w40 = np.random.uniform(-1,1, size=(num_reset,))

        # convert euler angles to quaternion
        quat0 = np.array(get_quat_from_euler(np.array([phi0, theta0, psi0])))
        qw0, qx0, qy0, qz0 = quat0

        self.world_states[dones] = np.stack([x0, y0, z0, vx0, vy0, vz0, qw0, qx0, qy0, qz0, p0, q0, r0, w10, w20, w30, w40], axis=1)

        self.step_counts[dones] = np.zeros(num_reset)
        
        # update params (domain randomization)
        self.params[dones] = self.randomization(num_reset)
        
        # update param encoding (used for parameter input)
        if self.param_input:
            params_with_noise = self.params[dones]*np.random.uniform(1-self.param_input_noise, 1+self.param_input_noise, size=self.params[dones].shape)
            self.param_encoding[dones] = param_encoding(params_with_noise.T).T
        
        # update states
        self.update_states()
        return np.nan_to_num(self.states)
    
    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=np.bool_))

    def step_async(self, actions):
        self.prev_actions = self.actions
        self.actions = actions

    def step_wait(self):
        dstate = np.nan_to_num(np.array(jax.device_get(f_func(self.world_states.T, self.actions.T, self.params.T).block_until_ready()), dtype=np.float32)).T
        new_states = self.world_states + self.dt * dstate

        # normalize quaternion
        new_states[:,6:10] /= np.linalg.norm(new_states[:,6:10], axis=1)[:,np.newaxis]

        # set accelerometer and gyro
        accelerometer_new = np.array(jax.device_get(get_accelerometer(self.world_states.T, self.params.T).block_until_ready()), dtype=np.float32).T
        self.gyro = new_states[:,10:13]
        
        self.step_counts += 1

        pos_old = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]
        pos_gate = self.gate_pos[self.target_gates%self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates%self.num_gates]

        # Rewards
        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)

        # progress reward (clipped by speed limit)
        prog_rewards = (d2g_old - d2g_new)
        if self.speed_limit is not None:
            prog_rewards[prog_rewards > self.speed_limit*self.dt] = self.speed_limit*self.dt
        rewards = self.progress_reward * prog_rewards

        # angular rate penalty
        rewards -= self.angular_rate_penalty * np.linalg.norm(new_states[:,10:13], axis=1)

        # low action penalty
        scaled_actions = (self.actions + 1) / 2
        scaled_prev_actions = (self.prev_actions + 1) / 2
        rewards -= self.low_action_penalty * np.sum(np.clip(0.5 - scaled_actions, 0, None), axis=1)

        # motor penalty (penalize action diffs above a threshold)
        action_diff = np.abs(scaled_actions - scaled_prev_actions)
        threshold_excess = np.clip(action_diff - self.motor_penalty_threshold, 0, None)
        rewards -= self.motor_penalty * np.sum(threshold_excess, axis=1)

        # perception penalty: keep the next gate in view
        look_gate_idx = self.target_gates.copy()
        look_gate_idx %= self.num_gates
        for gate_idx in self.fake_gates:
            look_gate_idx[look_gate_idx == gate_idx] += 1
        pos_look_gate = self.gate_pos[look_gate_idx]
        perc_angle = np.array(jax.device_get(self.perc_angle_func(new_states.T, pos_look_gate.T).block_until_ready()), dtype=np.float32)
        rewards[perc_angle > np.pi/3] -= self.perception_penalty * perc_angle[perc_angle > np.pi/3]

        # Gate passing/collision
        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)], dtype=np.float32).T
        # dot product of normal and position vector over axis 1
        pos_old_projected = (pos_old[:,0]-pos_gate[:,0])*normal[:,0] + (pos_old[:,1]-pos_gate[:,1])*normal[:,1]
        pos_new_projected = (pos_new[:,0]-pos_gate[:,0])*normal[:,0] + (pos_new[:,1]-pos_gate[:,1])*normal[:,1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        gate_passed = passed_gate_plane & (np.max(np.abs(pos_new - pos_gate), axis=1)<self.gate_size[self.target_gates]/2)
        gate_collision = passed_gate_plane & (np.max(np.abs(pos_new - pos_gate), axis=1)>self.gate_size[self.target_gates]/2)

        # collision boxes every gate has inner size 1.5 and outer size 2.7
        for idx, (pos_gate_i, yaw_gate_i) in enumerate(zip(self.gate_pos, self.gate_yaw)):
            # Rotation matrix from world frame to gate frame
            R = np.array([
                [np.cos(yaw_gate_i), np.sin(yaw_gate_i)],
                [-np.sin(yaw_gate_i), np.cos(yaw_gate_i)]
            ], dtype=np.float32)
            # position in gate frame
            pos_new_G = (pos_new[:,0:2] - pos_gate_i[0:2]) @ R.T
            # gate thickness (half)
            d = self.gate_thickness / 2
            # extra gate collision boxes
            extra_gate_collision = ((pos_new_G[:,0] > -d) & (pos_new_G[:,0] < d)) & (
                ((np.abs(pos_new_G[:,1]) > self.gate_size[self.target_gates]/2) | (np.abs(pos_new[:,2] - pos_gate[:,2]) > self.gate_size[self.target_gates]/2)) &
                ((np.abs(pos_new_G[:,1]) < 2.7/2) & (np.abs(pos_new[:,2] - pos_gate[:,2]) < 2.7/2))
            )
            # except fake gates
            for idx in self.fake_gates:
                extra_gate_collision &= (self.target_gates%self.num_gates != idx)
            gate_collision |= extra_gate_collision
        
        # Gate reward + dist penalty
        # (only for real gates)
        gate_passed_rew = gate_passed.copy()
        for gate_idx in self.fake_gates:
            gate_passed_rew &= (self.target_gates%self.num_gates != gate_idx)
        rewards[gate_passed_rew] = self.gate_reward - self.gate_offset_penalty*d2g_new[gate_passed_rew]/(self.gate_size[self.target_gates[gate_passed_rew]]/2)
        
        if (self.pos_jumps_prob > 0) and (self.pos_jumps_std > 0):
            # add position jump to the drones that passed the gate
            apply_jump = np.random.uniform(0,1,size=(self.num_envs))<self.pos_jumps_prob
            # dont apply jump when within 1m from the target gate
            apply_jump &= (d2g_new > 1)
        
            for gate_idx in self.fake_gates:
                apply_jump &= (self.target_gates%self.num_gates != gate_idx)
            if self.pos_jumps_std > 0:
                # gaussian noise
                pos_jump = np.random.normal(0,self.pos_jumps_std,size=(np.sum(apply_jump),3))
                new_states[apply_jump,0:3] = new_states[apply_jump,0:3] + pos_jump
        # add velocity offset to the drones that passed the gate
        if self.vel_offset_std > 0:
            self.vel_offset[apply_jump] = np.random.normal(0,self.vel_offset_std,size=(np.sum(apply_jump),3))
        
        # Gate collision penalty
        rewards[gate_collision] = -self.crash_penalty

        # Ground collision penalty (z > 0) & (v>10)
        ground_collision = (new_states[:,2] > -self.ground_height) & (np.linalg.norm(new_states[:,3:6], axis=1) > self.v_ground)
        rewards[ground_collision] = -self.crash_penalty
        
        # Check out of bounds
        # out_of_bounds = np.any(np.abs(new_states[:,0:2]) > 20, axis=1)          # edges of the grid
        
        out_of_bounds = np.zeros(self.num_envs, dtype=np.bool_)
        # Check if the drone is out of bounds
        if self.bounds_xy is not None:
            out_of_bounds  = (new_states[:,0] < self.bounds_xy[0,0])
            out_of_bounds |= (new_states[:,0] > self.bounds_xy[0,1])
            out_of_bounds |= (new_states[:,1] < self.bounds_xy[1,0])
            out_of_bounds |= (new_states[:,1] > self.bounds_xy[1,1])
        # z is in [-10,0]
        out_of_bounds |= new_states[:,2] < -10                                  # max height (z-axis point down)
        out_of_bounds |= np.any(np.abs(new_states[:,10:13]) > (1700*np.pi/180), axis=1)     # prevent numerical issues
        rewards[out_of_bounds] = -self.crash_penalty
        
        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        # Update target gate
        self.target_gates[gate_passed] += 1
        if self.loop_gates:
            self.target_gates %= self.num_gates
        else:
            # Check if final gate has been passed
            self.final_gate_passed = self.target_gates >= self.num_gates

        # Check if the episode is done
        dones = max_steps_reached | ground_collision | gate_collision | out_of_bounds
        if not self.loop_gates:
            dones |= self.final_gate_passed
        if self.disable_collision:
            dones = max_steps_reached | ground_collision
        elif self.disable_gate_collision:
            dones = max_steps_reached | ground_collision | out_of_bounds
        self.dones = dones

        self.update_states()
        # update accelerometer
        self.accelerometer = accelerometer_new
        states_to_return = self.states.copy()
        # Pause if collision
        if self.pause:
            dones = dones & ~dones
            self.dones = dones
        elif self.pause_if_collision:
            update = ~dones
            self.world_states[update] = new_states[update]
            self.update_states()
        else:
            self.world_states = new_states
            self.reset_(dones)


        # Write info dicts
        infos = [{}] * self.num_envs
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = states_to_return[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            # extra info for debugging
            infos[i]["ground_collision"] = ground_collision[i]
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["gate_collision"] = gate_collision[i]
            infos[i]["gate_passed"] = gate_passed[i]
        return np.nan_to_num(states_to_return), np.nan_to_num(rewards), np.nan_to_num(dones), np.nan_to_num(infos)
    
    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_attr(self, attr_name, indices=None):
        raise AttributeError()

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]*self.num_envs

    def render(self, mode='human'):
        # Outputs a dict containing all information for rendering
        state_dict = dict(zip(['x','y','z','vx','vy','vz','qw','qz','qy','qz','p','q','r','w1','w2','w3','w4'], self.world_states.T))
        # Rescale actions to [0,1] for rendering
        action_dict = dict(zip(['u1','u2','u3','u4'], (np.array(self.actions.T)+1)/2), dtype=np.float32)
        # Time
        t = self.step_counts*self.dt
        # Add euler angles to the state dict
        state_dict['phi'], state_dict['theta'], state_dict['psi'] = np.nan_to_num(np.array(jax.device_get(get_euler(self.world_states.T).block_until_ready()), dtype=np.float32))
        return {'dt': self.dt, 't':t, **state_dict, **action_dict}
