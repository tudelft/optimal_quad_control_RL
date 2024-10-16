# PLAN:
# 1) Define gym environtment (race track, randomization)
# 2) load NN controller
# 3) Get EKF functions from ekf_calc.c, ekf_calc.h
# 4) Inintialize ENV + EKF
# 5) Run EKF /visualize..

# 1) Define gym environtment (race track and randomization)
print("Loading env...")
from quad_race_env import *
from randomization import *

# long oval race track in TII hall
gate_pos = np.array([
    [1.5, -5., -1.5],
    [1.5,  5., -1.5],
    [0.0, 6.5, -1.5],
    [-1.5, 5., -1.5],
    [-1.5, -5., -1.5],
    [0.0, -6.5, -1.5]
])
gate_yaw = np.array([0, 0, 0.5, 1, 1, 1.5])*np.pi+np.pi/2
start_pos = gate_pos[0] + np.array([0,-2,0])
start_pos[2] = 0
bounds_xy = np.array([[-3, 3], [-8, 8]])

env = Quadcopter3DGates(
    num_envs=2,
    start_pos=start_pos,
    gates_pos=gate_pos,
    gate_yaw=gate_yaw,
    bounds_xy=bounds_xy,
    randomization=randomization_dummy_30_percent,
    initialize_at_random_gates=False,
    initialize_on_ground=True,
    pause_if_collision=True,
    cam_angle=45*np.pi/180,
    gate_size=1.5,
    disable_collision=True
)
env.dt = 1/500
env.max_steps = 20*500

# 2) compile NN controller
import os
import subprocess
import ctypes
print("Compiling NN controller...")
path_to_c_code = '/home/robinferede/Git/optimal_quad_control_RL/c_code_nn'
# Create object files
subprocess.call('gcc -fPIC -c *.c', shell=True, cwd=path_to_c_code)
# Create library
subprocess.call('gcc -shared -Wl,-soname,libtools.so -o libtools.so *.o', shell=True, cwd=path_to_c_code)
# Remove object files
subprocess.call('rm *.o', shell=True, cwd=path_to_c_code)

lib_path = os.path.abspath(path_to_c_code+"/libtools.so")
lib_nn = ctypes.CDLL(lib_path)

# define argument types 
lib_nn.nn_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib_nn.nn_control.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

def nn_control(x):
    x = np.array(x, dtype=np.float32)

    # scale to [w_min, w_max]
    x[12:16] = (x[12:16] + 1)/2*(3000.)

    c_net_input = (ctypes.c_float*len(x))(*x)
    c_net_output = (ctypes.c_float*4)()
    
    
    lib_nn.nn_control(c_net_input, c_net_output)
    out = np.array(c_net_output[:])
    # map back to [-1,1]
    out = (out*2) - 1
    return out

# ekf is in quaternions, sim is in euler angles so we need a conversion function:
def quat2euler(q):
    # normalize quaternion
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    theta = np.arcsin(2*(q0*q2 - q3*q1))
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    return np.array([phi, theta, psi])

def euler2quat(e):
    phi, theta, psi = e
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    q0 = cy * cr * cp + sy * sr * sp
    q1 = cy * sr * cp - sy * cr * sp
    q2 = cy * cr * sp + sy * sr * cp
    q3 = sy * cr * cp - cy * sr * sp
    return np.array([q0, q1, q2, q3])

# 3) Get EKF functions from ekf_calc.c, ekf_calc.h #TODO: change absolute path
print("Compiling EKF...")
path_to_c_code = '/home/robinferede/Git/optimal_quad_control_RL/c_code_ekf'

# https://cu7ious.medium.com/how-to-use-dynamic-libraries-in-c-46a0f9b98270
path = os.path.abspath(path_to_c_code)
# Remove old library
subprocess.call('rm -f *.so', shell=True, cwd=path)
subprocess.call('rm -f *.o', shell=True, cwd=path)
# Compile object files
subprocess.call('gcc -fPIC -c *.c common/*.c -Icommon', shell=True, cwd=path)
# Create library
subprocess.call('gcc -shared -Wl,-soname,libtools.so -o libtools.so *.o', shell=True, cwd=path)

lib_path = os.path.abspath(path_to_c_code+"/libtools.so")
lib = ctypes.CDLL(lib_path)

# define argument types 
lib.ekf_set_Q.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.ekf_set_R_mocap.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.ekf_set_R_vbody.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.ekf_set_X.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.ekf_set_P_diag.argtypes = [ctypes.POINTER(ctypes.c_float)]

lib.ekf_get_X.restype = ctypes.POINTER(ctypes.c_float)
lib.ekf_get_P.restype = ctypes.POINTER(ctypes.c_float)

lib.ekf_predict.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float]
lib.ekf_update_mocap.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.ekf_update_vbody.argtypes = [ctypes.POINTER(ctypes.c_float)]

ekf_use_quat = ctypes.c_bool.in_dll(lib, "ekf_use_quat")
ekf_use_quat.value = True

def predict(IMU, dt):
    lib.ekf_predict((ctypes.c_float*len(IMU))(*IMU), ctypes.c_float(dt))

def update_mocap(MOCAP):
    lib.ekf_update_mocap((ctypes.c_float*len(MOCAP))(*MOCAP))
    
    
# 4) Inintialize ENV + EKF
def INIT():
    # ENV
    env.reset()

    # EKF
    # we use ground truth initial pos and att for the ENV
    x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,w1,w2,w3,w4 = env.world_states[0]
    pos = np.array([x,y,z])
    eulers = np.array([phi,theta,psi])
    quat = euler2quat(eulers)

    # initial state (x,y,z,vx,vy,vz,qw,qx,qy,qz,lx,ly,lz,lp,lq,lr)
    X_init = [*pos, 0, 0, 0, *quat, 0, 0, 0, 0, 0, 0]
    
    # initial covariance matrix (diagonal)
    P_init = np.ones_like(X_init)

    # THESE COVARIANCES ARE THE EXACT SETTINGS WE USED IN OUR TII FLIGHTS
    # process noise covariance (ax,ay,az,p,q,r,lx,ly,lz,lp,lq,lr)
    Q_diag = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 1e-5, 1e-5, 1e-5, 5e-6, 5e-6, 5e-6])
    # measurement noise covariance (x,y,z,qw,qx,qy,qz)
    R_diag_mocap = np.array([1e-3, 1e-3, 1e-3, 1., 1., 1., 1.])

    # set the parameters in the EKF
    lib.ekf_set_Q((ctypes.c_float*len(Q_diag))(*Q_diag))
    lib.ekf_set_R_mocap((ctypes.c_float*len(R_diag_mocap))(*R_diag_mocap))
    lib.ekf_set_X((ctypes.c_float*len(X_init))(*X_init))
    lib.ekf_set_P_diag((ctypes.c_float*len(P_init))(*P_init))

# 5) Define IMU and MOCAP model from env
# for now we define the IMU and MOCAP std to match the noise covariances
std_IMU = np.array([1., 1.,1., 0.1, 0.1, 0.1]) # same as Till's HIL sim
bias_IMU = np.array([0.1, 0.1, -0.1, 0.001, 0.001, 0.001])
std_MOCAP = 0*np.array([0.1, 0.1, 0.1, 1., 1., 1., 1.])

def get_IMU(env):
    acc = env.accelerometer[0]
    gyro = env.gyro[0]
    return np.array([
        acc[0] + np.random.normal(0, std_IMU[0]),
        acc[1] + np.random.normal(0, std_IMU[1]),
        acc[2] + np.random.normal(0, std_IMU[2]),
        gyro[0] + np.random.normal(0, std_IMU[3]),
        gyro[1] + np.random.normal(0, std_IMU[4]),
        gyro[2] + np.random.normal(0, std_IMU[5])
    ])
    
def get_motor_speed(env):
    return env.world_states[0][12:16]
    
def get_MOCAP(env):
    return env.world_states[0][0:7] + np.random.normal(0, std_MOCAP)

# 6) Run EKF visualize
from quadcopter_animation import animation
print('Run ekf in sim')

def animate_policy(env, **kwargs):
    print('init')
    INIT() # resets env and ekf
    print('sim')
    def run():
        # 1) EKF STATE ESTIMATION
        imu = get_IMU(env)
        if not env.dones[0]:
            predict(imu, env.dt)
        
        # 2) NN CONTROL (calculate actions from ekf state)  
        x = lib.ekf_get_X()
        ekf_state = np.array([x[i] for i in range(16)])
        ekf_pos = ekf_state[0:3]
        ekf_vel = ekf_state[3:6]
        ekf_quat = ekf_state[6:10] # quaternion
        ekf_eulers = quat2euler(ekf_quat)
        world_state = np.array([*ekf_pos, *ekf_vel, *ekf_eulers, *imu[3:6], *get_motor_speed(env)])
        action = nn_control(world_state.copy())
        actions = np.array([action, action])
        
        steps = env.step_counts[0]+1
        states, rewards, dones, infos = env.step(actions)
        
        # 3) set env 2 state to ekf state
        env.world_states[1] = world_state
        env.update_states()
        
        if dones[0]:
            print(infos)
        
        out = env.render()
        out['color'] = [(255,0,0), (0,0,255)]
        out['names'] = ['GT', 'EKF']
        return out
    animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, reset_func=INIT, cam_angle=env.cam_angle, gate_size=env.gate_size, **kwargs)

animate_policy(env, hist_len=10000, fps=500)