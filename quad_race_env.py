import torch
import stable_baselines3
import sys
import numpy as np
from sympy import *

print("python version:", sys.version)
print("stable_baselines3 version:", stable_baselines3.__version__)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# set torch default device
torch.set_default_device(device)

# Equations of motion 3D quadcopter from https://arxiv.org/pdf/2304.13460.pdf
state = symbols('x y z v_x v_y v_z phi theta psi p q r w1 w2 w3 w4')
x,y,z,vx,vy,vz,phi,theta,psi,p,q,r,w1,w2,w3,w4 = state
control = symbols('U_1 U_2 U_3 U_4')    # normalized motor commands between [-1,1]
u1,u2,u3,u4 = control

g = 9.81
params = symbols('k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max')
k_x, k_y, k_w, k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, tau, k, w_min, w_max = params

# Rotation matrix 
Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
R = Rz*Ry*Rx

# Body velocity
vbx, vby, vbz = R.T@Matrix([vx,vy,vz])

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
T = -k_w*(W1**2 + W2**2 + W3**2 + W4**2)
Dx = -k_x*vbx*(W1+W2+W3+W4)
Dy = -k_y*vby*(W1+W2+W3+W4)

# Moments
Mx = -k_p1*W1**2 - k_p2*W2**2 + k_p3*W3**2 + k_p4*W4**2
My = -k_q1*W1**2 + k_q2*W2**2 - k_q3*W3**2 + k_q4*W4**2
Mz = -k_r1*W1 + k_r2*W2 + k_r3*W3 - k_r4*W4 - k_r5*d_W1 + k_r6*d_W1 + k_r7*d_W1 - k_r8*d_W1

# Dynamics
d_x = vx
d_y = vy
d_z = vz

d_vx, d_vy, d_vz = Matrix([0,0,g]) + R@Matrix([Dx, Dy,T])

d_phi   = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
d_theta = q*cos(phi) - r*sin(phi)
d_psi   = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)

d_p     = Mx
d_q     = My
d_r     = Mz

# State space model
f = [d_x, d_y, d_z, d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_p, d_q, d_r, d_w1, d_w2, d_w3, d_w4]

# lambdify
f_func = lambdify((Array(state), Array(control), Array(params)), Array(f), 'numpy')

# Efficient vectorized version of the environment
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

class Quadcopter3DGates(VecEnv):
    def __init__(self,
                 num_envs,
                 gates_pos,
                 gate_yaw,
                 start_pos,
                 randomization,
                 gates_ahead=1,
                 pause_if_collision=False,
                 motor_limit=1.0,
                 initialize_at_random_gates=True
                 ):
        
        # Define the race track
        self.start_pos = start_pos.astype(np.float32)
        self.gate_pos = gates_pos.astype(np.float32)
        self.gate_yaw = gate_yaw.astype(np.float32)
        self.num_gates = gates_pos.shape[0]
        self.gates_ahead = gates_ahead
        
        # Pause if collision
        self.pause_if_collision = pause_if_collision
        
        # Domain randomization
        self.randomization = lambda n: np.array([randomization(n)[p.name] for p in params]).T
        self.params = self.randomization(num_envs)
        
        # Motor limit
        self.motor_limit = motor_limit
        
        # Initialize at random gates
        self.initialize_at_random_gates = initialize_at_random_gates

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
            ])
            self.gate_pos_rel[i,0:2] = R@self.gate_pos_rel[i,0:2]
            self.gate_yaw_rel[i] = self.gate_yaw[i] - self.gate_yaw[i-1]
            # wrap yaw
            self.gate_yaw_rel[i] %= 2*np.pi
            if self.gate_yaw_rel[i] > np.pi:
                self.gate_yaw_rel[i] -= 2*np.pi
            elif self.gate_yaw_rel[i] < -np.pi:
                self.gate_yaw_rel[i] += 2*np.pi

        # Define the target gate for each environment
        self.target_gates = np.zeros(num_envs, dtype=int)

        # action space: [cmd1, cmd2, cmd3, cmd4]
        # U = (u+1)/2 --> u = 2U-1
        u_lim = 2*self.motor_limit-1
        action_space = spaces.Box(low=-1, high=u_lim, shape=(4,))

        # observation space: pos[G], vel[G], att[eulerB->G], rates[B], rpms, future_gates[G], future_gate_dirs[G]
        # [G] = reference frame aligned with target gate
        # [B] = body frame
        self.state_len = 16+4*self.gates_ahead
        self.input_hist = 1
        observation_space = spaces.Box(
            low  = np.array([-np.inf]*self.state_len*self.input_hist),
            high = np.array([ np.inf]*self.state_len*self.input_hist)
        )

        # Initialize the VecEnv
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # world state: pos[W], vel[W], att[eulerB->W], rates[B], rpms
        self.world_states = np.zeros((num_envs,16), dtype=np.float32)
        # observation state
        self.states = np.zeros((num_envs,self.state_len*self.input_hist), dtype=np.float32)
        # state history tracking
        num_hist = 10
        self.state_hist = np.zeros((num_envs,num_hist,self.state_len), dtype=np.float32)

        # Define any other environment-specific parameters
        self.max_steps = 1200      # Maximum number of steps in an episode
        self.dt = np.float32(0.01) # Time step duration

        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs,4), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs,4), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.final_gate_passed = np.zeros(num_envs, dtype=bool)

        self.update_states = self.update_states_gate
        
        self.pause = False

    def update_states_gate(self):
        # Transform pos and vel in gate frame
        gate_pos = self.gate_pos[self.target_gates%self.num_gates]
        gate_yaw = self.gate_yaw[self.target_gates%self.num_gates]

        # Rotation matrix from world frame to gate frame
        R = np.array([
            [np.cos(gate_yaw), np.sin(gate_yaw)],
            [-np.sin(gate_yaw), np.cos(gate_yaw)]
        ]).transpose((2,1,0))

        # new state array to prevent the weird bug related to indexing ([:] syntax)
        new_states = np.zeros((self.num_envs,self.state_len), dtype=np.float32)

        # Update positions
        pos_W = self.world_states[:,0:3]
        pos_G = (pos_W[:,np.newaxis,0:2] - gate_pos[:,np.newaxis,0:2]) @ R
        new_states[:,0:2] = pos_G[:,0,:]
        new_states[:,2] = pos_W[:,2] - gate_pos[:,2]

        # Update velocities
        vel_W = self.world_states[:,3:6]
        vel_G = (vel_W[:,np.newaxis,0:2]) @ R
        new_states[:,3:5] = vel_G[:,0,:]
        new_states[:,5] = vel_W[:,2]

        # Update attitude
        new_states[:,6:8] = self.world_states[:,6:8]
        yaw = self.world_states[:,8] - gate_yaw
        yaw %= 2*np.pi
        yaw[yaw > np.pi] -= 2*np.pi
        yaw[yaw < -np.pi] += 2*np.pi
        new_states[:,8] = yaw

        # Update rates
        new_states[:,9:12] = self.world_states[:,9:12]

        # Update rpms
        new_states[:,12:16] = self.world_states[:,12:16]

        # Update future gates relative to current gate ([0,0,0,0] for out of bounds)
        for i in range(self.gates_ahead):
            indices = (self.target_gates+i+1)
            # loop when out of bounds
            indices = indices % self.num_gates
            valid = indices < self.num_gates
            new_states[valid,16+4*i:16+4*i+3] = self.gate_pos_rel[indices[valid]]
            new_states[valid,16+4*i+3] = self.gate_yaw_rel[indices[valid]]

        # update history
        self.state_hist = np.roll(self.state_hist, 1, axis=1)
        self.state_hist[:,0] = new_states
        
        # print('new_states \n', new_states)
        
        # stack history up to self.input_hist
        self.states = self.state_hist[:,0:self.input_hist].reshape((self.num_envs,-1))
        # print('states \n', self.states)

    def reset_(self, dones):
        num_reset = dones.sum()
        
        if self.initialize_at_random_gates:
            # set target gates to random gates
            self.target_gates[dones] = np.random.randint(0,self.num_gates, size=num_reset)
            # set position to 1m in front of the target gate
            # gate_pos + [cos(gate_yaw), sin(gate_yaw), 0]
            pos = self.gate_pos[self.target_gates[dones]%self.num_gates]
            yaw = self.gate_yaw[self.target_gates[dones]%self.num_gates]
            
            pos = pos - np.array([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)]).T
            x0, y0, z0 = pos.T
            
            # add 0.5m of noise
            # x0 += np.random.uniform(-0.5,0.5, size=(num_reset,))
            # y0 += np.random.uniform(-0.5,0.5, size=(num_reset,))
            # z0 += np.random.uniform(-0.5,0.5, size=(num_reset,))
        else:
            # use start_pos
            x0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[0]
            y0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[1]
            z0 = np.random.uniform(-0.5,0.5, size=(num_reset,)) + self.start_pos[2]
        
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

        self.world_states[dones] = np.stack([x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, p0, q0, r0, w10, w20, w30, w40], axis=1)

        self.step_counts[dones] = np.zeros(num_reset)
        
        # self.target_gates[dones] = np.zeros(num_reset, dtype=int)
        
        # update params (domain randomization)
        self.params[dones] = self.randomization(num_reset)
        
        # update states
        self.update_states()
        return self.states
    
    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.prev_actions = self.actions
        self.actions = actions
    
    def step_wait(self):
        new_states = self.world_states + self.dt*f_func(self.world_states.T, self.actions.T, self.params.T).T
        
        self.step_counts += 1

        pos_old = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]
        pos_gate = self.gate_pos[self.target_gates%self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates%self.num_gates]

        # Rewards
        d2g_old = np.linalg.norm(pos_old - pos_gate, axis=1)
        d2g_new = np.linalg.norm(pos_new - pos_gate, axis=1)
        rat_penalty = 0.001*np.linalg.norm(new_states[:,9:12], axis=1)
        angle_penalty = 0.0*np.linalg.norm(new_states[:,6:8], axis=1)
        action_penalty = 0.0*np.linalg.norm((self.actions+1)/2, axis=1)
        action_penalty_delta = 0.001*np.linalg.norm((self.actions-self.prev_actions), axis=1)

        prog_rewards = d2g_old - d2g_new
        # max_speed = 12.0
        # cap progress rewards to be less than max_speed*dt
        # prog_rewards[prog_rewards > max_speed*self.dt] = max_speed*self.dt
        
        # rewards = (d2g_old - d2g_new) - rat_penalty - angle_penalty
        rewards = prog_rewards - rat_penalty #- action_penalty - action_penalty_delta
        
        # Gate passing/collision
        normal = np.array([np.cos(yaw_gate), np.sin(yaw_gate)]).T
        # dot product of normal and position vector over axis 1
        pos_old_projected = (pos_old[:,0]-pos_gate[:,0])*normal[:,0] + (pos_old[:,1]-pos_gate[:,1])*normal[:,1]
        pos_new_projected = (pos_new[:,0]-pos_gate[:,0])*normal[:,0] + (pos_new[:,1]-pos_gate[:,1])*normal[:,1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)
        gate_size = 1.5
        gate_passed = passed_gate_plane & np.all(np.abs(pos_new - pos_gate)<gate_size/2, axis=1)
        gate_collision = passed_gate_plane & np.any(np.abs(pos_new - pos_gate)>gate_size/2, axis=1)
        
        # Gate reward + dist penalty
        # rewards[gate_passed] = 1 #10 - 10*d2g_new[gate_passed]
        
        # Gate collision penalty
        # rewards[gate_collision] = -10

        # Ground collision penalty (z > 0)
        ground_collision = new_states[:,2] > 0
        rewards[ground_collision] = -10
        
        # Check out of bounds
        out_of_bounds = np.any(np.abs(new_states[:,0:2]) > 5, axis=1)          # edges of the grid
        out_of_bounds |= new_states[:,2] < -7                                  # max height (z-axis point down)
        out_of_bounds |= np.any(np.abs(new_states[:,9:12]) > 1000, axis=1)     # prevent numerical issues
        rewards[out_of_bounds] = -10
        
        # Check number of steps
        max_steps_reached = self.step_counts >= self.max_steps

        # Update target gate
        self.target_gates[gate_passed] += 1
        self.target_gates[gate_passed] %= self.num_gates
        
        # Check if final gate has been passed
        # self.final_gate_passed = self.target_gates >= self.num_gates

        # give reward for passing final gate
        rewards[self.final_gate_passed] = 10
        
        # Check if the episode is done
        dones = max_steps_reached | ground_collision | out_of_bounds | gate_collision  #| self.final_gate_passed
        self.dones = dones
        
        # Pause if collision
        if self.pause:
            dones = dones & ~dones
            self.dones = dones
        elif self.pause_if_collision:
            # dones = max_steps_reached | final_gate_passed | out_of_bounds
            update = ~dones #~(gate_collision | ground_collision)
            # Update world states
            self.world_states[update] = new_states[update]
            self.update_states()
            # Reset env if done (and update states)
            # self.reset_(dones)
        else:
            # Update world states
            self.world_states = new_states
            # reset env if done (and update states)
            self.reset_(dones)


        # Write info dicts
        infos = [{}] * self.num_envs
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
        return self.states, rewards, dones, infos
    
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
        state_dict = dict(zip(['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r','w1','w2','w3','w4'], self.world_states.T))
        # Rescale actions to [0,1] for rendering
        action_dict = dict(zip(['u1','u2','u3','u4'], (np.array(self.actions.T)+1)/2))
        return {**state_dict, **action_dict}