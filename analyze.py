# FUNCTIONS FOR LOADING AND ANALYZING FLIGHT DATA
import csv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from quadcopter_animation import animation

# rotatation matrix to convert from body to world frame
def Rmat(phi, theta, psi):
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],[np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

# quaternion functions https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
def quadMult(ql, qr):
    qli, qlx, qly, qlz = ql
    qri, qrx, qry, qrz = qr
    res = np.array([
        qli*qri - qlx*qrx - qly*qry - qlz*qrz,
        qlx*qri + qli*qrx + qly*qrz - qlz*qry,
        qli*qry - qlx*qrz + qly*qri + qlz*qrx,
        qli*qrz + qlx*qry - qly*qrx + qlz*qri
    ])
    return res

def quadRotate(q, v):
    qi, qx, qy, qz = q
    vx, vy, vz = v
    qv = np.array([0, vx, vy, vz])
    q_qv = quadMult(q, qv)
    qv = quadMult(q_qv, np.array([qi, -qx, -qy, -qz]))
    return qv[1:]

def quat_of_axang(ax, ang):
    ang2 = ang/2
    cang2 = np.cos(ang2)
    sang2 = np.sin(ang2)
    q = np.array([
        cang2,
        ax[0]*sang2,
        ax[1]*sang2,
        ax[2]*sang2
    ])
    return q

# testing new att setpoint calculation
def att_thrust_sp_from_acc(quat_NED, acc_sp_NED):
    # we need to rotate the drone such that the z-up axis is aligned with accSpNed-g
    # let v1 = accSpNed-g, v2 = z-up (both in NED frame)
    # then we need to rotate v2 to v1
    # the rotation axis is the cross product of v1 and v2
    # the rotation angle is the angle between v1 and v2
    
    v1 = acc_sp_NED - np.array([0, 0, 9.81])
    v2_body = np.array([0, 0, -1])
    # get v2 in NED frame (rotate v2_body by quat_NED)
    v2 = quadRotate(quat_NED, v2_body)
    # rotation axis
    axis = np.cross(v2, v1)
    axis = axis/np.linalg.norm(axis)
    angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    # convert axis and angle to quaternion
    attErrNed = quat_of_axang(axis, angle)
    # attitude setpoint
    att_sp_NED = quadMult(attErrNed, quat_NED)
    
    # thrust setpoint (acc_sp projected onto z-up axis)
    T_sp = -np.dot(v1, v2)
    
    # thrust setpoint 'direct'
    T_sp = -np.linalg.norm(v1)
    
    # EXTRA: get rate sp
    # we get the axis in body frame:
    quat_NED_inv = np.array([quat_NED[0], -quat_NED[1], -quat_NED[2], -quat_NED[3]])
    axis_body = quadRotate(quat_NED_inv, axis)
    
    # and scale the axis by 'angle' to get the att_error_axis
    att_error_axis = axis_body*angle
    
    # multiply by a gain to get the rate setpoint
    rate_sp = np.array([10, 10, 5]) * att_error_axis    
    
    return att_sp_NED, T_sp, rate_sp


def quat_to_euler(q):
    qw, qx, qy, qz = q
    phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    theta = np.arcsin(2*(qw*qy - qz*qx))
    psi = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return phi, theta, psi
    

def load_flight_data(file_name, new_format=True):
    print('Loading', file_name)
    with open(file_name) as file:
        reader = csv.reader(file)
        data = list(reader)
        
        # separate header and data
        header = [d for d in data if len(d)==2]
        header = {d[0]: d[1] for d in header}
        
        data = [d for d in data if len(d)>2]
        keys = data[0]
        data = data[1:]
        data = np.array([[float(d) if d else np.nan for d in row] for row in data]) # set empty strings to nan
        data = dict(zip(keys,data.T))
        
        # renaming keys
        data['t'] = data.pop('time')*1e-6 # us to s
        data['t'] -= data['t'][0] # start at 0
        
        # print logging frequency
        print('Logging frequency:', 1/np.mean(np.diff(data['t'])))

        # STATE ESTIMATE
        if 'pos[0]' in data:
            data['x'] = data.pop('pos[0]')*1e-3 # mm to m
            data['y'] = data.pop('pos[1]')*1e-3 # mm to m
            data['z'] = data.pop('pos[2]')*1e-3 # mm to m
        if 'vel[0]' in data:
            data['vx'] = data.pop('vel[0]')*1e-2 # cm/s to m/s
            data['vy'] = data.pop('vel[1]')*1e-2 # cm/s to m/s
            data['vz'] = data.pop('vel[2]')*1e-2 # cm/s to m/s
        
        quat_scaling = ((127 << 6) - 1) # float from -1 to +1 to VB that takes up 2 bytes max
        data['qw'] = data.pop('quat[0]')/quat_scaling
        data['qx'] = data.pop('quat[1]')/quat_scaling
        data['qy'] = data.pop('quat[2]')/quat_scaling
        data['qz'] = data.pop('quat[3]')/quat_scaling
        
        # quat to eulers
        data['phi'] = np.arctan2(2*(data['qw']*data['qx'] + data['qy']*data['qz']), 1 - 2*(data['qx']**2 + data['qy']**2))
        data['theta'] = np.arcsin(2*(data['qw']*data['qy'] - data['qz']*data['qx']))
        data['psi'] = np.arctan2(2*(data['qw']*data['qz'] + data['qx']*data['qy']), 1 - 2*(data['qy']**2 + data['qz']**2))
        
        # SETPOINTS
        if 'posSp[0]' in data:
            data['x_sp'] = data.pop('posSp[0]')*1e-3 # mm to m
            data['y_sp'] = data.pop('posSp[1]')*1e-3
            data['z_sp'] = data.pop('posSp[2]')*1e-3 
            
            data['vx_sp'] = data.pop('velSp[0]')*1e-2 # cm/s to m/s
            data['vy_sp'] = data.pop('velSp[1]')*1e-2
            data['vz_sp'] = data.pop('velSp[2]')*1e-2
            
            data['awx_sp'] = data.pop('accSp[0]')*1e-2 # cm/s^2 to m/s^2
            data['awy_sp'] = data.pop('accSp[1]')*1e-2
            data['awz_sp'] = data.pop('accSp[2]')*1e-2
            
            data['qw_sp'] = data.pop('quatSp[0]')/quat_scaling
            data['qx_sp'] = data.pop('quatSp[1]')/quat_scaling
            data['qy_sp'] = data.pop('quatSp[2]')/quat_scaling
            data['qz_sp'] = data.pop('quatSp[3]')/quat_scaling
        
            # quat to eulers
            data['phi_sp'] = np.arctan2(2*(data['qw_sp']*data['qx_sp'] + data['qy_sp']*data['qz_sp']), 1 - 2*(data['qx_sp']**2 + data['qy_sp']**2))
            data['theta_sp'] = np.arcsin(2*(data['qw_sp']*data['qy_sp'] - data['qz_sp']*data['qx_sp']))
            data['psi_sp'] = np.arctan2(2*(data['qw_sp']*data['qz_sp'] + data['qx_sp']*data['qy_sp']), 1 - 2*(data['qy_sp']**2 + data['qz_sp']**2))
        
        # euler_sp = np.array([
        #     quat_to_euler([qw, qx, qy, qz]) for qw, qx, qy, qz in zip(data['qw_sp'], data['qx_sp'], data['qy_sp'], data['qz_sp'])
        # ])
        # data['phi_sp2'] = euler_sp[:, 0]
        # data['theta_sp2'] = euler_sp[:, 1]
        # data['psi_sp2'] = euler_sp[:, 2] 
        
        data['p_sp'] = data.pop('gyroSp[0]')*np.pi/180 # deg/s to rad/s
        data['q_sp'] = data.pop('gyroSp[1]')*np.pi/180
        data['r_sp'] = data.pop('gyroSp[2]')*np.pi/180
        
        # data['dp_sp'] = data.pop('alphaSp[0]')*np.pi/180 # deg/s^2 to rad/s^2
        # data['dq_sp'] = data.pop('alphaSp[1]')*np.pi/180
        # data['dr_sp'] = data.pop('alphaSp[2]')*np.pi/180
        
        data['spf_sp_x'] = data.pop('spfSp[0]')/100
        data['spf_sp_y'] = data.pop('spfSp[1]')/100
        data['spf_sp_z'] = data.pop('spfSp[2]')/100
        
        data['T_sp'] = data['spf_sp_z']
        
        # INDI
        data['alpha[0]'] = data.pop('alpha[0]')*10*np.pi/180
        data['alpha[1]'] = data.pop('alpha[1]')*10*np.pi/180
        data['alpha[2]'] = data.pop('alpha[2]')*10*np.pi/180
        
        data['omega[0]'] = data.pop('omega[0]') # already in rad/s
        data['omega[1]'] = data.pop('omega[1]')
        data['omega[2]'] = data.pop('omega[2]')
        data['omega[3]'] = data.pop('omega[3]')
        
        data['omega_dot[0]'] = data.pop('omega_dot[0]')*100
        data['omega_dot[1]'] = data.pop('omega_dot[1]')*100
        data['omega_dot[2]'] = data.pop('omega_dot[2]')*100
        data['omega_dot[3]'] = data.pop('omega_dot[3]')*100
        
        data['u[0]'] = data['u[0]']/quat_scaling
        data['u[1]'] = data['u[1]']/quat_scaling
        data['u[2]'] = data['u[2]']/quat_scaling
        data['u[3]'] = data['u[2]']/quat_scaling
        
        
        # MOTOR CMDS
        motor_limits = header['motorOutput'].split(',')
        umin = int(motor_limits[0])
        umax = int(motor_limits[1])
        data['motor_min'] = umin
        data['motor_max'] = umax
        data['u1'] = (data['motor[0]'] - umin)/(umax - umin)
        data['u2'] = (data['motor[1]'] - umin)/(umax - umin)
        data['u3'] = (data['motor[2]'] - umin)/(umax - umin)
        data['u4'] = (data['motor[3]'] - umin)/(umax - umin)
        
        # OPTITRACK
        if 'extPos[0]' in data:
            data['x_opti'] = data.pop('extPos[0]')*1e-3 # mm to m
            data['y_opti'] = data.pop('extPos[1]')*1e-3 # mm to m
            data['z_opti'] = data.pop('extPos[2]')*1e-3 # mm to m
            
            data['vx_opti'] = data.pop('extVel[0]')*1e-2 # cm/s to m/s
            data['vy_opti'] = data.pop('extVel[1]')*1e-2 # cm/s to m/s
            data['vz_opti'] = data.pop('extVel[2]')*1e-2 # cm/s to m/s
            
            data['phi_opti'] = data.pop('extAtt[0]')/1000 # mrad to rad
            data['theta_opti'] = data.pop('extAtt[1]')/1000 # mrad to rad
            data['psi_opti'] = data.pop('extAtt[2]')/1000 # mrad to rad
            
            data['vel_test'] = np.sqrt(data['vx_opti']**2 + data['vy_opti']**2 + data['vz_opti']**2)
        
            # OPTITRACK INTERPOLATED
            updates = (np.diff(data['x_opti']) != 0) | (np.diff(data['y_opti']) != 0) | (np.diff(data['z_opti']) != 0) | (np.diff(data['phi_opti']) != 0) | (np.diff(data['theta_opti']) != 0) | (np.diff(data['psi_opti']) != 0)
            updates = np.where(updates)[0]+1
            
            # data['t_opti_rec'] = data['t'][updates]
            # data['t_opti_sent'] = data['extTime'][updates]/1000 # ms to s
            # data['x_opti_int'] = np.interp(data['t'], data['t'][updates], data['x_opti'][updates])
            # data['y_opti_int'] = np.interp(data['t'], data['t'][updates], data['y_opti'][updates])
            # data['z_opti_int'] = np.interp(data['t'], data['t'][updates], data['z_opti'][updates])
            # data['phi_opti_int'] = np.interp(data['t'], data['t'][updates], data['phi_opti'][updates])
            # data['theta_opti_int'] = np.interp(data['t'], data['t'][updates], data['theta_opti'][updates])
            # data['psi_opti_int'] = np.interp(data['t'], data['t'][updates], data['psi_opti'][updates])
        
        
        # IMU
        gyro_scale = np.pi/180 # deg/s to rad/s
        if bool(float(header['blackbox_high_resolution'])):
            gyro_scale /= 10
        data['p'] = data.pop('gyroADC[0]')*gyro_scale       # (from FLU to FRD)
        data['q'] =-data.pop('gyroADC[1]')*gyro_scale
        data['r'] =-data.pop('gyroADC[2]')*gyro_scale
        if new_format:
            data['q']=-data['q']
            data['r']=-data['r']
        
        acc_scale = 9.81/float(header['acc_1G'])
        data['ax'] = data.pop('accSmooth[0]')*acc_scale     # (from FLU to FRD)
        data['ay'] =-data.pop('accSmooth[1]')*acc_scale
        data['az'] =-data.pop('accSmooth[2]')*acc_scale
        if new_format:
            data['ay']=-data['ay']
            data['az']=-data['az']
        
        if 'accUnfiltered[0]' in data.keys():
            data['ax_unfiltered'] = data.pop('accUnfiltered[0]')*acc_scale     # (from FLU to FRD)
            data['ay_unfiltered'] =-data.pop('accUnfiltered[1]')*acc_scale
            data['az_unfiltered'] =-data.pop('accUnfiltered[2]')*acc_scale
            if new_format:
                data['ay_unfiltered']=-data['ay_unfiltered']
                data['az_unfiltered']=-data['az_unfiltered']
        elif 'accADCafterRpm[0]' in data.keys():
            data['ax_unfiltered'] = data.pop('accADCafterRpm[0]')*acc_scale
            data['ay_unfiltered'] =-data.pop('accADCafterRpm[1]')*acc_scale
            data['az_unfiltered'] =-data.pop('accADCafterRpm[2]')*acc_scale
            if new_format:
                data['ay_unfiltered']=-data['ay_unfiltered']
                data['az_unfiltered']=-data['az_unfiltered']
        
        # filter acc
        cutoff = 8 # Hz
        sos = sp.signal.butter(2, cutoff, 'low', fs=1/np.mean(np.diff(data['t'])), output='sos')
        data['ax_filt'] = sp.signal.sosfiltfilt(sos, data['ax'])
        data['ay_filt'] = sp.signal.sosfiltfilt(sos, data['ay'])
        data['az_filt'] = sp.signal.sosfiltfilt(sos, data['az'])
        
        # EKF
        if 'ekf_pos[0]' in data:
            data['ekf_x'] = data.pop('ekf_pos[0]')*1e-3 # mm to m
            data['ekf_y'] = data.pop('ekf_pos[1]')*1e-3 # mm to m
            data['ekf_z'] = data.pop('ekf_pos[2]')*1e-3 # mm to m
            data['ekf_vx'] = data.pop('ekf_vel[0]')*1e-2 # cm/s to m/s
            data['ekf_vy'] = data.pop('ekf_vel[1]')*1e-2 # cm/s to m/s
            data['ekf_vz'] = data.pop('ekf_vel[2]')*1e-2 # cm/s to m/s
            data['ekf_phi'] = data.pop('ekf_att[0]')/1000 # mrad to rad
            data['ekf_theta'] = data.pop('ekf_att[1]')/1000 # mrad to rad
            data['ekf_psi'] = data.pop('ekf_att[2]')/1000 # mrad to rad
            data['ekf_acc_b_x'] = data.pop('ekf_acc_b[0]')/1000 # mm/s^2 to m/s^2
            data['ekf_acc_b_y'] = data.pop('ekf_acc_b[1]')/1000 # mm/s^2 to m/s^2
            data['ekf_acc_b_z'] = data.pop('ekf_acc_b[2]')/1000 # mm/s^2 to m/s^2
            data['ekf_gyro_b_x'] = data.pop('ekf_gyro_b[0]')*np.pi/180
            data['ekf_gyro_b_y'] = data.pop('ekf_gyro_b[1]')*np.pi/180
            data['ekf_gyro_b_z'] = data.pop('ekf_gyro_b[2]')*np.pi/180
        
            data['ax_filt_unbiased'] = data['ax_filt'] - data['ekf_acc_b_x']
            data['ay_filt_unbiased'] = data['ay_filt'] - data['ekf_acc_b_y']
            data['az_filt_unbiased'] = data['az_filt'] - data['ekf_acc_b_z']
    
        # VIO
        if 'vioPos[0]' in data and False:
            print('vioooo')
            print(np.any(data['vioPos[0]'] != 0))
            data['x_vio'] = data.pop('vioPos[0]')*1e-3 # mm to m
            data['y_vio'] = data.pop('vioPos[1]')*1e-3
            data['z_vio'] = data.pop('vioPos[2]')*1e-3
            data['vx_vio'] = data.pop('vioVel[0]')*1e-2 # cm/s to m/s
            data['vy_vio'] = data.pop('vioVel[1]')*1e-2
            data['vz_vio'] = data.pop('vioVel[2]')*1e-2
            data['qw_vio'] = data.pop('vioQuat[0]')/quat_scaling
            data['qx_vio'] = data.pop('vioQuat[1]')/quat_scaling
            data['qy_vio'] = data.pop('vioQuat[2]')/quat_scaling
            data['qz_vio'] = data.pop('vioQuat[3]')/quat_scaling
            data['phi_vio'] = np.arctan2(2*(data['qw_vio']*data['qx_vio'] + data['qy_vio']*data['qz_vio']), 1 - 2*(data['qx_vio']**2 + data['qy_vio']**2))
            data['theta_vio'] = np.arcsin(2*(data['qw_vio']*data['qy_vio'] - data['qz_vio']*data['qx_vio']))
            data['psi_vio'] = np.arctan2(2*(data['qw_vio']*data['qz_vio'] + data['qx_vio']*data['qy_vio']), 1 - 2*(data['qy_vio']**2 + data['qz_vio']**2))
            data['p_vio'] = data.pop('vioRate[0]')*np.pi/180 # deg/s to rad/s
            data['q_vio'] = data.pop('vioRate[1]')*np.pi/180
            data['r_vio'] = data.pop('vioRate[2]')*np.pi/180

        # UGLY HACK: overwrite states with ekf states
        print('WARNING: overwriting states with ekf states')
        data['x'] = data['ekf_x']
        data['y'] = data['ekf_y']
        data['z'] = data['ekf_z']
        data['vx'] = data['ekf_vx']
        data['vy'] = data['ekf_vy']
        data['vz'] = data['ekf_vz']
        data['phi'] = data['ekf_phi']
        data['theta'] = data['ekf_theta']
        data['psi'] = data['ekf_psi']
        
        
        # EXTRA
        data['v'] = np.sqrt(data['vx']**2 + data['vy']**2 + data['vz']**2)

        # body velocities
        v_body = np.stack([
            Rmat(phi, theta, psi).T@[vx, vy, vz]
            for vx, vy, vz, phi, theta, psi
            in zip(data['vx'],data['vy'],data['vz'],data['phi'],data['theta'],data['psi'])
        ])
        data['vbx'] = v_body[:,0]
        data['vby'] = v_body[:,1]
        data['vbz'] = v_body[:,2]
        
        # simple drag model
        data['Dx'] = -0.43291866*data['vbx']
        data['Dy'] = -0.49557959*data['vby']
        
        # world accelerations
        a_world = np.stack([
            Rmat(phi, theta, psi)@np.array([ax, ay, az]) + np.array([0, 0, 9.81])
            for ax, ay, az, phi, theta, psi
            in zip(data['ax'],data['ay'],data['az'],data['phi'],data['theta'],data['psi'])
        ])
        data['awx'] = a_world[:,0]
        data['awy'] = a_world[:,1]
        data['awz'] = a_world[:,2]
        
        data['awx_filt'] = sp.signal.sosfiltfilt(sos, data['awx'])
        data['awy_filt'] = sp.signal.sosfiltfilt(sos, data['awy'])
        data['awz_filt'] = sp.signal.sosfiltfilt(sos, data['awz'])
        
        # # reconstruct acc setpoint by rotating the thrust setpoint with the attitude setpoint
        # a_world_sp_rec = np.stack([
        #     Rmat(phi, theta, psi)@np.array([0,0,T]) + np.array([0, 0, 9.81])
        #     for T, phi, theta, psi
        #     in zip(data['T_sp'],data['phi_sp'],data['theta_sp'],data['psi_sp'])
        # ])
        # data['awx_sp_rec'] = a_world_sp_rec[:, 0]
        # data['awy_sp_rec'] = a_world_sp_rec[:, 1]
        # data['awz_sp_rec'] = a_world_sp_rec[:, 2]
        
        # # test att, thrust setpoint calculation
        # att_sp_test = np.array([
        #     att_thrust_sp_from_acc(np.array([qw,qx,qy,qz]), np.array([awx,awy,awz]))[0] for 
        #     qw,qx,qy,qz,awx,awy,awz in zip(data['qw'],data['qx'],data['qy'],data['qz'],data['awx_sp'],data['awy_sp'],data['awz_sp'])
        # ])
        
        # data['qw_sp_test'] = att_sp_test[:, 0]
        # data['qx_sp_test'] = att_sp_test[:, 1]
        # data['qy_sp_test'] = att_sp_test[:, 2]
        # data['qz_sp_test'] = att_sp_test[:, 3]
        
        # data['phi_sp_test'] = np.arctan2(2*(data['qw_sp_test']*data['qx_sp_test'] + data['qy_sp_test']*data['qz_sp_test']), 1 - 2*(data['qx_sp_test']**2 + data['qy_sp_test']**2))
        # data['theta_sp_test'] = np.arcsin(2*(data['qw_sp_test']*data['qy_sp_test'] - data['qz_sp_test']*data['qx_sp_test']))
        # data['psi_sp_test'] = np.arctan2(2*(data['qw_sp_test']*data['qz_sp_test'] + data['qx_sp_test']*data['qy_sp_test']), 1 - 2*(data['qy_sp_test']**2 + data['qz_sp_test']**2))
        
        # # Thrust setpoint
        # thrust_sp_test = np.array([
        #     att_thrust_sp_from_acc(np.array([qw,qx,qy,qz]), np.array([awx,awy,awz]))[1] for 
        #     qw,qx,qy,qz,awx,awy,awz in zip(data['qw'],data['qx'],data['qy'],data['qz'],data['awx_sp'],data['awy_sp'],data['awz_sp'])
        # ])
        # data['T_sp_test'] = thrust_sp_test
        
        # # Rate setpoint
        # rate_sp_test = np.array([
        #     att_thrust_sp_from_acc(np.array([qw,qx,qy,qz]), np.array([awx,awy,awz]))[2] for 
        #     qw,qx,qy,qz,awx,awy,awz in zip(data['qw'],data['qx'],data['qy'],data['qz'],data['awx_sp'],data['awy_sp'],data['awz_sp'])
        # ])
        # data['p_sp_test'] = rate_sp_test[:, 0]
        # data['q_sp_test'] = rate_sp_test[:, 1]
        # data['r_sp_test'] = rate_sp_test[:, 2]
        
        # # reconstruct acc setpoint from test att, thrust setpoint
        # a_world_sp_rec_test = np.stack([
        #     Rmat(phi, theta, psi)@np.array([0,0,T]) + np.array([0, 0, 9.81])
        #     for T, phi, theta, psi
        #     in zip(data['T_sp_test'],data['phi_sp_test'],data['theta_sp_test'],data['psi_sp_test'])
        # ])
        # data['awx_sp_rec_test'] = a_world_sp_rec_test[:, 0]
        # data['awy_sp_rec_test'] = a_world_sp_rec_test[:, 1]
        # data['awz_sp_rec_test'] = a_world_sp_rec_test[:, 2]
        
        return data

# FUNCTIONS FOR ANIMATIONS  
from quadcopter_animation import animation
import importlib
importlib.reload(animation)

# RACE TRACK:
r=3.
num = 8
gate_pos = np.array([
    [r*np.cos(angle), r*np.sin(angle), -1.5] for angle in np.linspace(0,2*np.pi,num,endpoint=False)
])
gate_yaw = np.array([np.pi/2 + i*2*np.pi/num for i in range(num)])

def animate_data(data):
    animation.animate(
        data['t'],
        data['x'], data['y'], data['z'],
        data['phi'], data['theta'], data['psi'],
        np.stack([data['u1'], data['u2'], data['u3'], data['u4']], axis=1),
        target=np.stack([data['x_sp'], data['y_sp'], data['z_sp']], axis=1)
    )
    
def animate_data_double(data1, data2):
    animation.animate(
        [data1['t'], data2['t']],
        [data1['x'], data2['x']],
        [data1['y'], data2['y']],
        [data1['z'], data2['z']],
        [data1['phi'], data2['phi']],
        [data1['theta'], data2['theta']],
        [data1['psi'], data2['psi']],
        [
            np.stack([data1['u1'], data1['u2'], data1['u3'], data1['u4']], axis=1),
            np.stack([data2['u1'], data2['u2'], data2['u3'], data2['u4']], axis=1)
        ],
        multiple_trajectories=True,
        simultaneous=True,
        colors=[(255,0,0), (0,255,0)]
    )
    
def animate_data_multiple(*data_list, **kwargs):
    if 'colors' in kwargs:
        colors_ = kwargs.pop('colors')
        colors = lambda i: colors_[i]
    else:
        # color map
        import matplotlib.cm as cm
        colors_ = cm.get_cmap('jet', len(data_list))
        # to int
        colors = lambda i: tuple(int(c*255) for c in colors_(i)[:-1])
    
    animation.animate(
        [d['t'] for d in data_list],
        [d['x'] for d in data_list],
        [d['y'] for d in data_list],
        [d['z'] for d in data_list],
        [d['phi'] for d in data_list],
        [d['theta'] for d in data_list],
        [d['psi'] for d in data_list],
        [np.stack([d['u1'], d['u2'], d['u3'], d['u4']], axis=1) for d in data_list],
        # target=[np.stack([d['x_opti'], d['y_opti'], d['z_opti']], axis=1) for d in data_list],
        multiple_trajectories=True,
        simultaneous=True,
        colors=[colors(i) for i in range(len(data_list))],
        **kwargs
    )
    
def animate_data_multiple2(*data_list, **kwargs):
    # color map
    import matplotlib.cm as cm
    colors_ = cm.get_cmap('jet', len(data_list))
    # to int
    colors = lambda i: tuple(int(c*255) for c in colors_(i)[:-1])
    colors2 = lambda i: tuple(int(c*55+200) for c in colors_(i)[:-1])
    
    animation.animate(
        [d['t'] for d in data_list]*2,
        [d['x'] for d in data_list] + [d['x_opti'] for d in data_list],
        [d['y'] for d in data_list] + [d['y_opti'] for d in data_list],
        [d['z'] for d in data_list] + [d['z_opti'] for d in data_list],
        [d['phi'] for d in data_list] + [d['phi_opti'] for d in data_list],
        [d['theta'] for d in data_list] + [d['theta_opti'] for d in data_list],
        [d['psi'] for d in data_list] + [d['psi_opti'] for d in data_list],
        [np.stack([d['u1'], d['u2'], d['u3'], d['u4']], axis=1) for d in data_list]+[np.zeros((len(data_list[0]['t']), 4)) for _ in data_list],
        target=np.stack([data_list[0]['x_sp'], data_list[0]['y_sp'], data_list[0]['z_sp']], axis=1),
        multiple_trajectories=True,
        simultaneous=True,
        colors=[colors(i) for i in range(len(data_list))]+[colors2(i) for i in range(len(data_list))],
        gate_pos=gate_pos,
        gate_yaw=gate_yaw
    )
    
    
def animate_data2(data):
    animation.animate(
        [data['t'], data['t']],
        [data['x'], data['x_opti']],
        [data['y'], data['y_opti']],
        [data['z'], data['z_opti']],
        [data['phi'], data['phi_opti']],
        [data['theta'], data['theta_opti']],
        [data['psi'], data['psi_opti']],
        [
            np.stack([data['u1'], data['u2'], data['u3'], data['u4']], axis=1),
            np.stack([data['u1'], data['u2'], data['u3'], data['u4']], axis=1)
        ],
        multiple_trajectories=True,
        simultaneous=True,
        target=np.stack([data['x_sp'], data['y_sp'], data['z_sp']], axis=1),
        colors=[(255,0,0), (0,255,0)]
    )
    
# FUNCTIONS FOR SYSTEM IDENTIFICATION
def fit_thrust_drag_model(data):
    print('fitting thrust and drag model')
    fig, axs = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
    
    # THRUST MODEL ------------------------------------------------------------------------------
    # az = k_w*sum(omega_i**2)
    # we will find k_w, by linear regression
    X = np.stack([
        data['omega[0]']**2 + data['omega[1]']**2 + data['omega[2]']**2 + data['omega[3]']**2,
        # data['vbx']**2 + data['vby']**2,
        # data['vz']*(data['omega[0]']+data['omega[1]']+data['omega[2]']+data['omega[3]'])
    ])
    Y = data['az']
    k_w, = A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    
    if 'az_unfiltered' in data:
        axs[0].plot(data['t'], data['az_unfiltered'], label='az raw', alpha=0.1, color='blue')
    axs[0].plot(data['t'], Y, label='az') #, alpha=0.2)
    # axs[0].plot(data['t'], data['az_filt'], label='az filt')
    axs[0].plot(data['t'], A@X, label='T model')
    # axs[0].plot(data['t'], A_nom@X, label='T model nominal')
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel('acc [m/s^2]')
    axs[0].legend()
    axs[0].set_title('Thrust model: \n az = k_w*sum(omega_i**2) \n k_w = {:.2e}'.format(k_w))
    # axs[0].set_title('Thrust model: \n az = k_w*sum(omega_i**2) + k_h*(vbx**2+vby**2) + k_z*vbz*sum(omega_i) \n k_w, k_h, k_z = {:.2e}, {:.2e}, {:.2e}'.format(k_w, k_h, k_z))
    
    # DRAG MODEL X ------------------------------------------------------------------------------
    # Eq. 2 from https://doi.org/10.1016/j.robot.2023.104588
    # ax = -k_x*vbx*sum(omega_i)
    # we will find k_x by linear regression
    X = np.stack([data['vbx']*(data['omega[0]']+data['omega[1]']+data['omega[2]']+data['omega[3]'])])
    # X = np.stack([data['vbx']])
    Y = data['ax']
    k_x, = A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    
    if 'ax_unfiltered' in data:
        axs[1].plot(data['t'], data['ax_unfiltered'], label='ax raw', alpha=0.1, color='blue')
    axs[1].plot(data['t'], Y, label='ax') #, alpha=0.2)
    # axs[1].plot(data['t'], data['ax_filt'], label='ax filt')
    axs[1].plot(data['t'], A@X, label='Dx model')
    # axs[1].plot(data['t'], A_nom@X, label='Dx model nominal')
    axs[1].set_xlabel('t [s]')
    axs[1].set_ylabel('acc [m/s^2]')
    axs[1].legend()
    axs[1].set_title('Drag model X: \n ax = k_x*vbx*sum(omega_i) \n k_x = {:.2e}'.format(k_x))
    
    # DRAG MODEL Y ------------------------------------------------------------------------------
    # Eq. 2 from https://doi.org/10.1016/j.robot.2023.104588
    # ay = -k_y*vby*sum(omega_i)
    # we will find k_y by linear regression
    X = np.stack([data['vby']*(data['omega[0]']+data['omega[1]']+data['omega[2]']+data['omega[3]'])])
    # X = np.stack([data['vby']])
    Y = data['ay']
    k_y, = A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    
    if 'ay_unfiltered' in data:
        axs[2].plot(data['t'], data['ay_unfiltered'], label='ay raw', alpha=0.1, color='blue')
    axs[2].plot(data['t'], Y, label='ay') #, alpha=0.2)
    # axs[2].plot(data['t'], data['ay_filt'], label='ay filt')
    axs[2].plot(data['t'], A@X, label='Dy model')
    # axs[2].plot(data['t'], A_nom@X, label='Dy model nominal')
    axs[2].set_xlabel('t [s]')
    axs[2].set_ylabel('acc [m/s^2]')
    axs[2].legend()
    axs[2].set_title('Drag model Y: \n ay = k_y*vby*sum(omega_i) \n k_y = {:.2e}'.format(k_y))
    
    # show fig with the window name 'Thrust and Drag Model'
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Thrust and Drag Model')
    plt.show()
    
    # print('k_w = {:.2e}, k_x = {:.2e}, k_y = {:.2e}'.format(k_w, k_x, k_y))
    return k_w, k_x, k_y

from scipy.optimize import minimize

def fit_actuator_model(data):
    # the steadystate rpm motor response to the motor command u is described by:
    # w_c = (w_max-w_min)*sqrt(k u**2 + (1-k)*u) + w_min
    # the dynamics of the motor is described by:
    # dw/dt = (w_c - w)/tau
    # dw/dt = ((w_max-w_min)*sqrt(k u**2 + (1-k)*u) + w_min - w)*tau_inv
    # we will find w_min, w_max, k, tau_inv by nonlinear optimization
    
    def get_w_est(params, u, w):
        w_min, w_max, k, tau_inv = params
        w_c = (w_max-w_min)*np.sqrt(k*u**2 + (1-k)*u) + w_min
        # progate the dynamics
        w_est = np.zeros_like(u)
        w_est[0] = w[0]
        for i in range(1, len(w_est)):
            dt = data['t'][i] - data['t'][i-1]
            w_est[i] = w_est[i-1] + (w_c[i] - w_est[i-1])*dt*tau_inv
        return w_est

    def get_error(params, u, w):
        return np.linalg.norm(get_w_est(params, u, w) - w)
    
    # w_min, w_max, k, tau_inv
    initial_guess = [285, 2700, 0.75, 100]
    bounds = [(0, 1000), (0, 6000), (0, 1), (1, 1000.)]
    
    # minimize for each motor
    err_1 = lambda x: get_error(x, data['u1'], data['omega[0]'])
    err_2 = lambda x: get_error(x, data['u2'], data['omega[1]'])
    err_3 = lambda x: get_error(x, data['u3'], data['omega[2]'])
    err_4 = lambda x: get_error(x, data['u4'], data['omega[3]'])
    err_tot = lambda x: err_1(x) + err_2(x) + err_3(x) + err_4(x)
    
    print('fitting actuator model...')
    res_1 = minimize(err_1, initial_guess, bounds=bounds)
    res_2 = minimize(err_2, initial_guess, bounds=bounds)
    res_3 = minimize(err_3, initial_guess, bounds=bounds)
    res_4 = minimize(err_4, initial_guess, bounds=bounds)
    res_tot = minimize(err_tot, initial_guess, bounds=bounds)
    
    # set k to 45
    # res_tot.x[2] = 0.45
    
    # plot results
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    
    axs[0,0].plot(data['t'], data['omega[0]'], label='w')
    axs[0,0].plot(data['t'], get_w_est(res_1.x, data['u1'], data['omega[0]']), label='w est')
    axs[0,0].plot(data['t'], get_w_est(res_tot.x, data['u1'], data['omega[0]']), label='w est tot')
    # axs[0,0].plot(data['t'], get_w_est(res_nom, data['u1'], data['omega[0]']), label='w est nom')
    axs[0,0].set_xlabel('t [s]')
    axs[0,0].set_ylabel('w [rad/s]')
    axs[0,0].legend()
    params = res_1.x
    params[3] = 1/params[3]
    axs[0,0].set_title('Motor 1: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}'.format(*params))
    
    axs[0,1].plot(data['t'], data['omega[1]'], label='w')
    axs[0,1].plot(data['t'], get_w_est(res_2.x, data['u2'], data['omega[1]']), label='w_est')
    axs[0,1].plot(data['t'], get_w_est(res_tot.x, data['u2'], data['omega[1]']), label='w est tot')
    # axs[0,1].plot(data['t'], get_w_est(res_nom, data['u2'], data['omega[1]']), label='w est nom')
    axs[0,1].set_xlabel('t [s]')
    axs[0,1].set_ylabel('w [rad/s]')
    axs[0,1].legend()
    params = res_2.x
    params[3] = 1/params[3]
    axs[0,1].set_title('Motor 2: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}'.format(*params))
    
    axs[1,0].plot(data['t'], data['omega[2]'], label='w')
    axs[1,0].plot(data['t'], get_w_est(res_3.x, data['u3'], data['omega[2]']), label='w_est')
    axs[1,0].plot(data['t'], get_w_est(res_tot.x, data['u3'], data['omega[2]']), label='w est tot')
    # axs[1,0].plot(data['t'], get_w_est(res_nom, data['u3'], data['omega[2]']), label='w est nom')
    axs[1,0].set_xlabel('t [s]')
    axs[1,0].set_ylabel('w [rad/s]')
    axs[1,0].legend()
    params = res_3.x
    params[3] = 1/params[3]
    axs[1,0].set_title('Motor 3: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}'.format(*params))
    
    axs[1,1].plot(data['t'], data['omega[3]'], label='w')
    axs[1,1].plot(data['t'], get_w_est(res_4.x, data['u4'], data['omega[3]']), label='w_est')
    axs[1,1].plot(data['t'], get_w_est(res_tot.x, data['u4'], data['omega[3]']), label='w est tot')
    # axs[1,1].plot(data['t'], get_w_est(res_nom, data['u4'], data['omega[3]']), label='w est nom')
    axs[1,1].set_xlabel('t [s]')
    axs[1,1].set_ylabel('w [rad/s]')
    axs[1,1].legend()
    params = res_4.x
    params[3] = 1/params[3]
    axs[1,1].set_title('Motor 4: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}'.format(*params))
    
    # suptitle
    params = res_tot.x
    params[3] = 1/params[3]
    fig.suptitle('Actuator model: \n dw/dt = dw/dt = ((w_max-w_min)*sqrt(k u**2 + (1-k)*u) + w_min - w)/tau \n Total fit: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, 1/tau = {:.2f}'.format(*params))
    
    # show fig with the window name 'Actuator Model'
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Actuator Model')
    plt.show()
    
    # print('w_min={:.2f}, w_max={:.2f}, k={:.2f}, tau={:.2f}'.format(*res_tot.x))
    return res_tot.x

def fit_moments_model(data):
    print('fitting moments model')
    # model from https://doi.org/10.1016/j.robot.2023.104588
    # d_p     = (q*r*(Iyy-Izz) + Mx)/Ixx = Jx*q*r + Mx_
    # d_q     = (p*r*(Izz-Ixx) + My)/Iyy = Jy*p*r + My_
    # d_r     = (p*q*(Ixx-Iyy) + Mz)/Izz = Jz*p*q + Mz_
    
    # where    
    # Mx_ = k_p1*omega_1**2 + k_p2*omega_2**2 + k_p3*omega_3**2 + k_p4*omega_4**2
    # My_ = k_q1*omega_1**2 + k_q2*omega_2**2 + k_q3*omega_3**2 + k_q4*omega_4**2
    # Mz_ = k_r1*omega_1 + k_r2*omega_2 + k_r3*omega_3 + k_r4*omega_4 + k_r5*d_omega_1 + k_r6*d_omega_2 + k_r7*d_omega_3 + k_r8*d_omega_4
    
    # to get the derivative we use a low pass filter
    cutoff = 64 # Hz
    sos = sp.signal.butter(2, cutoff, 'low', fs=1/np.mean(np.diff(data['t'])), output='sos')
    
    dp = sp.signal.sosfiltfilt(sos, np.gradient(data['p'])/np.gradient(data['t']))
    dq = sp.signal.sosfiltfilt(sos, np.gradient(data['q'])/np.gradient(data['t']))
    dr = sp.signal.sosfiltfilt(sos, np.gradient(data['r'])/np.gradient(data['t']))
    
    domega_1 = sp.signal.sosfiltfilt(sos, np.gradient(data['omega[0]'])/np.gradient(data['t']))
    domega_2 = sp.signal.sosfiltfilt(sos, np.gradient(data['omega[1]'])/np.gradient(data['t']))
    domega_3 = sp.signal.sosfiltfilt(sos, np.gradient(data['omega[2]'])/np.gradient(data['t']))
    domega_4 = sp.signal.sosfiltfilt(sos, np.gradient(data['omega[3]'])/np.gradient(data['t']))
    
    params = np.load('params/aggressive_cmds2.npz')
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    
    X = np.stack([
        data['omega[0]']**2,
        data['omega[1]']**2,
        data['omega[2]']**2,
        data['omega[3]']**2,
    ])
    Y = dp
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_p1, k_p2, k_p3, k_p4 = A
    dp_fit = A@X
    
    axs[0,0].plot(data['t'], Y, label='dp')
    axs[0,0].plot(data['t'], dp_fit, label='dp fit')
    axs[0,0].set_xlabel('t [s]')
    axs[0,0].set_ylabel('dp [rad/s^2]')
    axs[0,0].legend()
    axs[0,0].set_title('dp = k_p1*w1**2 + k_p2*w2**2 + k_p3*w3**2 + k_p4*w4**2 \n k_p1, k_p2, k_p3, k_p4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(*A))
    
    X = np.stack([
        data['omega[0]']**2,
        data['omega[1]']**2,
        data['omega[2]']**2,
        data['omega[3]']**2,
    ])
    Y = dq
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_q1, k_q2, k_q3, k_q4 = A
    dq_fit = A@X
    
    axs[1,0].plot(data['t'], Y, label='dq')
    axs[1,0].plot(data['t'], dq_fit, label='dq fit')
    axs[1,0].set_xlabel('t [s]')
    axs[1,0].set_ylabel('dq [rad/s^2]')
    axs[1,0].legend()
    axs[1,0].set_title('dq = k_q1*w1**2 + k_q2*w2**2 + k_q3*w3**2 + k_q4*w4**2 \n k_q1, k_q2, k_q3, k_q4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(*A))
    
    
    # X = np.stack([
    #     data['omega[0]'],
    #     data['omega[1]'],
    #     data['omega[2]'],
    #     data['omega[3]'],
    #     domega_1,
    #     domega_2,
    #     domega_3,
    #     domega_4,
    # ])
    # Y = dr
    # A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    # k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8 = A
    # dr_fit = A@X
    
    X = np.stack([
        -data['omega[0]']+data['omega[1]']+data['omega[2]']-data['omega[3]'],
        -domega_1+domega_2+domega_3-domega_4,
    ])
    Y = dr
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_r, k_rd = A
    dr_fit = A@X

     
    axs[2,0].plot(data['t'], Y, label='dr')
    # axs[2,0].plot(data['t'], dr_fit, label='dr fit')
    axs[2,0].plot(data['t'], dr_fit, label='dr fit')
    axs[2,0].set_xlabel('t [s]')
    axs[2,0].set_ylabel('dr [rad/s^2]')
    axs[2,0].legend()
    # title = 'dr = k_r1*w1 + k_r2*w2 + k_r3*w3 + k_r4*w4 + k_r5*dw1 + k_r6*dw2 + k_r7*dw3 + k_r8*dw4 \n k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8 = {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(*A)
    title = 'dr = k_r*(w1-w2+w3-w4) + k_rd*(dw1-dw2+dw3-dw4) \n k_r, k_rd = {:.2e}, {:.2e}'.format(*A)
    axs[2,0].set_title(title)
    
    # 3 plots with p,q,r
    axs[0,1].plot(data['t'], dp_fit-dp, label='dp fit error')
    axs[0,1].set_xlabel('t [s]')
    axs[0,1].set_ylabel('p [rad/s]')
    axs[0,1].legend()
    
    axs[1,1].plot(data['t'], dq_fit-dq, label='dq fit error')
    axs[1,1].set_xlabel('t [s]')
    axs[1,1].set_ylabel('q [rad/s]')
    axs[1,1].legend()
    
    axs[2,1].plot(data['t'], dr_fit-dr, label='dr fit error')
    axs[2,1].set_xlabel('t [s]')
    axs[2,1].set_ylabel('r [rad/s]')
    axs[2,1].legend()
    
    # show fig with the window name 'Moments Model'
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Moments Model')
    plt.show()
    
    # print the results
    # print('k_p1, k_p2, k_p3, k_p4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(k_p1, k_p2, k_p3, k_p4))
    # print('k_q1, k_q2, k_q3, k_q4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(k_q1, k_q2, k_q3, k_q4))
    # print('k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8 = {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8))
    return k_p1, k_p2, k_p3, k_p4, k_q1, k_q2, k_q3, k_q4, k_r, k_rd


# FUNCTIONS FOR PLOTTING
def ekf_plot(data):
    ekf_updates = (np.diff(data['x_opti']) != 0) | (np.diff(data['y_opti']) != 0) | (np.diff(data['z_opti']) != 0) | (np.diff(data['phi_opti']) != 0) | (np.diff(data['theta_opti']) != 0) | (np.diff(data['psi_opti']) != 0)
    ekf_updates = np.where(ekf_updates)[0]+1

    # subplots 1x3 with ekf_x, ekf_y, ekf_z
    fig, axs = plt.subplots(5, 3, figsize=(5,5), sharex=True, sharey='row', tight_layout=True)

    # POSITION
    plt.sca(axs[0,0])
    plt.plot(data['t'], data['ekf_x'], label='ekf')
    plt.plot(data['t'][ekf_updates], data['x_opti'][ekf_updates], label='opti')
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.legend()
    plt.sca(axs[0,1])
    plt.plot(data['t'], data['ekf_y'], label='ekf')
    plt.plot(data['t'][ekf_updates], data['y_opti'][ekf_updates], label='opti')
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.sca(axs[0,2])
    plt.plot(data['t'], data['ekf_z'], label='ekf')
    plt.plot(data['t'][ekf_updates], data['z_opti'][ekf_updates], label='opti')
    plt.xlabel('t [s]')
    plt.ylabel('z [m]')
    plt.legend()

    # VELOCITY
    plt.sca(axs[1,0])
    plt.plot(data['t'], data['ekf_vx'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('vx [m/s]')
    plt.legend()
    plt.sca(axs[1,1])
    plt.plot(data['t'], data['ekf_vy'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('vy [m/s]')
    plt.legend()
    plt.sca(axs[1,2])
    plt.plot(data['t'], data['ekf_vz'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('vz [m/s]')
    plt.legend()

    # ATTITUDE
    plt.sca(axs[2,0])
    plt.plot(data['t'], data['ekf_phi'], label='ekf')
    plt.plot(data['t'][ekf_updates], data['phi_opti'][ekf_updates], label='opti')
    plt.xlabel('t [s]')
    plt.ylabel('phi [rad]')
    plt.legend()
    plt.sca(axs[2,1])
    plt.plot(data['t'], data['ekf_theta'], label='ekf')
    plt.plot(data['t'][ekf_updates], data['theta_opti'][ekf_updates], label='opti')
    plt.xlabel('t [s]')
    plt.ylabel('theta [rad]')
    plt.legend()
    plt.sca(axs[2,2])
    ekf_psi = ((data['ekf_psi']+np.pi)%(2*np.pi))-np.pi
    plt.plot(data['t'], ekf_psi, label='ekf')
    plt.plot(data['t'][ekf_updates], data['psi_opti'][ekf_updates], label='opti')
    plt.xlabel('t [s]')
    plt.ylabel('psi [rad]')
    plt.legend()

    # ACC BIAS
    plt.sca(axs[3,0])
    plt.plot(data['t'], data['ekf_acc_b_x'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('acc_b_x [m/s^2]')
    plt.legend()
    plt.sca(axs[3,1])
    plt.plot(data['t'], data['ekf_acc_b_y'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('acc_b_y [m/s^2]')
    plt.legend()
    plt.sca(axs[3,2])
    plt.plot(data['t'], data['ekf_acc_b_z'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('acc_b_z [m/s^2]')
    plt.legend()

    # GYRO BIAS
    plt.sca(axs[4,0])
    plt.plot(data['t'], data['ekf_gyro_b_x'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('gyro_b_x [rad/s]')
    plt.legend()
    plt.sca(axs[4,1])
    plt.plot(data['t'], data['ekf_gyro_b_y'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('gyro_b_y [rad/s]')
    plt.legend()
    plt.sca(axs[4,2])
    plt.plot(data['t'], data['ekf_gyro_b_z'], label='ekf')
    plt.xlabel('t [s]')
    plt.ylabel('gyro_b_z [rad/s]')
    plt.legend()

    plt.show()
    
def acc_plot(data):
    # plot imu measurements
    fig, axs = plt.subplots(1, 3, figsize=(10,5), sharex=True, sharey='col', tight_layout=True)

    # ACCELEROMETER
    # X
    plt.sca(axs[0])
    if 'ax_unfiltered' in data:
        plt.plot(data['t'], data['ax_unfiltered'], label='ax raw', alpha=0.5, color='blue')
    plt.plot(data['t'], data['ax'], label='ax')
    # plt.plot(data['t'], data['ax_filt'], label='ax_filt')
    plt.ylim([-160,160])
    plt.xlabel('t [s]')
    plt.ylabel('ax [m/s^2]')
    plt.legend()
    # Y
    plt.sca(axs[1])
    if 'ay_unfiltered' in data:
        plt.plot(data['t'], data['ay_unfiltered'], label='ay raw', alpha=0.5, color='blue')
    plt.plot(data['t'], data['ay'], label='ay')
    # plt.plot(data['t'], data['ay_filt'], label='ay_filt')
    plt.ylim([-160,160])
    plt.xlabel('t [s]')
    plt.ylabel('ay [m/s^2]')
    plt.legend()
    # Z
    plt.sca(axs[2])
    if 'az_unfiltered' in data:
        plt.plot(data['t'], data['az_unfiltered'], label='az raw', alpha=0.5, color='blue')
    plt.plot(data['t'], data['az'], label='az')
    # plt.plot(data['t'], data['az_filt'], label='az_filt')
    plt.ylim([-160,160])
    plt.xlabel('t [s]')
    plt.ylabel('az [m/s^2]')
    plt.legend()

    plt.show()
    
def imu_plot(data, **kwargs):
    # plot imu measurements
    fig, axs = plt.subplots(2, 3, figsize=(10,5), sharex=True, sharey='row', tight_layout=True)

    # ACCELEROMETER
    # X
    plt.sca(axs[0,0])
    if 'ax_unfiltered' in data:
        plt.plot(data['t'], data['ax_unfiltered'], label='ax raw', alpha=0.5, color='blue')
    plt.plot(data['t'], data['ax'], label='ax')
    # plt.plot(data['t'], data['ax_filt'], label='ax_filt')
    plt.ylim([-160,160])
    plt.xlabel('t [s]')
    plt.ylabel('ax [m/s^2]')
    plt.legend()
    # Y
    plt.sca(axs[0,1])
    if 'ay_unfiltered' in data:
        plt.plot(data['t'], data['ay_unfiltered'], label='ay raw', alpha=0.5, color='blue')
    plt.plot(data['t'], data['ay'], label='ay')
    # plt.plot(data['t'], data['ay_filt'], label='ay_filt')
    plt.ylim([-160,160])
    plt.xlabel('t [s]')
    plt.ylabel('ay [m/s^2]')
    plt.legend()
    # Z
    plt.sca(axs[0,2])
    if 'az_unfiltered' in data:
        plt.plot(data['t'], data['az_unfiltered'], label='az raw', alpha=0.5, color='blue')
    plt.plot(data['t'], data['az'], label='az')
    if 'k_w' in kwargs:
        plt.plot(data['t'], -kwargs['k_w']*(data['omega[0]']**2+data['omega[1]']**2+data['omega[2]']**2+data['omega[3]']**2), label='thrust model')
    # plt.plot(data['t'], data['az_filt'], label='az_filt')
    plt.ylim([-160,160])
    plt.xlabel('t [s]')
    plt.ylabel('az [m/s^2]')
    plt.legend()
    
    # GYROSCOPE
    # X
    plt.sca(axs[1,0])
    plt.plot(data['t'], data['p'], label='p')
    plt.xlabel('t [s]')
    plt.ylabel('p [rad/s]')
    plt.legend()
    # Y
    plt.sca(axs[1,1])
    plt.plot(data['t'], data['q'], label='q')
    plt.xlabel('t [s]')
    plt.ylabel('q [rad/s]')
    plt.legend()
    # Z
    plt.sca(axs[1,2])
    plt.plot(data['t'], data['r'], label='r')
    plt.xlabel('t [s]')
    plt.ylabel('r [rad/s]')
    plt.legend()

    plt.show()
    
    
def actuator_plot(data):
    # 4x2 subplots with u, omega
    fig, axs = plt.subplots(4, 2, figsize=(5,5), sharex=True, sharey='col', tight_layout=True)
    
    # MOTOR COMMANDS
    # 1
    plt.sca(axs[0,0])
    plt.plot(data['t'], data['u1'], label='u1')
    plt.xlabel('t [s]')
    plt.ylabel('u1')
    plt.legend()
    # 2
    plt.sca(axs[1,0])
    plt.plot(data['t'], data['u2'], label='u2')
    plt.xlabel('t [s]')
    plt.ylabel('u2')
    plt.legend()
    # 3
    plt.sca(axs[2,0])
    plt.plot(data['t'], data['u3'], label='u3')
    plt.xlabel('t [s]')
    plt.ylabel('u3')
    plt.legend()
    # 4
    plt.sca(axs[3,0])
    plt.plot(data['t'], data['u4'], label='u4')
    plt.xlabel('t [s]')
    plt.ylabel('u4')
    plt.legend()
    
    # MOTOR SPEED
    # 1
    plt.sca(axs[0,1])
    plt.plot(data['t'], data['omega[0]'], label='omega1')
    plt.xlabel('t [s]')
    plt.ylabel('omega1 [rad/s]')
    plt.legend()
    # 2
    plt.sca(axs[1,1])
    plt.plot(data['t'], data['omega[1]'], label='omega2')
    plt.xlabel('t [s]')
    plt.ylabel('omega2 [rad/s]')
    plt.legend()
    # 3
    plt.sca(axs[2,1])
    plt.plot(data['t'], data['omega[2]'], label='omega3')
    plt.xlabel('t [s]')
    plt.ylabel('omega3 [rad/s]')
    plt.legend()
    # 4
    plt.sca(axs[3,1])
    plt.plot(data['t'], data['omega[3]'], label='omega4')
    plt.xlabel('t [s]')
    plt.ylabel('omega4 [rad/s]')
    plt.legend()
    
    plt.show()
    
    
def xy_plot(data, **kwargs):
    # figure with a xy plot of 10x10m that shows the drone trajectory + gates (if provided)
    plt.figure(figsize=(10,10))
    # if 'x' in data:
    #     plt.plot(data['x'], data['y'], label='est')
    # if 'ekf_x' in data:
    #     plt.plot(data['ekf_x'], data['ekf_y'], label='ekf')
    if 'x_opti' in data:
        plt.plot(data['x_opti'], data['y_opti'], label='opti')
    # gate gate_pos and gate_yaw form the kwargs
    if 'gate_pos' in kwargs and 'gate_yaw' in kwargs:
        for i in range(len(kwargs['gate_pos'])):
            x, y, z = kwargs['gate_pos'][i]
            yaw = kwargs['gate_yaw'][i]
            plt.plot([x-np.sin(yaw)*0.75, x+np.sin(yaw)*0.75], [y-np.cos(yaw)*0.75, y+np.cos(yaw)*0.75], color='red')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.show()