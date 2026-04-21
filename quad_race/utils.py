import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from quadcopter_animation import animation


# ANIMATION

action_list = []
reward_list = []


def animate_policy(model, env, **kwargs):
    env.reset()

    def run():
        actions, _ = model.predict(env.states, deterministic=True)
        action_list.append(actions)
        states, rewards, dones, infos = env.step(actions)
        reward_list.append(rewards)

        out = env.render()

        crashed = np.sum(dones) / len(dones)
        out['crashed'] = crashed

        target = env.gate_pos[env.target_gates[0]]
        out['traj_x'] = np.array([target[0]])
        out['traj_y'] = np.array([target[1]])
        out['traj_z'] = np.array([target[2]])

        out['v'] = np.array([np.linalg.norm(env.world_states[0, 3:6])])

        return out

    animation.view(run, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, fps=1/env.dt,
                   reset_func=env.reset, cam_angle=env.cam_angle, grid_bounds=env.bounds_xy,
                   gate_size=env.gate_size, fake_gates=env.fake_gates, **kwargs)


# EPISODE STATS

def get_ep_info(model, env, plotting=True, printing=True):
    env.reset_seed()
    env.reset()
    env.pause_if_collision = True

    ep_done = np.zeros(env.num_envs, dtype=bool)
    ep_rew = np.zeros(env.num_envs)
    ep_len = np.zeros(env.num_envs, dtype=int)
    ep_vel = np.zeros(env.num_envs)
    ep_acc_z = np.zeros(env.num_envs)
    ep_jerk_z = np.zeros(env.num_envs)
    max_jerk_z = np.zeros(env.num_envs)
    max_acc_z = np.zeros(env.num_envs)
    max_vel = np.zeros(env.num_envs)
    ep_gate_dist = np.zeros(env.num_envs)
    ep_max_gate_dist = np.zeros(env.num_envs)
    num_gates_passed = np.zeros(env.num_envs, dtype=int)
    crashed = np.zeros(env.num_envs, dtype=bool)

    time_1st_gate = np.full(env.num_envs, -1.0)
    time_2_laps = np.zeros(env.num_envs)

    while not np.all(ep_done):
        actions, _ = model.predict(env.states, deterministic=True)
        target_gates = env.target_gates.copy()
        gate_pos = env.gate_pos[target_gates % env.num_gates].copy()
        accelerometer_old = env.accelerometer.copy()
        states, rewards, dones, infos = env.step(actions)

        still_active = ~ep_done
        ep_rew[still_active] += rewards[still_active]
        ep_len[still_active] += 1

        ep_vel[still_active] += np.linalg.norm(states[still_active, 3:6], axis=1)
        max_vel[still_active] = np.maximum(
            max_vel[still_active],
            np.linalg.norm(states[still_active, 3:6], axis=1)
        )

        ep_acc_z[still_active] += np.abs(accelerometer_old[still_active, 2])
        max_acc_z[still_active] = np.maximum(
            max_acc_z[still_active],
            np.abs(accelerometer_old[still_active, 2])
        )

        ep_jerk_z[still_active] += np.abs((env.accelerometer[still_active, 2] - accelerometer_old[still_active, 2]) / env.dt)
        max_jerk_z[still_active] = np.maximum(
            max_jerk_z[still_active],
            np.abs((env.accelerometer[still_active, 2] - accelerometer_old[still_active, 2]) / env.dt)
        )

        gate_passed = target_gates != env.target_gates
        num_gates_passed[still_active] += gate_passed[still_active]

        newly_first_gate = (num_gates_passed == 1) & gate_passed & still_active
        time_1st_gate[newly_first_gate] = ep_len[newly_first_gate] * env.dt

        # 2 laps is 22 gates passed
        finished_2_laps = (num_gates_passed >= 22)
        time_2_laps[finished_2_laps] = ep_len[finished_2_laps] * env.dt

        gate_dist = np.linalg.norm(gate_pos - env.world_states[:, 0:3], axis=1)
        just_passed_mask = gate_passed & still_active
        ep_gate_dist[just_passed_mask] += gate_dist[just_passed_mask]
        ep_max_gate_dist[just_passed_mask] = np.maximum(
            ep_max_gate_dist[just_passed_mask],
            gate_dist[just_passed_mask]
        )

        ep_done |= dones | finished_2_laps
        crashed |= dones & (~finished_2_laps) & (ep_len < env.max_steps)

    mean_vel_all = ep_vel / ep_len
    mean_acc_z_all = ep_acc_z / ep_len
    mean_jerk_z_all = ep_jerk_z / ep_len

    has_passed_gate = (num_gates_passed > 0)
    ep_gate_dist[has_passed_gate] /= num_gates_passed[has_passed_gate]

    completed_mask = (time_2_laps > 0)
    mean_vel = mean_vel_all[completed_mask]
    mean_acc_z = mean_acc_z_all[completed_mask]
    max_acc_z = max_acc_z[completed_mask]
    mean_jerk_z = mean_jerk_z_all[completed_mask]
    max_jerk_z = max_jerk_z[completed_mask]
    max_vel = max_vel[completed_mask]
    ep_max_gate_dist = ep_max_gate_dist[completed_mask]
    ep_gate_dist = ep_gate_dist[completed_mask]
    time_2_laps = time_2_laps[completed_mask]

    time_1st_gate_valid = time_1st_gate[completed_mask]
    valid_first_gate = (time_1st_gate_valid >= 0)
    time_race = time_2_laps[valid_first_gate] - time_1st_gate_valid[valid_first_gate]

    crashed_percentage = 100.0 * np.sum(crashed) / env.num_envs

    info = {
        'ep_rew': ep_rew,
        'ep_len': ep_len,
        'mean_vel': mean_vel,
        'max_vel': max_vel,
        'num_gates_passed': num_gates_passed,
        'mean_gate_dist': ep_gate_dist,
        'max_gate_dist': ep_max_gate_dist,
        'time_2_laps': time_2_laps,
        'time_1st_gate': time_1st_gate_valid[valid_first_gate],
        'time_race': time_race,
        'crashed_percentage': crashed_percentage,
        'mean_az': mean_acc_z,
        'max_az': max_acc_z,
        'mean_jerk_z': mean_jerk_z,
        'max_jerk_z': max_jerk_z,
    }

    if plotting:
        plt.figure()
        plt.scatter(time_2_laps, ep_max_gate_dist)
        plt.xlabel('time full track')
        plt.ylabel('max gate distance')
        plt.title('max gate distance vs total track time')
        plt.show()

    if printing:
        print('--------------------------------')
        print('mean episode reward:', np.mean(info['ep_rew']))
        print('mean episode length:', np.mean(info['ep_len']))
        print('num gates passed:', np.mean(info['num_gates_passed']))
        print('crash percentage:', info['crashed_percentage'])
        print('mean velocity:', np.mean(info['mean_vel']))
        print('mean_az:', np.mean(info['mean_az']))
        print('max_az:', np.mean(info['max_az']))
        print('mean gate distance:', np.mean(info['mean_gate_dist']))
        if len(info['time_2_laps']) > 0:
            print('mean full track time:', np.mean(info['time_2_laps']))
        if len(info['time_race']) > 0:
            print('mean first gate to last time:', np.mean(info['time_race']))
        print('--------------------------------\n')
    return info


# C CODE EXPORT

def generate_c_code(source_file_path, name, network, network_std, flightplan):

    float_to_str = lambda x: str(float(x))

    with open(source_file_path, "w") as file:
        file.write("#include <stdlib.h>\n")
        file.write("#include \"common/neural_math.h\"\n")
        file.write("#include \"flight/nn_control.h\"\n")
        file.write("#include \"flight/flightplan.h\"\n\n")

        j = 0
        for i, layer in enumerate(network):
            if isinstance(layer, nn.Linear):
                j += 1
                weights_layer = layer.weight.data.cpu().numpy()
                biases_layer = layer.bias.data.cpu().numpy()

                file.write(f"static const float weights_fc{j}[] = {{ \n")
                for i in range(weights_layer.shape[0]):
                    file.write(", ".join([float_to_str(x) for x in weights_layer[i]]))
                    file.write(", \n")
                file.write("};\n\n")
                file.write(f"static const float biases_fc{j}[] = {{ \n")
                file.write(", ".join([float_to_str(x) for x in biases_layer.flatten()]))
                file.write("\n};\n\n")

        j = 0
        for i, layer in enumerate(network):
            if isinstance(layer, nn.Linear):
                j += 1
                file.write(f"static float activations_fc{j}[{layer.out_features}];\n\n")

        file.write("static const float output_std[] = { \n")
        file.write(", ".join([float_to_str(x) for x in network_std.flatten()]))
        file.write("\n};\n\n")

        file.write("static nn_layer_t layers[] = {\n")
        j = 0
        for i, layer in enumerate(network):
            next_layer = network[i+1] if i+1 < len(network) else None
            activation = 'NULL'
            if next_layer is not None and isinstance(next_layer, nn.ReLU):
                activation = 'nn_neuron_relu'
            elif next_layer is not None and isinstance(next_layer, nn.Tanh):
                activation = 'nn_neuron_tanh'
            if isinstance(layer, nn.Linear):
                j += 1
                file.write(f"    {{{layer.in_features}, {layer.out_features}, weights_fc{j}, biases_fc{j}, activations_fc{j}, {activation}}},\n")
        n_layers = j
        file.write("};\n\n")

        file.write("static nn_network_t network = {\n")
        file.write(f"    .name = \"{name}\",\n")
        file.write(f"    .n_in = {network[0].in_features},\n")
        file.write(f"    .n_out = {network[-1].out_features},\n")
        file.write(f"    .n_layers = {n_layers},\n")
        file.write("    .layers = layers,\n")
        file.write("    .stddev_gaussian = output_std,\n")
        file.write("};\n\n")

        file.write("static nn_controller_t controller = {\n")
        file.write("    .network = &network,\n")
        file.write("    .gates_ahead = 1,\n")
        file.write("    .run_func = nn_default_run_func,\n")
        file.write("};\n\n")

        file.write(f"static waypoint_t waypoints[{flightplan.num_gates+1}] = " + "{\n")

        file.write("    // x,y,z, psi, gate_psi, v_sp, type\n")
        file.write("    // first waypoint must have WAYPOINT_START, second must have WAYPOINT_GATE\n")
        file.write("    // END or anything else not supported\n")
        file.write("    {")
        file.write(", ".join([float_to_str(x) for x in flightplan.start_pos]))
        file.write(", ")
        file.write(", ".join([float_to_str(x) for x in [flightplan.gate_yaw[0], 0., 0.]]))
        file.write(", WAYPOINT_START},\n")
        for i in range(len(flightplan.gate_pos)):
            file.write("    {")
            file.write(", ".join([float_to_str(x) for x in flightplan.gate_pos[i]]))
            file.write(", ")
            file.write(", ".join([float_to_str(x) for x in [0., flightplan.gate_yaw[i], 0.]]))
            file.write(", WAYPOINT_GATE},\n")
        file.write("};\n\n")
        file.write("static flightplan_t flightplan_static = {\n")
        file.write(f"    .num = {flightplan.num_gates+1},\n")
        file.write("    .waypoints = waypoints,\n")
        file.write("    .next_waypoint = &waypoints[1],\n")
        file.write("    .next_waypoint_idx = 1,\n")
        file.write("    .num_gates_passed = 0,\n")
        file.write(f"    .num_gates_recovery = {flightplan.num_gates_recovery}\n")
        file.write("};\n\n")
        file.write("\n")

        file.write(f"void nn_use_{name}(void) {{\n")
        file.write("    neural_controller = &controller;\n")
        file.write("    flightplan = &flightplan_static;\n")
        file.write("}\n")
        file.write("\n")
