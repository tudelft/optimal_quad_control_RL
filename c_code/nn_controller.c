#include "nn_controller.h"
#include <math.h>
#include <stdlib.h>

bool deterministic = false;

const float output_std[4] = {
    0.8907372355461121,
    0.8487850427627563,
    0.8973739147186279,
    0.8708917498588562,
};

const float gate_pos[NUM_GATES][3] = {
    {2.0, -1.5, -1.5},
    {2.0, 1.5, -1.5},
    {-2.0, 1.5, -1.5},
    {-2.0, -1.5, -1.5},
    {2.0, -1.5, -1.5},
    {2.0, 1.5, -1.5},
    {-2.0, 1.5, -1.5},
    {-2.0, -1.5, -1.5},
};

const float gate_yaw[NUM_GATES] = {
    0.7853981852531433,
    2.356194496154785,
    3.9269907474517822,
    5.497786998748779,
    0.7853981852531433,
    2.356194496154785,
    3.9269907474517822,
    5.497786998748779,
};

const float start_pos[3] = {
    -2.0, -1.5, -1.5
};

const float gate_pos_rel[NUM_GATES][3] = {
    {2.8284265995025635, 2.82842755317688, 0.0},
    {2.1213202476501465, 2.1213202476501465, 0.0},
    {2.8284270763397217, 2.8284270763397217, 0.0},
    {2.1213202476501465, 2.1213204860687256, 0.0},
    {2.8284265995025635, 2.82842755317688, 0.0},
    {2.1213202476501465, 2.1213202476501465, 0.0},
    {2.8284270763397217, 2.8284270763397217, 0.0},
    {2.1213202476501465, 2.1213204860687256, 0.0},
};

const float gate_yaw_rel[NUM_GATES] = {
    -4.71238899230957,
    1.570796251296997,
    1.570796251296997,
    1.570796251296997,
    -4.71238899230957,
    1.570796251296997,
    1.570796251296997,
    1.570796251296997,
};

uint8_t target_gate_index = 0;

void nn_reset() {
    target_gate_index = 0;
}

void nn_control(const float world_state[16], const float disturbances[4], float rpms[4]) {
    // Get the current position, velocity and heading
    float pos[3] = {world_state[0], world_state[1], world_state[2]};
    float vel[3] = {world_state[3], world_state[4], world_state[5]};
    float yaw = world_state[8];

    // Get the position and heading of the target gate
    float target_pos[3] = {gate_pos[target_gate_index][0], gate_pos[target_gate_index][1], gate_pos[target_gate_index][2]};
    float target_yaw = gate_yaw[target_gate_index];

    // Set the target gate index to the next gate if we passed through the current one
    if (cosf(target_yaw) * (pos[0] - target_pos[0]) + sinf(target_yaw) * (pos[1] - target_pos[1]) > 0) {
        target_gate_index++;
        // loop back to the first gate if we reach the end
        target_gate_index = target_gate_index % NUM_GATES;
        // reset the target position and heading
        target_pos[0] = gate_pos[target_gate_index][0];
        target_pos[1] = gate_pos[target_gate_index][1];
        target_pos[2] = gate_pos[target_gate_index][2];
        target_yaw = gate_yaw[target_gate_index];
    }

    // Get the position of the drone in gate frame
    float pos_rel[3] = {
        cosf(target_yaw) * (pos[0] - target_pos[0]) + sinf(target_yaw) * (pos[1] - target_pos[1]),
        -sinf(target_yaw) * (pos[0] - target_pos[0]) + cosf(target_yaw) * (pos[1] - target_pos[1]),
        pos[2] - target_pos[2]
    };

    // Get the velocity of the drone in gate frame
    float vel_rel[3] = {
        cosf(target_yaw) * vel[0] + sinf(target_yaw) * vel[1],
        -sinf(target_yaw) * vel[0] + cosf(target_yaw) * vel[1],
        vel[2]
    };

    // Get the heading of the drone in gate frame
    float yaw_rel = yaw - target_yaw;
    while (yaw_rel > M_PI) {yaw_rel -= 2*M_PI;}
    while (yaw_rel < -M_PI) {yaw_rel += 2*M_PI;}

    // Get the neural network input
    float nn_input[16+4*GATES_AHEAD+4];
    // position and velocity
    for (int i = 0; i < 3; i++) {
        nn_input[i] = pos_rel[i];
        nn_input[i+3] = vel_rel[i];
    }
    // attitude
    nn_input[6] = world_state[6];
    nn_input[7] = world_state[7];
    nn_input[8] = yaw_rel;
    // body rates
    nn_input[9] = world_state[9];
    nn_input[10] = world_state[10];
    nn_input[11] = world_state[11];
    // motor rpms scaled to [-1,1]
    float w_min = 3000;
    float w_max = 11000;
    nn_input[12] = (world_state[12] - w_min) * 2 / (w_max - w_min) - 1;
    nn_input[13] = (world_state[13] - w_min) * 2 / (w_max - w_min) - 1;
    nn_input[14] = (world_state[14] - w_min) * 2 / (w_max - w_min) - 1;
    nn_input[15] = (world_state[15] - w_min) * 2 / (w_max - w_min) - 1;

    // relative gate positions and headings
    for (int i = 0; i < GATES_AHEAD; i++) {
        uint8_t index = target_gate_index + i + 1;
        // loop back to the first gate if we reach the end
        index = index % NUM_GATES;
        nn_input[16+4*i]   = gate_pos_rel[index][0];
        nn_input[16+4*i+1] = gate_pos_rel[index][1];
        nn_input[16+4*i+2] = gate_pos_rel[index][2];
        nn_input[16+4*i+3] = gate_yaw_rel[index];
    }
    // disturbance input
    float Mx_min = -0.03;
    float Mx_max = 0.03;
    float My_min = -0.03;
    float My_max = 0.03;
    float Mz_min = -0.01;
    float Mz_max = 0.01;
    float Fz_min = -0.5;
    float Fz_max = 0.5;
    nn_input[16+4*GATES_AHEAD]   = (disturbances[0] - Mx_min) * 2 / (Mx_max - Mx_min) - 1;
    nn_input[16+4*GATES_AHEAD+1] = (disturbances[1] - My_min) * 2 / (My_max - My_min) - 1;
    nn_input[16+4*GATES_AHEAD+2] = (disturbances[2] - Mz_min) * 2 / (Mz_max - Mz_min) - 1;
    nn_input[16+4*GATES_AHEAD+3] = (disturbances[3] - Fz_min) * 2 / (Fz_max - Fz_min) - 1;

    // Get the neural network output and write to the action array
    float nn_output[4];
    nn_forward(nn_input, nn_output);

    // add gaussian noise to the output
    if (!deterministic) {
        for (int i = 0; i < 4; i++) {
            // generate random gaussian variables using the Box–Muller transform
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float rand_std = sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
            // add the noise to the output
            nn_output[i] += output_std[i] * rand_std;
        }
    }

    for (int i = 0; i < 4; i++) {
        // clip the output to the range [-1, 1]
        if (nn_output[i] > 1) {nn_output[i] = 1;}
        if (nn_output[i] < -1) {nn_output[i] = -1;}
         // map the output to the range [w_min, w_max]
        rpms[i] = (w_max - w_min) * (nn_output[i] + 1) / 2 + w_min;
    }
}
