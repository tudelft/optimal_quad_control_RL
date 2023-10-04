#ifndef NN_CONTROLLER_H
#define NN_CONTROLLER_H
#define NN_INDI_CMDS

#include <stdint.h>
#include <stdbool.h>

#define GATES_AHEAD 1
#define NUM_GATES 8

// Include the neural network code
#include "neural_network.h"

const float gate_pos[NUM_GATES][3];
const float gate_yaw[NUM_GATES];
const float start_pos[3];
uint8_t target_gate_index;

void nn_reset(void);
void nn_control(const float world_state[13], float indi_cmd[4]);

#endif
