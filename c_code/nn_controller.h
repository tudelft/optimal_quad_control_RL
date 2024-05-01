#ifndef NN_CONTROLLER_H
#define NN_CONTROLLER_H

#include <stdint.h>
#include <stdbool.h>

#define GATES_AHEAD 1
#define NUM_GATES 8

// Include the neural network code
#include "neural_network.h"

extern const float gate_pos[NUM_GATES][3];
extern const float gate_yaw[NUM_GATES];
extern const float start_pos[3];
extern const float start_yaw;
extern uint8_t target_gate_index;

void nn_reset(void);
void nn_control(const float world_state[16], float motor_cmds[4]);

#endif
