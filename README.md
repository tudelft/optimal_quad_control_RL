# quad_RL
Reinforcement learning for time optimal end-to-end quadcopter control

Main notebooks:

**3D quad race:**
- gym environment of end-to-end quadcopter model
    - bebop quadcopter model from https://arxiv.org/pdf/2304.13460.pdf
    - learned residual model (trained in the NNDroneModel notebook)
    - constant disturbances
- training pipeline
- automatic c code generation

**3D quad race INDI inner loop:**
- gym environment of quadcopter model with thrust and rate inputs
    - INDI controller is modeled as first order delay
- training pipeline
- automatic c code generation
