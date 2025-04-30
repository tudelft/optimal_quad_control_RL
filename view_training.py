import os
from stable_baselines3 import PPO
from quad_race_env import *
from quadcopter_animation import animation

# we will animate the trained models in folder model/...
models_dir = 'models/DR_E2E'

# let the user select the model to animate
models = os.listdir(models_dir)
print("Select a model to animate:")
for i, model in enumerate(models):
    print(f"{i}: {model}")
model_idx = int(input("Enter model index: "))

# load the selected model
model = models[model_idx]
print(f"View models saved in {models_dir}/{model}")

models = os.listdir(f'{models_dir}/{model}')
print('the following models are available:')
# the names are a number followed by .zip
# sort the names by the number
models.sort(key=lambda x: int(x.split('.')[0]))

i = 0
# load model
model = PPO.load(f'{models_dir}/{model}/{models[i]}')

# create the environment
env = model.env
print(env)