import os
import sys
import time

def run_tmux_session(session_name, command):
    # Create a new tmux session and run the command
    os.system(f"tmux new-session -d -s {session_name} {command}")

if __name__ == "__main__":
    # we will run the following training session in parallel
    architectures = [(32,32,32), (64,64,64), (128,128,128), (256,256,256)]
    session_name = 'nn_size_comparison'
    for arch in architectures:
        for i in range(3):
           model_name = f"run{i}_{arch[0]}_{arch[1]}_{arch[2]}"
           command = f'python train.py {session_name} {model_name} --pi {arch[0]} {arch[1]} {arch[2]} --vf {arch[0]} {arch[1]} {arch[2]}'
           run_tmux_session(model_name, command)

    print(f'Running {len(architectures)*3} parallel training sessions')
