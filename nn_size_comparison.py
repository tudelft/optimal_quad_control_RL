import os
import sys
import time

def run_tmux_session(session_name, command):
    # Create a new tmux session and run the command
    os.system(f"tmux new-session -d -s {session_name} {command}")

if __name__ == "__main__":
    # we will run the following training session in parallel
    # 2 layer neural networks: (32,32), (64,64), (128,128), (256,256)
    architectures = [(32,32), (64,64), (128,128), (256,256)]
    for arch in architectures:
        session_name = f"session_{arch[0]}_{arch[1]}"
        command = f'python train.py {session_name} --pi {arch[0]} {arch[1]} --vf {arch[0]} {arch[1]}'
        run_tmux_session(session_name, command)
    
    print(f'Running {len(architectures)} parallel training sessions')
    
    # periodically check the status of the training sessions
    while True:
        os.system("tmux ls")
        print("Press Ctrl+C to stop")
        # check every 10 seconds
        time.sleep(10)
