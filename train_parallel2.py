import os
import sys

def run_tmux_session(session_name, command):
    # Create a new tmux session and run the command
    os.system(f"tmux new-session -d -s {session_name} {command}")

if __name__ == "__main__":
    # 5 inch drone experiments
    # for r in ['5inch_10_percent', '5inch_20_percent', '5inch_30_percent']:
    #     for j in range(3):
    #         session_name = f'5inch_drone'
    #         model_name = f"run{j}_{r}"
    #         command = f'python train.py {session_name} {model_name} --pi 64 64 64 --vf 64 64 64 --randomization {r}'
    #         run_tmux_session(model_name, command)
            
    # 3 inch drone experiments
    for r in ['fixed_3inch', '3inch_10_percent', '3inch_20_percent', '3inch_30_percent']:
        for j in range(3):
            session_name = f'3inch_drone'
            model_name = f"run{j}_{r}"
            command = f'python train.py {session_name} {model_name} --pi 64 64 64 --vf 64 64 64 --randomization {r}'
            run_tmux_session(model_name, command)
    
    # State history experiments
    # for i in [1,2,3]:
    #     for j in range(3):
    #         session_name = 'input_comparison'
    #         model_name = f"run{j}_state_history{i}"
    #         command = f'python train.py {session_name} {model_name} --pi 64 64 64 --vf 64 64 64 --state_history {i}'
    #         run_tmux_session(model_name, command)
    
    # Action history experiments
    # for i in [0,1,2]:      
    #     for j in range(3):
    #         session_name = 'input_comparison'
    #         model_name = f"run{j}_param_input_noise=0{i}"
    #         command = f'python train.py {session_name} {model_name} --pi 64 64 64 --vf 64 64 64 --param_input --param_input_noise 0.{i}'
    #         run_tmux_session(model_name, command)

# open tensorboard
os.system(f"tensorboard --logdir logs/{session_name}")