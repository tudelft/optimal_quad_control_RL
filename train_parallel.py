import os
import sys

def run_tmux_session(session_name, command):
    # Create a new tmux session and run the command
    os.system(f"tmux new-session -d -s {session_name} {command}")

if __name__ == "__main__":
    # get number of parallel sessions to run
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        for i in range(n):
            session_name = f"session_{i}"
            command = f'python train.py parallel_test_n={n}_i={i}'
            run_tmux_session(session_name, command)
        print(f'Running {n} parallel training sessions')
    else:
        print("Please provide the number of parallel sessions to run")
        sys.exit(1)
