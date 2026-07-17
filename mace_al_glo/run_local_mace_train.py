# -*- coding: utf-8 -*-
"""
MACE Train Launcher:
Reads work directory from TIMESTAMP_ID, executes mace_train.py,
filters specific warnings, and logs output to both terminal and file.
"""

import os
import sys
import subprocess
import datetime


def launch_mace_train():
    with open('TIMESTAMP_ID', 'r') as file:
        work_dir = file.readline().strip()

    train_script = os.path.join(work_dir, 'train', 'mace_train.py')

    if not os.path.exists(train_script):
        print(f"Error: Training script not found at {train_script}")
        return

    log_filename = f"mace_train_{work_dir}.log"
    print("Starting MACE training wrapper...")
    print(f"Log file: {log_filename}")

    child_env = os.environ.copy()
    child_env["PYTHONWARNINGS"] = "once::FutureWarning,once::UserWarning"

    with open(log_filename, 'w', encoding='utf-8') as log_f:
        process = subprocess.Popen(
            [sys.executable, "-u", 'mace_train.py'],
            cwd=os.path.join(work_dir, 'train'),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env
        )

        skip_next_line = False
        for line in process.stdout:
            if "FutureWarning: You are using `torch.load` with `weights_only=False`" in line:
                skip_next_line = True
                continue
            if "UserWarning: To copy construct from a tensor" in line:
                skip_next_line = True
                continue
            if "UserWarning: The TorchScript type system" in line:
                skip_next_line = True
                continue

            if skip_next_line:
                skip_next_line = False
                continue

            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()

        process.wait()

    if process.returncode == 0:
        print(f"\n[{datetime.datetime.now()}] Training completed successfully.")
    else:
        print(f"\n[{datetime.datetime.now()}] Training failed with exit code {process.returncode}.")


if __name__ == "__main__":
    launch_mace_train()
