# -*- coding: utf-8 -*-
"""
Local Distributed Executor:
Distributes MH tasks across multiple containers/GPUs by creating and reading a DISTRIBUTION file.
"""

import os
import sys
import subprocess
import datetime
import argparse
import json


def run_distributed_mh(node_id, total_nodes):
    with open('TIMESTAMP_ID', 'r') as file:
        work_dir = file.readline().strip()

    mh_base = os.path.join(work_dir, 'mh')
    if not os.path.exists(mh_base):
        print(f"Error: Directory {mh_base} does not exist.")
        return

    system_dirs = [d for d in os.listdir(mh_base) if os.path.isdir(os.path.join(mh_base, d))]
    system_dirs.sort()

    if not system_dirs:
        print("No system directories found to process.")
        return

    dist_file = os.path.join(mh_base, 'DISTRIBUTION.json')

    if not os.path.exists(dist_file):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DISTRIBUTION file not found.")
        print(f"Initializing distributed task mapping for {total_nodes} nodes...")

        mapping = {}
        for i, sys_dir in enumerate(system_dirs):
            assigned_node = (i * total_nodes) // len(system_dirs)
            mapping[sys_dir] = assigned_node

        tmp_file = dist_file + f".tmp_{node_id}"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4)

        try:
            os.replace(tmp_file, dist_file)
        except Exception:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    with open(dist_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    my_tasks = [d for d in system_dirs if mapping.get(d) == node_id]
    total_my_tasks = len(my_tasks)

    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] === Node Status ===")
    print(f"Current Node ID   : {node_id} (0-indexed)")
    print(f"Total System Nodes: {total_nodes}")
    print(f"Total MH Tasks    : {len(system_dirs)}")
    print(f"Tasks for Node {node_id}  : {total_my_tasks}")
    print("=========================\n")

    if total_my_tasks == 0:
        print("No tasks assigned to this node. Exiting gracefully.")
        return

    child_env = os.environ.copy()
    child_env["PYTHONWARNINGS"] = "once::FutureWarning,once::UserWarning"

    for idx, sys_dir in enumerate(my_tasks, 1):
        target_dir = os.path.join(mh_base, sys_dir)

        if not os.path.exists(os.path.join(target_dir, 'run_mh.py')):
            print(f"  [{idx}/{total_my_tasks}] Skipping {sys_dir}: run_mh.py not found.")
            continue

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{current_time}] [Node {node_id}] [{idx}/{total_my_tasks}] Executing workflow for {sys_dir}...")
        print("-" * 60)

        log_file = os.path.join(target_dir, 'job.local.out')

        with open(log_file, 'w') as out_f:
            process = subprocess.Popen(
                [sys.executable, "-u", "run_mh.py"],
                cwd=target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=child_env
            )

            skip = False
            skip_next_line = False
            skipped_text = [
                "FutureWarning: You are using `torch.load` with `weights_only=False`",
                "UserWarning: To copy construct from a tensor",
                "DeprecationWarning: Set OLD_ERROR_HANDLING",
            ]
            for line in process.stdout:
                for text in skipped_text:
                    if text in line:
                        skip = True
                        break

                if skip:
                    skip = False
                    skip_next_line = True
                    continue
                if skip_next_line:
                    skip_next_line = False
                    continue

                sys.stdout.write(line)
                sys.stdout.flush()
                out_f.write(line)
                out_f.flush()

            process.wait()

        print("-" * 60)
        if process.returncode == 0:
            print(f"  -> Finished {sys_dir} successfully.")
        else:
            print(f"  -> Error in {sys_dir}. Process exited with code {process.returncode}.")

    print(f"\nAll tasks assigned to Node {node_id} have been completed on local GPU!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Distributed MH Executor across Containers")
    parser.add_argument("--node_id", type=int, default=0, help="ID of this container/node (0-indexed, default: 0)")
    parser.add_argument("--total_nodes", type=int, default=3, help="Total number of containers/nodes to distribute tasks among (default: 3)")
    args = parser.parse_args()

    run_distributed_mh(args.node_id, args.total_nodes)

    # nohup python run_local_serial_mh.py --node_id 0 --total_nodes 3 > scheduler_00.log 2>&1 &
