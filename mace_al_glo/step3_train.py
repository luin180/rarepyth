# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:56:41 2026

@author: Wang Junhao
"""

import os
import shutil

from step0_mh import template_dir

with open('ITER', 'r') as file:
    _iter = int(file.readline())

with open('TIMESTAMP_ID', 'r') as file:
    work_dir = file.readline().strip()

all_set_dir = os.path.realpath('all_set')
os.makedirs(all_set_dir, exist_ok=True)

if __name__ == "__main__":
    # Collect all generated trajectories (now in .xyz format)
    for root, dirs, files in os.walk(os.path.join(work_dir, 'rlx')):
        for struct_dir in dirs:
            traj_path = os.path.join(root, struct_dir, 'traj.xyz')
            if os.path.exists(traj_path):
                shutil.copy(traj_path, os.path.join(all_set_dir, f'iter{_iter}_{struct_dir}.xyz'))

    train_dir = os.path.join(work_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    # Provide simple text file indicating data pool path for MACE
    with open(os.path.join(train_dir, 'data_source.txt'), 'w') as file:
        file.write(f"ALL_SET_DIR={all_set_dir}\n")
        file.write(f"CURRENT_ITER={_iter}\n")

    # Copy MACE training templates (Assumed names adapted to MACE)
    shutil.copy(os.path.join(template_dir, 'mace_train.py'), train_dir)
    shutil.copy(os.path.join(template_dir, 'mace_train.sh'), train_dir)

    print(f"cd {train_dir} && sbatch --job-name=train_iter{_iter} mace_train.sh")
