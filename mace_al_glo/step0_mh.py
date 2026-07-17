# -*- coding: utf-8 -*-
"""
Lightweight Job Dispatcher with 2D High-Throughput Batching:
Strictly reads structures from 'workbench/slabs' and 'workbench/molecules'.
Generates the Cartesian product of selected slabs and molecules,
then submits Minima Hopping jobs to the scheduler.
"""

import os
import shutil
import random
import datetime
import sys

# Batch configurations
slab_batch = 1
molecule_batch = 3

workbench_dir = os.path.realpath('workbench')
slabs_dir = os.path.join(workbench_dir, 'slabs')
molecules_dir = os.path.join(workbench_dir, 'molecules')

template_dir = os.path.realpath('templates')
filtered_checkpoints_dir = os.path.realpath('filtered_checkpoints')

if __name__ == "__main__":
    # 1. Strict Directory Validation
    if not os.path.exists(slabs_dir) or not os.path.exists(molecules_dir):
        print(f"Error: Strict mode requires both '{slabs_dir}' and '{molecules_dir}' to exist.")
        sys.exit(1)

    # Gather structural pools
    slabs_list = sorted([os.path.join(slabs_dir, f) for f in os.listdir(slabs_dir) if os.path.isfile(os.path.join(slabs_dir, f))])
    mols_list = sorted([os.path.join(molecules_dir, f) for f in os.listdir(molecules_dir) if os.path.isfile(os.path.join(molecules_dir, f))])

    if not slabs_list or not mols_list:
        print("Error: The 'slabs' or 'molecules' directory is empty.")
        sys.exit(1)

    # 2. Iteration Tracking
    if not os.path.exists('ITER'):
        _iter = 0
        with open('ITER', 'w') as file:
            file.write('0')
    else:
        with open('ITER', 'r') as file:
            _iter = int(file.readline())

    if _iter:
        best_checkpoint = os.path.join(filtered_checkpoints_dir, f'ft_iter{_iter - 1}.model')
    else:
        best_checkpoint = os.path.join(filtered_checkpoints_dir, 'mace_mp_medium.model')

    # 3. Workspace Initialization
    work_dir = f'work_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    mh_base = os.path.join(work_dir, 'mh')
    os.makedirs(mh_base, exist_ok=True)

    with open('TIMESTAMP_ID', 'w') as file:
        file.write(work_dir)

    # 4. 2D Strided Random Sampling
    slab_len = len(slabs_list)
    mol_len = len(mols_list)

    # Sample Slabs
    start_s = random.randint(0, max(0, slab_len // slab_batch - 1)) if slab_len >= slab_batch else 0
    selected_slabs = [
        slabs_list[(count * slab_len // slab_batch + start_s) % slab_len]
        for count in range(min(slab_batch, slab_len))
    ]

    # Sample Molecules
    start_m = random.randint(0, max(0, mol_len // molecule_batch - 1)) if mol_len >= molecule_batch else 0
    selected_mols = [
        mols_list[(count * mol_len // molecule_batch + start_m) % mol_len]
        for count in range(min(molecule_batch, mol_len))
    ]

    total_jobs = len(selected_slabs) * len(selected_mols)
    print("Step 1: High-Throughput Batching Initialized.")
    print(f"  -> Slabs Pool: {slab_len} | Selected: {len(selected_slabs)}")
    print(f"  -> Molecules Pool: {mol_len} | Selected: {len(selected_mols)}")
    print(f"  -> Total Jobs in this batch: {total_jobs}")

    # 5. Generate Cartesian Product and Dispatch
    print("\nStep 2: Preparing workspaces and dispatching jobs...")

    for slab_file in selected_slabs:
        for mol_file in selected_mols:
            # Extract filenames without extensions for the job directory
            s_name = os.path.splitext(os.path.basename(slab_file))[0]
            m_name = os.path.splitext(os.path.basename(mol_file))[0]
            system_name = f"{s_name}_{m_name}"

            target_dir = os.path.join(mh_base, system_name)
            os.makedirs(target_dir, exist_ok=True)

            # Copy selected structures to the system directory
            shutil.copy(slab_file, os.path.join(target_dir, 'slab.vasp'))
            shutil.copy(mol_file, os.path.join(target_dir, 'adsorbate.vasp'))

            # Copy execution scripts from templates
            shutil.copy(os.path.join(template_dir, 'run_mh.py'), target_dir)
            shutil.copy(os.path.join(template_dir, 'run_mh.sh'), target_dir)

            # Write checkpoint configuration
            with open(os.path.join(target_dir, 'CHECKPOINT_PATH'), 'w') as file:
                file.write(best_checkpoint)

            print(f"cd {target_dir} && sbatch --job-name=mh_{system_name} run_mh.sh")

    print(f"\nBatch generation complete. Submitted {total_jobs} combined tasks.")
