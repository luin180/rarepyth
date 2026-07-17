# -*- coding: utf-8 -*-
"""
VASP SP Preparation & Transfer Packager:
Reads pre-clustered global minima from the MH phase, selects the top candidates,
generates VASP Single-Point calculation directories, and packages them for cross-cluster transfer.
"""

import os
import shutil
import math
import tarfile

template_dir = os.path.realpath('templates')

if not os.path.exists('TIMESTAMP_ID'):
    print("Error: TIMESTAMP_ID not found. Run step0_mh.py first.")
    exit(1)

with open('TIMESTAMP_ID', 'r') as file:
    work_dir = file.readline().strip()

if __name__ == "__main__":
    sp_base = os.path.join(work_dir, 'sp')
    os.makedirs(sp_base, exist_ok=True)

    mh_base = os.path.join(work_dir, 'mh')
    if not os.path.exists(mh_base):
        print(f"Error: Directory {mh_base} does not exist. Please run MH phase first.")
        exit(1)

    system_dirs = [d for d in os.listdir(mh_base) if os.path.isdir(os.path.join(mh_base, d))]
    total_sp_tasks = 0

    print("Step 1: Extracting Top Unique Minima for VASP Single-Point Calculations...")

    for sys_name in system_dirs:
        sys_dir = os.path.join(mh_base, sys_name)
        final_dir = os.path.join(sys_dir, "FINAL_GLOBAL_MINIMA")
        ee_file = os.path.join(final_dir, "FINAL_MINIMA_EE")

        if not os.path.exists(ee_file):
            print(f"  [Skip] {sys_name}: No FINAL_GLOBAL_MINIMA found.")
            continue

        # Find a TAG file from any of the seed directories (geometry constraints are identical)
        tag_file = None
        for root, dirs, files in os.walk(sys_dir):
            if 'TAG' in files:
                tag_file = os.path.join(root, 'TAG')
                break

        with open(ee_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        total_unique = len(lines)
        if total_unique == 0:
            continue

        # Select Top 20% (minimum 6, up to total available)
        select_count = min(max(6, math.ceil(total_unique * 0.20)), total_unique)
        print(f"  -> Processing {sys_name}: Selected Top {select_count}/{total_unique} states.")

        for idx in range(select_count):
            source_vasp = os.path.join(final_dir, f"FINAL_MINIMA_{idx}.vasp")
            if not os.path.exists(source_vasp):
                continue

            sp_path = os.path.join(sp_base, f'{sys_name}_UNIQUE_{idx}')
            os.makedirs(sp_path, exist_ok=True)

            # Copy geometry and TAG
            shutil.copy(source_vasp, os.path.join(sp_path, 'POSCAR.vasp'))
            if tag_file:
                shutil.copy(tag_file, sp_path)

            # Copy SLURM and VASP templates
            for tpl in ['vasp_sp.py', 'vasp_sp.sh', 'INCAR.ini']:
                tpl_path = os.path.join(template_dir, tpl)
                if os.path.exists(tpl_path):
                    shutil.copy(tpl_path, sp_path)
                else:
                    print(f"    [Warning] Template {tpl} not found in {template_dir}")

            total_sp_tasks += 1

    print(f"\nSuccessfully prepared {total_sp_tasks} VASP computation directories.")

    # Step 2: Package for Cross-Cluster Transfer
    if total_sp_tasks > 0:
        print("\nStep 2: Packaging for cross-cluster transfer...")

        timestamp_label = os.path.basename(work_dir)
        tar_filename = os.path.join(work_dir, f'vasp_sp_tasks_{timestamp_label}.tar.gz')

        with tarfile.open(tar_filename, "w:gz") as tar:
            tar.add(sp_base, arcname=os.path.basename(sp_base))

        print(f"  -> Archive created: {tar_filename}")
        print("Done! You can now use SCP/Rsync to transfer this archive to your VASP cluster.")
        print(f"Example: scp {tar_filename} user@vasp-cluster:~/scratch/")
    else:
        print("\nNo tasks were generated. Packaging skipped.")
