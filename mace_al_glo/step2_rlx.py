# -*- coding: utf-8 -*-
"""
Relaxation Task Dispatcher & Packager:
Evaluates MACE predictive power, dispatches the worst-performing structures
for VASP local relaxation, and packages the generated tasks for cross-cluster transfer.
"""

import os
import shutil
import logging
import tarfile

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

from step0_mh import template_dir

rlx_batch = 20

with open('ITER', 'r') as file:
    _iter = int(file.readline())

if not os.path.exists('TIMESTAMP_ID'):
    print("Error: TIMESTAMP_ID not found. Run step0_mh.py first.")
    exit(1)

with open('TIMESTAMP_ID', 'r') as file:
    work_dir = file.readline().strip()

if __name__ == "__main__":
    mh_base = os.path.join(work_dir, 'mh')
    sp_base = os.path.join(work_dir, 'sp')
    rlx_base = os.path.join(work_dir, 'rlx')

    evaluation_list = []
    detailed_records = []
    total_rlx_tasks = 0

    print("Step 1: Evaluating MACE vs DFT energies...")

    for struct_dir in os.listdir(mh_base):
        task_dir = os.path.join(mh_base, struct_dir)
        mace_ee_file = os.path.join(task_dir, "FINAL_GLOBAL_MINIMA", "FINAL_MINIMA_EE")

        if not os.path.isdir(task_dir) or not os.path.exists(mace_ee_file):
            continue

        with open(mace_ee_file, 'r', encoding='utf-8') as f:
            mace_energies = [float(line.split()[0]) for line in f if line.strip()]

        valid_pairs = []
        for idx, e_mace in enumerate(mace_energies):
            dft_sp_dir = os.path.join(sp_base, f"{struct_dir}_UNIQUE_{idx}")
            dft_ee_file = os.path.join(dft_sp_dir, 'MINIMA_DFT_EE')

            if os.path.exists(dft_ee_file):
                with open(dft_ee_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    if lines:
                        valid_pairs.append((idx, e_mace, float(lines[-1])))

        if len(valid_pairs) < 3:
            logging.warning(f'Insufficient DFT energies for {struct_dir}. Skipped.')
            continue

        mace_e = np.array([p[1] for p in valid_pairs])
        dft_e = np.array([p[2] for p in valid_pairs])

        # 1. Base Correlation Metrics
        rho, _ = spearmanr(mace_e, dft_e)
        rho = rho if not np.isnan(rho) else 0.0

        tau, _ = kendalltau(mace_e, dft_e)
        tau = tau if not np.isnan(tau) else 0.0

        # 2. Rankings and Anchor points
        dft_min_index = np.argmin(dft_e)
        mace_sort_indices = np.argsort(mace_e)
        mace_rank_of_dft_min = np.where(mace_sort_indices == dft_min_index)[0][0] + 1  # 1-based

        mace_rel_true = mace_e - mace_e[dft_min_index]
        dft_rel_true = dft_e - dft_e[dft_min_index]

        for i, (idx, e_mace, e_dft) in enumerate(valid_pairs):
            detailed_records.append({
                'SYSTEM': struct_dir,
                'CONF_IDX': idx,
                'MACE_ENERGY_eV': e_mace,
                'DFT_ENERGY_eV': e_dft,
                'MACE_REL_eV': mace_rel_true[i],
                'DFT_REL_eV': dft_rel_true[i],
                'ERROR_REL_eV': mace_rel_true[i] - dft_rel_true[i]
            })

        # 3. Local DDE MAE (Dynamic Window)
        window_size = min(mace_rank_of_dft_min + 1, len(mace_e))
        window_indices = mace_sort_indices[:window_size]

        mace_window_rel = mace_e[window_indices] - mace_e[dft_min_index]
        dft_window_rel = dft_e[window_indices] - dft_e[dft_min_index]
        local_dde_mae = np.mean(np.abs(mace_window_rel - dft_window_rel))

        top1_hit = bool(mace_rank_of_dft_min == 1)
        top3_hit = bool(mace_rank_of_dft_min <= 3)

        evaluation_list.append({
            'STRUCTURE': struct_dir,
            'SPEARMAN_RHO': rho,
            'KENDALL_TAU': tau,
            'TOP1_HIT': top1_hit,
            'TOP3_HIT': top3_hit,
            'MACE_RANK_OF_DFT_MIN': mace_rank_of_dft_min,
            'LOCAL_DDE_MAE_eV': local_dde_mae,
            'VALID_INDICES': [p[0] for p in valid_pairs]
        })

    if detailed_records:
        df_details = pd.DataFrame(detailed_records)
        details_file = f'energy_details_iter{_iter}.csv'
        df_details.to_csv(details_file, index=False)
        print(f"  -> Detailed conformation energy comparisons saved to {details_file}")

    if evaluation_list:
        print("\nStep 2: Dispatching worst structures for VASP relaxation...")
        # Active Learning selection criteria (descending by LOCAL_DDE_MAE_eV)
        evaluation_list = sorted(evaluation_list, key=lambda x: x['LOCAL_DDE_MAE_eV'], reverse=True)

        df_eval = pd.DataFrame(evaluation_list)
        summary_file = f'energy_summary_iter{_iter}.csv'
        df_eval.to_csv(summary_file, index=False)
        print(f"  -> System-level evaluation summary saved to {summary_file}")

        os.makedirs(rlx_base, exist_ok=True)

        filtered_targets = [target for target in evaluation_list if target['LOCAL_DDE_MAE_eV'] >= 0.15]

        for target in filtered_targets[:min(len(filtered_targets), rlx_batch)]:
            struct_dir = target['STRUCTURE']
            for idx in target['VALID_INDICES']:
                subname = f'{struct_dir}_UNIQUE_{idx}'
                rlx_path = os.path.join(rlx_base, subname)
                os.makedirs(rlx_path, exist_ok=True)

                sp_path = os.path.join(sp_base, subname)

                shutil.copy(os.path.join(template_dir, 'vasp_relax.py'), rlx_path)
                shutil.copy(os.path.join(template_dir, 'vasp_relax.sh'), rlx_path)
                shutil.copy(os.path.join(template_dir, 'INCAR.ini'), rlx_path)

                shutil.copy(os.path.join(sp_path, 'POSCAR.vasp'), rlx_path)
                if os.path.exists(os.path.join(sp_path, 'TAG')):
                    shutil.copy(os.path.join(sp_path, 'TAG'), rlx_path)

                total_rlx_tasks += 1
                # print(f"cd {rlx_path} && sbatch --job-name=rlx_{subname} vasp_relax.sh")

        print(f"\nSuccessfully prepared {total_rlx_tasks} VASP relaxation directories.")

        # =====================================================================
        # Step 3: Package for Cross-Cluster Transfer
        # =====================================================================
        if total_rlx_tasks > 0:
            print("\nStep 3: Packaging for cross-cluster transfer...")
            timestamp_label = os.path.basename(work_dir)
            tar_filename = os.path.join(work_dir, f'vasp_rlx_tasks_{timestamp_label}.tar.gz')

            with tarfile.open(tar_filename, "w:gz") as tar:
                tar.add(rlx_base, arcname=os.path.basename(rlx_base))

            print(f"  -> Archive created: {tar_filename}")
            print("Done! You can now use SCP/Rsync to transfer this archive to your VASP cluster.")
            print(f"Example: scp {tar_filename} user@vasp-cluster:~/scratch/")

    else:
        print("\nNo valid evaluation pairs found. Skipping task generation and packaging.")
