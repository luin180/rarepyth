# -*- coding: utf-8 -*-
"""
Compute-node Entry Point:
Executes the full pipeline for a single slab-adsorbate system.
1. Generates and screens initial guess configurations.
2. Runs parallel Minima Hopping for each selected seed.
3. Performs global secondary aggregation across all seeds.
"""

import os
import sys
import time
import shutil
import threading
import subprocess
import numpy as np

import ase.io
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import MACECalculator

from rarepyth.slab.adsorber import (
    load_molecule_from_file, generate_rotated_molecules,
    remove_duplicate_molecules_robust, find_filtered_adsorption_sites,
    generate_adsorption_structures_by_sites
)
from rarepyth.mace.matcher import is_equivalent_state

# =============================================================================
# Helper Utilities
# =============================================================================


def get_ram_used_mb():
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        return (int(lines[0].split()[1]) - int(lines[2].split()[1])) / 1024.0
    except Exception:
        return 0.0


def monitor_resources(stop_event, interval=5, log_file="hardware_monitor.csv"):
    with open(log_file, "w") as f:
        f.write("timestamp,ram_used_MB,gpu_vram_used_MB,gpu_util_percent\n")
        while not stop_event.is_set():
            ram_used = get_ram_used_mb()
            try:
                smi_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                ).strip().split('\n')
                vram_used = sum(float(line.split(',')[0]) for line in smi_output)
                gpu_util = sum(float(line.split(',')[1]) for line in smi_output) / len(smi_output)
            except Exception:
                vram_used, gpu_util = 0.0, 0.0

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{ram_used:.1f},{vram_used:.1f},{gpu_util:.1f}\n")
            f.flush()
            stop_event.wait(interval)


def generate_constraint_files(ads_struct, slab_len, target_dir):
    tag_path = os.path.join(target_dir, 'TAG')
    hookean_path = os.path.join(target_dir, 'HOOKEAN')
    ads_len = len(ads_struct) - slab_len

    with open(tag_path, 'w', encoding='utf-8') as f:
        for i in range(slab_len):
            f.write(f"{i + 1} 0\n")
        for i in range(ads_len):
            f.write(f"{slab_len + i + 1} 2\n")

    slab_coords = ads_struct.cart_coords[:slab_len]
    slab_max_z = np.max(slab_coords[:, 2])
    z_threshold = slab_max_z + 6.0

    with open(hookean_path, 'w', encoding='utf-8') as f:
        f.write("# Z-axis Hookean Constraint\n")
        f.write(f"Z_MAX {z_threshold:.4f}\n")
        f.write("SPRING_K 10.0\n")
        f.write("ATOMS " + " ".join(str(slab_len + i + 1) for i in range(ads_len)) + "\n")

# =============================================================================
# Global Aggregation (Secondary Clustering)
# =============================================================================


def perform_final_aggregation(work_root=".",
                              output_dir="FINAL_GLOBAL_MINIMA",
                              adsorbate_tag=2,
                              energy_tol=0.1,
                              rel_dev_tol=0.15):
    """
    Performs the secondary (global) aggregation across all initial guess directories,
    applying the 4-step topological and spatial equivalence filter to remove duplicate states.
    """
    print("\n--- Starting Secondary Aggregation ---")

    top_folders = [f for f in os.listdir(work_root) if f.startswith("Top") and os.path.isdir(os.path.join(work_root, f))]
    if not top_folders:
        print("No 'TopX_idxY' directories found to aggregate.")
        return

    adsorbate_indices = []
    for top_folder in top_folders:
        tag_path = os.path.join(work_root, top_folder, "TAG")
        if os.path.exists(tag_path):
            with open(tag_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and int(line.split()[1]) == adsorbate_tag:
                        adsorbate_indices.append(int(line.split()[0]) - 1)
            break

    if not adsorbate_indices:
        print("Warning: Could not identify adsorbate atoms from TAG files.")
        return

    raw_minima = []
    for top_folder in sorted(top_folders):
        local_results_dir = os.path.join(work_root, top_folder, "GLOBAL_MINIMA_RESULTS")
        local_ee_file = os.path.join(local_results_dir, "GLOBAL_MINIMA_EE")

        if not os.path.exists(local_ee_file):
            continue

        vasp_files = [f for f in os.listdir(local_results_dir) if f.startswith("GLOBAL_MINIMA") and f.endswith(".vasp")]
        for vasp_file in vasp_files:
            idx_str = vasp_file.replace("GLOBAL_MINIMA", "").replace(".vasp", "")
            try:
                idx = int(idx_str)
                with open(local_ee_file, 'r', encoding='utf-8') as f:
                    lines = [l for l in f.readlines() if l.strip()]
                    if idx < len(lines):
                        energy = float(lines[idx].split('#')[0].strip())
                        raw_minima.append({
                            "id": f"{top_folder}_{vasp_file}",
                            "energy": energy,
                            "atoms": ase.io.read(os.path.join(local_results_dir, vasp_file)),
                            "source": top_folder
                        })
            except ValueError:
                continue

    if not raw_minima:
        print("No valid minima found across local directories.")
        return

    print(f"Collected {len(raw_minima)} local minima. Starting final deduplication...")

    sample_atoms = raw_minima[0]["atoms"]
    substrate_indices = [i for i in range(len(sample_atoms)) if i not in adsorbate_indices]

    num_items = len(raw_minima)
    adj_list = {i: [] for i in range(num_items)}

    # Graph Theory Clustering using the integrated 4-step equivalence logic
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if is_equivalent_state(
                atoms1=raw_minima[i]["atoms"],
                atoms2=raw_minima[j]["atoms"],
                energy1=raw_minima[i]["energy"],
                energy2=raw_minima[j]["energy"],
                ads_idx=adsorbate_indices,
                sub_idx=substrate_indices,
                energy_tol=energy_tol,
                rel_dev_tol=rel_dev_tol
            ):
                adj_list[i].append(j)
                adj_list[j].append(i)

    visited = set()
    clusters = []
    for i in range(num_items):
        if i not in visited:
            queue = [i]
            current_cluster = []
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                current_cluster.append(curr)
                for neighbor in adj_list[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(current_cluster)

    unique_groups = []
    for cluster_indices in clusters:
        best_idx = min(cluster_indices, key=lambda idx: raw_minima[idx]["energy"])
        sources_found = set(raw_minima[idx]["source"] for idx in cluster_indices)
        unique_groups.append({
            "representative": raw_minima[best_idx],
            "members_count": len(cluster_indices),
            "sources": sources_found
        })

    unique_groups.sort(key=lambda g: g["representative"]["energy"])

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    out_ee_path = os.path.join(output_dir, "FINAL_MINIMA_EE")
    with open(out_ee_path, 'w', encoding='utf-8') as f_out:
        for new_idx, group in enumerate(unique_groups):
            rep = group["representative"]
            sources_str = ",".join(sorted(group["sources"]))
            out_vasp = os.path.join(output_dir, f"FINAL_MINIMA_{new_idx}.vasp")
            ase.io.write(out_vasp, rep["atoms"], format="vasp")
            f_out.write(f"{rep['energy']:.6f}  # Orig: {rep['id']} | Merged: {group['members_count']} | Found By: [{sources_str}]\n")

    strategy_coverage = {folder: [] for folder in top_folders}
    for new_idx, group in enumerate(unique_groups):
        for source in group["sources"]:
            if source in strategy_coverage:
                strategy_coverage[source].append(new_idx)

    coverage_file = os.path.join(output_dir, "INITIAL_GUESS_COVERAGE.txt")
    with open(coverage_file, 'w', encoding='utf-8') as f_cov:
        f_cov.write("==================================================\n")
        f_cov.write("Initial Guess Coverage Report\n")
        f_cov.write("==================================================\n")
        f_cov.write(f"Total Unique Minima Found Globally: {len(unique_groups)}\n\n")
        for top_folder in sorted(top_folders):
            covered_indices = strategy_coverage[top_folder]
            hit_rate = len(covered_indices) / len(unique_groups) * 100 if unique_groups else 0
            f_cov.write(f"Initial Guess: {top_folder}\n")
            f_cov.write(f"  Coverage: {len(covered_indices)}/{len(unique_groups)} ({hit_rate:.1f}%)\n")
            f_cov.write(f"  Captured FINAL IDs: {covered_indices}\n\n")

    print(f"Secondary aggregation complete! Reduced down to {len(unique_groups)} unique states.")

# =============================================================================
# Main Execution Flow
# =============================================================================


def main():
    print(f"=== Starting System Workflow in {os.getcwd()} ===")

    with open('CHECKPOINT_PATH', 'r') as file:
        mace_model_path = os.path.abspath(file.readline().strip())

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, 5, "hardware_monitor.csv"))
    monitor_thread.start()

    try:
        # Phase 1: Pre-screening and Seed Generation
        print("\n--- Phase 1: Initial Guess Generation ---")
        mace_calc = MACECalculator(
            model_paths=[mace_model_path], device='cuda', default_dtype='float64', enable_cueq=True
        )

        base_molecule = load_molecule_from_file('adsorbate.vasp')
        rotated_mols = generate_rotated_molecules(base_molecule, step_degrees=45.0)
        rotated_mols = remove_duplicate_molecules_robust(rotated_mols, rmsd_tol=0.05)

        slab = Structure.from_file('slab.vasp')
        slab_len = len(slab)
        target_sites = find_filtered_adsorption_sites(slab)

        if not target_sites:
            print("Error: No valid adsorption sites found.")
            return

        results = []
        for idx, mol in enumerate(rotated_mols):
            for jdx, site in enumerate(target_sites):
                ads_struct = generate_adsorption_structures_by_sites(slab, mol, [site])[0]
                ase_atoms = AseAtomsAdaptor.get_atoms(ads_struct)
                ase_atoms.calc = mace_calc
                try:
                    results.append({'idx': idx, 'jdx': jdx, 'energy': ase_atoms.get_potential_energy(), 'struct': ads_struct})
                except Exception:
                    pass

        results.sort(key=lambda x: x['energy'])
        N_SEEDS = 8
        top_seeds = results[:N_SEEDS]

        # Template for subdirectory execution
        run_mh_template = f"""import os
from rarepyth.mace.optimizer import MultiGlobalOptimizer

def main():
    multi_opt = MultiGlobalOptimizer(model_path="{mace_model_path}", initial_structure_path="POSCAR.vasp")

    multi_opt.add_strategy("aggressive_600", totalsteps=60, max_optimize_steps=750, use_auto_tether=True, use_auto_hookean=True,
                           T0=600.0, Ediff0=0.15, beta1=1.1, beta2=1.1, fmax=0.02, mdmin=3)

    multi_opt.add_strategy("extreme_1200", totalsteps=50, max_optimize_steps=750, use_auto_tether=True, use_auto_hookean=True,
                           T0=1200.0, Ediff0=0.15, beta1=1.1, beta2=1.1, fmax=0.02, mdmin=3)

    # Launch 2 Strategies = 2 Workers per seed directory
    multi_opt.run_parallel_exploration(manual_workers=2, memory_per_worker_gb=4.5)

if __name__ == "__main__":
    main()
"""

        print(f"\n--- Phase 2: Preparing workspaces & Launching {N_SEEDS * 2}-process parallel exploration ---")
        task_dirs = []
        for rank, res in enumerate(top_seeds):
            task_dir = os.path.join(os.getcwd(), f"Top{rank + 1}_{res['idx']}_{res['jdx']}")
            os.makedirs(task_dir, exist_ok=True)
            task_dirs.append(task_dir)

            res['struct'].to(os.path.join(task_dir, 'POSCAR.vasp'), fmt='poscar')
            generate_constraint_files(res['struct'], slab_len, task_dir)

            with open(os.path.join(task_dir, "run_mh.py"), "w", encoding="utf-8") as f:
                f.write(run_mh_template)

        child_env = os.environ.copy()
        child_env["PYTHONWARNINGS"] = "once::FutureWarning,once::UserWarning"

        running_processes = []
        for t_dir in task_dirs:
            print(f"  -> Starting workers in {os.path.basename(t_dir)}")
            p = subprocess.Popen([sys.executable, "run_mh.py"], cwd=t_dir, env=child_env)
            running_processes.append(p)

        for p in running_processes:
            p.wait()

        print(f"\nAll {N_SEEDS * 2} Parallel Minima Hopping tasks have completed successfully!")

        # Phase 3: Secondary Aggregation
        perform_final_aggregation(work_root=os.getcwd())

    finally:
        stop_event.set()
        monitor_thread.join()
        print("\nWorkflow completed. Hardware monitoring stopped.")


if __name__ == "__main__":
    main()
