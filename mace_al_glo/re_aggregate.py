'''re_aggregate.py — Re-run perform_final_aggregation on MH results.

Walks work_TIMESTAMP/mh/ and re-clusters all existing GLOBAL_MINIMA_RESULTS
from Top* directories via the 4-step equivalence classifier. Overwrites
the existing FINAL_GLOBAL_MINIMA/ in each system directory.

No GPU dependencies — pure CPU: ASE + numpy + rarepyth.mace.matcher.

Usage:
  python re_aggregate.py                              # all systems
  python re_aggregate.py --system Pt111_CO            # single system
  python re_aggregate.py --energy_tol 0.05 --rel_dev_tol 0.05  # tighter thresholds
'''

import os, sys, shutil, argparse
import ase.io
from rarepyth.mace.matcher import is_equivalent_state

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)


def perform_final_aggregation(work_root=".",
                              output_dir="FINAL_GLOBAL_MINIMA",
                              adsorbate_tag=2,
                              energy_tol=0.1,
                              rel_dev_tol=0.15):
    """
    Performs the secondary (global) aggregation across all initial guess
    directories, applying the 4-step topological and spatial equivalence
    filter to remove duplicate states.
    """
    print("--- Starting Secondary Aggregation ---")

    top_folders = [f for f in os.listdir(work_root)
                   if f.startswith("Top") and os.path.isdir(os.path.join(work_root, f))]
    if not top_folders:
        print("No 'Top*' directories found to aggregate.")
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

        vasp_files = [f for f in os.listdir(local_results_dir)
                      if f.startswith("GLOBAL_MINIMA") and f.endswith(".vasp")]
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


# ===================================================================
# CLI & main
# ===================================================================
def main():
    _tsid = os.path.join(root_dir, 'TIMESTAMP_ID')
    if not os.path.exists(_tsid):
        raise RuntimeError('TIMESTAMP_ID not found. Has step0_mh run?')
    with open(_tsid, 'r') as f:
        work_dir = os.path.join(root_dir, f.read().strip())

    mh_dir = os.path.join(work_dir, 'mh')
    if not os.path.isdir(mh_dir):
        raise RuntimeError(f'mh/ not found under {work_dir}.')

    parser = argparse.ArgumentParser(description='Re-aggregate MH minima across Top* seeds')
    parser.add_argument('--system', default=None, help='Only process a single system directory name')
    parser.add_argument('--energy_tol', type=float, default=0.1)
    parser.add_argument('--rel_dev_tol', type=float, default=0.15)
    args = parser.parse_args()

    if args.system:
        system_dirs = [args.system]
        if not os.path.isdir(os.path.join(mh_dir, args.system)):
            raise RuntimeError(f'System directory not found: mh/{args.system}')
    else:
        system_dirs = sorted([d for d in os.listdir(mh_dir)
                              if os.path.isdir(os.path.join(mh_dir, d))])

    if not system_dirs:
        print('No system directories found.')
        sys.exit(0)

    print(f'Work dir: {work_dir}')
    print(f'Systems : {len(system_dirs)}')
    print(f'Tolerances: energy_tol={args.energy_tol} eV, rel_dev_tol={args.rel_dev_tol}')
    print()

    for sys_dir in system_dirs:
        sys_path = os.path.join(mh_dir, sys_dir)
        top_dirs = [d for d in os.listdir(sys_path)
                    if d.startswith('Top') and os.path.isdir(os.path.join(sys_path, d))
                    and os.path.exists(os.path.join(sys_path, d, 'GLOBAL_MINIMA_RESULTS'))]

        if not top_dirs:
            print(f'  [Skip] {sys_dir}: no Top*/GLOBAL_MINIMA_RESULTS found')
            continue

        print(f'  [{sys_dir}]  {len(top_dirs)} Top* dirs -> re-aggregating...')
        owd = os.getcwd()
        os.chdir(sys_path)
        perform_final_aggregation(
            work_root='.',
            output_dir='FINAL_GLOBAL_MINIMA',
            adsorbate_tag=2,
            energy_tol=args.energy_tol,
            rel_dev_tol=args.rel_dev_tol,
        )
        os.chdir(owd)
        print()

    print('Done.')


if __name__ == '__main__':
    main()
