# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:26:57 2026

@author: Wang Junhao
"""

import os
import re
import shutil
import time
import multiprocessing as mp

import numpy as np
import torch
import ase.io
from ase import units
from ase.md.langevin import Langevin
from ase.constraints import FixAtoms, Hookean
from ase.optimize import FIRE2, BFGS
from ase.optimize.minimahopping import MinimaHopping
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase.md import MDLogger, VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Structure
from mace.calculators import MACECalculator

from rarepyth.mace.matcher import is_equivalent_state


AUTO_HOOKEAN_DIST = {frozenset({'C'}): [1.89, 1.89], frozenset({'N'}): [1.88, 1.88],
                     frozenset({'C', 'O'}): [1.97, 1.97], frozenset({'C', 'H'}): [1.20, 1.20],
                     frozenset({'C', 'N'}): [1.79, 1.79], frozenset({'H', 'O'}): [1.20, 1.20],
                     frozenset({'N', 'O'}): [1.82, 1.82], frozenset({'N', 'H'}): [1.20, 1.20]}

# =============================================================================
# 1. MACEOptimizer (For Local Relaxation & PES Exploration)
# =============================================================================
# (Content remains the same as the previous version)


class MACEOptimizer:
    def __init__(self, model_paths, device="cuda" if torch.cuda.is_available() else "cpu", default_dtype="float64", enable_cueq=True, dispersion=False):
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        self.calc = MACECalculator(model_paths=model_paths, device=device, default_dtype=default_dtype, enable_cueq=enable_cueq, dispersion=dispersion)
        self.structures = []

    def load_structure(self, filename):
        atoms = ase.io.read(filename)
        self.structures.append(atoms)
        struct = Structure.from_file(filename)
        constraint = []
        for idx in range(struct.num_sites):
            if 'selective_dynamics' in struct.sites[idx].properties.keys():
                if False in struct.sites[idx].properties['selective_dynamics']:
                    constraint.append(idx)
        if constraint:
            self.structures[-1].constraints = [FixAtoms(indices=constraint)]

    def calc_relaxation(self, forces_criteria=0.05, max_steps=500):
        energies = []
        for idx, structure in enumerate(self.structures):
            structure.calc = self.calc
            opt = FIRE2(structure, logfile=f'mace_relax_{idx}.log', trajectory=f'mace_relax_{idx}.traj')
            opt.run(fmax=forces_criteria, steps=max_steps)
            energy = structure.get_potential_energy()
            energies.append(energy)
            write_vasp(f'POSCAR_RELAXED_MACE_{idx}.vasp', structure)
            print(f"Relaxation {idx} complete. Energy: {energy:.4f} eV")
        return energies

    def robust_relax(self, forces_criteria=0.005, temp_k=200.0, md_steps=2000):
        energies = []
        for idx, structure in enumerate(self.structures):
            structure.calc = self.calc
            print(f"Phase 1: Thermal crawling at {temp_k} K for {md_steps} steps.")
            dyn = Langevin(structure, 0.5 * units.fs, temperature_K=temp_k, friction=0.02)
            dyn.run(md_steps)
            print(f"Phase 2: Quenching with FIRE2 (fmax={forces_criteria}, maxstep=0.4).")
            opt = FIRE2(structure, maxstep=0.4, trajectory=f'robust_relax_{idx}.traj', logfile=f'robust_relax_{idx}.log')
            opt.run(fmax=forces_criteria, steps=1500)
            energy = structure.get_potential_energy()
            energies.append(energy)
            write_vasp(f'POSCAR_ROBUST_RELAXED_{idx}.vasp', structure)
            print(f"Robust relaxation {idx} complete. Energy: {energy:.4f} eV")
        return energies

    def simulated_annealing(self, start_temp_k=150.0, anneal_steps=2000, fmax_criteria=0.01):
        energies = []
        for idx, structure in enumerate(self.structures):
            structure.calc = self.calc
            dyn = Langevin(structure, 0.5 * units.fs, temperature_K=start_temp_k, friction=0.02)
            cooling_stages = 10
            steps_per_stage = anneal_steps // cooling_stages
            for stage in range(cooling_stages):
                current_temp = start_temp_k * (1.0 - stage / (cooling_stages - 1))
                dyn.set_temperature(temperature_K=current_temp)
                dyn.run(steps_per_stage)
            opt = FIRE2(structure, trajectory=f'anneal_quench_{idx}.traj')
            opt.run(fmax=fmax_criteria, steps=500)
            energy = structure.get_potential_energy()
            energies.append(energy)
            write_vasp(f'POSCAR_ANNEALED_{idx}.vasp', structure)
            print(f"Annealing {idx} complete. Energy: {energy:.4f} eV")
        return energies


# =============================================================================
# 2. AdjustedMinimaHopping (Helper Class)
# =============================================================================
class MHExplosionError(Exception):
    """Raised when MD energy becomes unphysical (E > 0 eV)."""
    pass


class AdjustedMinimaHopping(MinimaHopping):
    def __call__(self, totalsteps=None, maxtemp=None, max_optimize_steps=None):
        self.max_optimize_steps = max_optimize_steps if max_optimize_steps else 500
        self._startup()
        while True:
            if (totalsteps and self._counter >= totalsteps):
                self._log('msg', f'Run terminated. Step #{self._counter} reached.')
                return
            if (maxtemp and self._temperature >= maxtemp):
                self._log('msg', f'Run terminated. Max temp {maxtemp} K reached.')
                return
            self._previous_optimum = self._atoms.copy()
            self._previous_energy = self._atoms.get_potential_energy()
            try:
                self._molecular_dynamics()
                self._optimize()
            except MHExplosionError as e:
                self._log('msg', f'MD Explosion at step {self._counter}, terminating MH gracefully. '
                                 f'Minima found before explosion will be collected.')
                break
            self._counter += 1
            self._check_results()

    def _optimize(self):
        self._atoms.set_momenta(np.zeros(self._atoms.get_momenta().shape))
        with self._optimizer(self._atoms, trajectory=f'qn{self._counter:05d}.traj', logfile=f'qn{self._counter:05d}.log') as opt:
            opt.run(fmax=self._fmax, steps=self.max_optimize_steps)
            self._log('ene')

    def _molecular_dynamics(self, resume=None):
        self._log('msg', 'Molecular dynamics: md%05i' % self._counter)
        mincount = 0
        energies, oldpositions = [], []
        thermalized = False

        if resume:
            self._log('msg', 'Resuming MD from md%05i.traj' % resume)
            if os.path.getsize('md%05i.traj' % resume) == 0:
                self._log('msg', 'md%05i.traj is empty. Resuming from '
                          'qn%05i.traj.' % (resume, resume - 1))
                atoms = ase.io.read('qn%05i.traj' % (resume - 1), index=-1)
            else:
                with ase.io.Trajectory('md%05i.traj' % resume, 'r') as images:
                    for atoms in images:
                        energies.append(atoms.get_potential_energy())
                        oldpositions.append(atoms.positions.copy())
                        passedmin = self._passedminimum(energies)
                        if passedmin:
                            mincount += 1
                self._atoms.set_momenta(atoms.get_momenta())
                thermalized = True
            self._atoms.positions = atoms.get_positions()
            self._log('msg', 'Starting MD with %i existing energies.' % len(energies))

        if not thermalized:
            MaxwellBoltzmannDistribution(self._atoms,
                                         temperature_K=self._temperature,
                                         force_temp=True)

        traj = ase.io.Trajectory('md%05i.traj' % self._counter, 'a', self._atoms)
        dyn = VelocityVerlet(self._atoms, timestep=self._timestep * units.fs)
        log = MDLogger(dyn, self._atoms, 'md%05i.log' % self._counter,
                       header=True, stress=False, peratom=False)

        with traj, dyn, log:
            dyn.attach(log, interval=1)
            dyn.attach(traj, interval=1)
            while mincount < self._mdmin:
                dyn.run(1)

                current_energy = self._atoms.get_potential_energy()

                if current_energy > 0.0:
                    error_message = f"MD Explosion! Energy soared to {current_energy:.2f} eV during Step {self._counter}."
                    self._log('msg', error_message)
                    raise MHExplosionError(error_message)

                energies.append(current_energy)
                passedmin = self._passedminimum(energies)
                if passedmin:
                    mincount += 1
                oldpositions.append(self._atoms.positions.copy())

            self._atoms.positions = oldpositions[passedmin[0]]


# =============================================================================
# 3. MACEGlobalOptimizer (Single-thread Global Search with Constraints)
# =============================================================================
class MACEGlobalOptimizer:
    def __init__(self, model_paths, device="cuda" if torch.cuda.is_available() else "cpu", default_dtype="float64", enable_cueq=True, dispersion=False):
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        self.calc = MACECalculator(model_paths=model_paths, device=device, default_dtype=default_dtype, enable_cueq=enable_cueq, dispersion=dispersion)
        self.constraints = []
        self.hookean_constraints = []
        self.structure = None
        self.free_idx = []

    def load_structure(self, filename):
        self.structure = ase.io.read(filename)
        struct = Structure.from_file(filename)
        constraint = []
        for idx in range(struct.num_sites):
            if 'selective_dynamics' in struct.sites[idx].properties.keys():
                if not struct.sites[idx].properties['selective_dynamics'][0]:
                    constraint.append(idx)
        self.constraints = [FixAtoms(indices=constraint)]
        self.structure.calc = self.calc

    def set_constraint_by_tagfile(self, tagfile='TAG', fixed_tags=[0]):
        tags = [0] * len(self.structure)
        constraint = []
        with open(tagfile, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                tags[int(line.split()[0]) - 1] = int(line.split()[1])
        for idx in range(len(self.structure)):
            if tags[idx] in fixed_tags:
                constraint.append(idx)
            else:
                self.free_idx.append(idx)
        self.constraints = [FixAtoms(indices=constraint)]
        print(f"Applied fixed constraints from {tagfile}. Fixed atoms: {len(constraint)}")

    def set_hookean_constraint_by_tagfile(self, tagfile='TAG', adsorbate_tag=2):
        """Current: C-H, C-C, C-O, O-H"""
        adsorbate_atom_idx = []
        with open(tagfile, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                if int(line.split()[1]) == adsorbate_tag:
                    adsorbate_atom_idx.append(int(line.split()[0]) - 1)

        added_count = 0
        for i, idx in enumerate(adsorbate_atom_idx[:-1]):
            i_atom = self.structure[idx]
            for jdx in adsorbate_atom_idx[i + 1:]:
                j_atom = self.structure[jdx]

                symbol = frozenset([i_atom.symbol, j_atom.symbol])
                dist = np.linalg.norm(i_atom.position - j_atom.position)
                if symbol not in AUTO_HOOKEAN_DIST.keys():
                    continue
                if dist <= AUTO_HOOKEAN_DIST[symbol][0]:
                    self.hookean_constraints.append(Hookean(idx, jdx, 20.0, AUTO_HOOKEAN_DIST[symbol][1]))
                    added_count += 1

        print(f"Applied {added_count} Hookean constraints based on {tagfile}.")

    def load_hookean_constraints(self, filename='HOOKEAN'):
        with open(filename, 'r') as file:
            content = file.read()
        # Extract non-empty lines to check the format signature
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        added_count = 0

        # Version Routing: New format starts with '#' or text keywords
        is_new_format = lines[0].startswith('#') or lines[0].startswith('Z_MAX') or lines[0].startswith('ATOMS')

        if is_new_format:
            # ==========================================
            # Parser for the NEW format
            # ==========================================
            z_max = None
            spring_k = None
            atoms = []

            for line in lines:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'Z_MAX':
                    z_max = float(parts[1])
                elif parts[0] == 'SPRING_K':
                    spring_k = float(parts[1])
                elif parts[0] == 'ATOMS':
                    # Convert 1-based indexing from file to 0-based for Python/ASE
                    atoms = [int(x) - 1 for x in parts[1:]]

            if z_max is not None and spring_k is not None and atoms:
                # Map Z_MAX to ASE's plane equation [A, B, C, D] where Ax + By + Cz + D = 0
                plane_a2 = [0.0, 0.0, 1.0, -z_max]

                # Apply the constraint to each adsorbate atom individually
                for atom_idx in atoms:
                    self.hookean_constraints.append(Hookean(atom_idx, plane_a2, k=spring_k))
                    added_count += 1

        else:
            # ==========================================
            # Parser for the OLD format (Original Logic)
            # ==========================================
            contents = content.strip().split('\n\n')

            for block in contents:
                if not block.strip():
                    continue
                blocks = block.split('\n')

                if len(blocks) == 3:
                    a1 = int(blocks[0]) - 1
                    k = float(blocks[2])
                    a2_line = blocks[1].split()
                    if len(a2_line) == 1:
                        a2 = int(a2_line[0]) - 1
                    elif len(a2_line) in [3, 4]:
                        a2 = [float(v) for v in a2_line]

                    self.hookean_constraints.append(Hookean(a1, a2, k))
                    added_count += 1

                elif len(blocks) == 4:
                    a1 = int(blocks[0]) - 1
                    k = float(blocks[2])
                    rt = float(blocks[3])
                    a2_line = blocks[1].split()
                    if len(a2_line) == 1:
                        a2 = int(a2_line[0]) - 1
                    elif len(a2_line) in [3, 4]:
                        a2 = [float(v) for v in a2_line]

                    self.hookean_constraints.append(Hookean(a1, a2, k, rt=rt))
                    added_count += 1

        print(f"Loaded {added_count} external Hookean constraints from {filename}.")

    def auto_tether_surface_atoms(self, tagfile='TAG', adsorbate_tag=2, rest_radius=0.3, spring_constant=5.0, dz=1.0):
        tags = [0] * len(self.structure)
        if os.path.exists(tagfile):
            with open(tagfile, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.split()
                        tags[int(parts[0]) - 1] = int(parts[1])
        else:
            print("Warning: TAG file not found. Assuming all atoms are substrate.")

        substrate_indices = [i for i, tag in enumerate(tags) if tag != adsorbate_tag]
        if not substrate_indices:
            print("Error: No substrate atoms identified. Check tag matching.")
            return

        z_coords = [self.structure.get_positions()[i][2] for i in substrate_indices]
        z_max = max(z_coords)
        z_min = z_max - dz

        tethers = []
        tethered_count = 0
        for idx in substrate_indices:
            pos = self.structure.get_positions()[idx]
            if z_min <= pos[2] <= (z_max + 0.1):
                tethers.append(Hookean(a1=idx, a2=(pos[0], pos[1], pos[2]), k=spring_constant, rt=rest_radius))
                tethered_count += 1

        self.hookean_constraints.extend(tethers)

        print(f"Auto-tethered {tethered_count} atoms in Z range [{z_min:.2f}, {z_max:.2f}].")
        print(f"Rest radius (free movement): {rest_radius} A")
        print(f"Spring constant (restoring force): {spring_constant} eV/A^2.")

    def run_minima_hopping(self,
                           totalsteps=10,
                           max_optimize_steps=500,
                           unconstrained_fmax_threshold=0.05,
                           optimizer_type='FIRE2',
                           **kwargs):
        if os.path.exists('hop.log'):
            os.remove('hop.log')
        if os.path.exists('minima.traj'):
            os.remove('minima.traj')

        patterns = [re.compile(r'^md\d{5}\.log$'), re.compile(r'^md\d{5}\.traj$'),
                    re.compile(r'^qn\d{5}\.log$'), re.compile(r'^qn\d{5}\.traj$')]
        for filename in os.listdir('.'):
            if os.path.isfile(filename):
                for pattern in patterns:
                    if pattern.match(filename):
                        os.remove(filename)

        # Apply all accumulated constraints right before running
        self.structure.set_constraint(self.constraints + self.hookean_constraints)
        opt_class = BFGS if optimizer_type.upper() == 'BFGS' else FIRE2

        default_mh_kwargs = {
            'T0': 2000.0, 'beta1': 1.15, 'beta2': 1.15, 'beta3': 1.0 / 1.1,
            'Ediff0': 2.0, 'alpha1': 0.98, 'alpha2': 1.0 / 0.98, 'mdmin': 2,
            'minima_threshold': 0.8, 'timestep': 1.0, 'fmax': 0.02,
            'logfile': 'hop.log', 'minima_traj': 'minima.traj'
        }
        default_mh_kwargs.update(kwargs)

        amh = AdjustedMinimaHopping(atoms=self.structure, optimizer=opt_class, **default_mh_kwargs)
        amh(totalsteps=totalsteps, max_optimize_steps=max_optimize_steps)

        # 1. Add a brief pause to ensure Lustre filesystem buffers are fully synced
        time.sleep(2.0)

        traj_file = default_mh_kwargs['minima_traj']

        # 2. Safety check for empty or missing files
        if not os.path.exists(traj_file) or os.path.getsize(traj_file) == 0:
            print(f"Warning: {traj_file} is empty or missing. No accepted hops during this run.")
            return

        minima_list = []
        rejected_count = 0
        # 3. Use 'with' context manager to automatically close the file stream

        with Trajectory(traj_file, 'r') as trajectory:
            for atoms_traj in trajectory:
                atoms = atoms_traj.copy()
                atoms.set_constraint([])
                atoms.calc = self.structure.calc

                forces = atoms.get_forces()
                fmax_actual = np.max(np.linalg.norm(forces[self.free_idx], axis=1))

                if fmax_actual <= unconstrained_fmax_threshold:
                    pure_energy = atoms.get_potential_energy()
                    atoms.set_constraint(self.constraints)
                    minima_list.append([atoms, pure_energy])
                else:
                    rejected_count += 1

        if rejected_count > 0:
            print(f"Filtered out {rejected_count} unphysical structures (unconstrained fmax > {unconstrained_fmax_threshold} eV/A).")

        if not minima_list:
            print("No valid minima extracted from the trajectory.")
            return

        minima_list = sorted(minima_list, key=lambda x: x[1])

        with open('MINIMA_EE', 'w') as f:
            for idx, minima in enumerate(minima_list):
                write_vasp(f'MINIMA{idx}.vasp', minima[0])
                f.write(f'{minima[1]:.6f}\n')


# =============================================================================
# 4. MultiGlobalOptimizer (Multi-process Orchestrator)
# =============================================================================
def _worker_task(task_id, params, model_path, initial_structure_path):
    """Top-level picklable function for multiprocessing spawn pool."""
    print(f"[Worker {task_id}] Strategy: {params.get('strategy', 'custom')}")
    work_dir = f"mh_task_{task_id}"
    os.makedirs(work_dir, exist_ok=True)
    current_dir = os.getcwd()

    # 1. Copy required files to the worker's isolated directory
    files_to_copy = [initial_structure_path, 'TAG', 'HOOKEAN']
    for file in files_to_copy:
        src_path = os.path.join(current_dir, file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(work_dir, file))

    _model_path = os.path.realpath(model_path)
    os.chdir(work_dir)

    try:
        optimizer = MACEGlobalOptimizer(model_paths=_model_path)
        # Assuming the structure file was copied directly into the folder
        local_struct_name = os.path.basename(initial_structure_path)
        optimizer.load_structure(local_struct_name)

        # 2. Re-apply constraints inside the worker process
        if os.path.exists('TAG'):
            optimizer.set_constraint_by_tagfile(tagfile='TAG')
            if params.get('use_auto_hookean', False):
                optimizer.set_hookean_constraint_by_tagfile(tagfile='TAG', adsorbate_tag=2)
            if params.get('use_auto_tether', False):
                optimizer.auto_tether_surface_atoms(tagfile='TAG')

        if os.path.exists('HOOKEAN'):
            optimizer.load_hookean_constraints(filename='HOOKEAN')

        # 3. Filter out non-MH parameters and run
        mh_kwargs = {k: v for k, v in params.items() if k not in ['strategy', 'totalsteps', 'use_auto_tether', 'use_auto_hookean']}
        optimizer.run_minima_hopping(totalsteps=params.get('totalsteps', 50), **mh_kwargs)
        print(f"[Worker {task_id}] Completed successfully.")
    except MHExplosionError as e:
        print(f"[Worker {task_id}] MD explosion at worker, minima before explosion retained: {e}")
    except Exception as e:
        print(f"[Worker {task_id}] Failed: {e}")
    finally:
        os.chdir(current_dir)


class MultiGlobalOptimizer:
    def __init__(self, model_path, initial_structure_path):
        self.model_path = model_path
        self.initial_structure_path = initial_structure_path
        self.strategy_grid = []

    def add_strategy(self, strategy_name, totalsteps=50, max_optimize_steps=500, use_auto_tether=False, use_auto_hookean=False, **kwargs):
        kwargs['strategy'] = strategy_name
        kwargs['totalsteps'] = totalsteps
        kwargs['max_optimize_steps'] = max_optimize_steps
        kwargs['use_auto_tether'] = use_auto_tether
        kwargs['use_auto_hookean'] = use_auto_hookean
        self.strategy_grid.append(kwargs)

    def _get_dynamic_worker_count(self, memory_per_worker_gb=8.0):
        if not torch.cuda.is_available():
            print("CUDA unavailable. Falling back to CPU cores.")
            return max(1, mp.cpu_count() - 2)

        free_mem, total_mem = torch.cuda.mem_get_info()
        free_mem_gb = free_mem / (1024 ** 3)
        total_mem_gb = total_mem / (1024 ** 3)

        max_workers = int(free_mem_gb // memory_per_worker_gb)
        max_workers = max(1, min(max_workers, mp.cpu_count() - 2))

        print(f"GPU Memory: {free_mem_gb:.1f} GB free / {total_mem_gb:.1f} GB total.")
        print(f"Allocating {max_workers} concurrent workers (assuming {memory_per_worker_gb} GB/worker).")
        return max_workers

    def run_parallel_exploration(self, manual_workers=None, memory_per_worker_gb=8.0):
        mp.set_start_method('spawn', force=True)
        num_workers = manual_workers if manual_workers else self._get_dynamic_worker_count(memory_per_worker_gb)

        active_processes = []
        task_queue = list(enumerate(self.strategy_grid))

        while task_queue or active_processes:
            active_processes = [p for p in active_processes if p.is_alive()]
            while len(active_processes) < num_workers and task_queue:
                task_id, params = task_queue.pop(0)
                p = mp.Process(target=_worker_task, args=(task_id, params, self.model_path, self.initial_structure_path))
                p.start()
                active_processes.append(p)
                time.sleep(2)
            time.sleep(5)

        print("All parallel search tasks have finished. Starting aggregation...")
        self.aggregate_and_sort_minima()

    def aggregate_and_sort_minima(self, adsorbate_tag=2, energy_tol=0.05, rel_dev_tol=0.08):
        """
        Aggregates results from all workers, groups them using Graph Theory based on
        thermodynamic, topological, and spatial equivalence, and outputs unique configurations.
        """
        print("Aggregating and clustering results across all workers...")
        output_dir = "GLOBAL_MINIMA_RESULTS"
        tagfile = "TAG"

        if not os.path.exists(tagfile):
            print(f"Warning: Missing {tagfile} file. Cannot identify adsorbate for clustering.")
            return

        adsorbate_indices = []
        with open(tagfile, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and int(line.split()[1]) == adsorbate_tag:
                    adsorbate_indices.append(int(line.split()[0]) - 1)

        raw_minima = []
        task_folders = [f for f in os.listdir('.') if f.startswith("mh_task_") and os.path.isdir(f)]

        for task_folder in sorted(task_folders):
            ee_file = os.path.join(task_folder, "MINIMA_EE")
            if not os.path.exists(ee_file):
                continue
            with open(ee_file, 'r', encoding='utf-8') as f:
                energies = [float(line.strip()) for line in f.readlines()]
            for idx, energy in enumerate(energies):
                vasp_path = os.path.join(task_folder, f"MINIMA{idx}.vasp")
                if os.path.exists(vasp_path):
                    raw_minima.append({
                        "id": f"{task_folder}_min{idx}", "energy": energy,
                        "atoms": ase.io.read(vasp_path), "source": task_folder
                    })

        if not raw_minima:
            print("No valid VASP minima found in worker directories.")
            return

        sample_atoms = raw_minima[0]["atoms"]
        substrate_indices = [i for i in range(len(sample_atoms)) if i not in adsorbate_indices]

        # Graph Theory Clustering using the 4-step physical equivalence logic
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

        strategy_coverage = {folder: [] for folder in task_folders}
        for new_idx, group in enumerate(unique_groups):
            for source in group["sources"]:
                strategy_coverage[source].append(new_idx)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        out_ee_path = os.path.join(output_dir, "GLOBAL_MINIMA_EE")
        with open(out_ee_path, 'w', encoding='utf-8') as f_out:
            for new_idx, group in enumerate(unique_groups):
                rep = group["representative"]
                sources_str = ",".join(sorted(group["sources"]))

                out_vasp = os.path.join(output_dir, f"GLOBAL_MINIMA{new_idx}.vasp")
                ase.io.write(out_vasp, rep["atoms"], format="vasp")

                f_out.write(f"{rep['energy']:.6f}  # Rep: {rep['id']} | Size: {group['members_count']} | Found by: [{sources_str}]\n")

        coverage_file = os.path.join(output_dir, "STRATEGY_COVERAGE.txt")
        with open(coverage_file, 'w', encoding='utf-8') as f_cov:
            f_cov.write("==================================================\n")
            f_cov.write("MH Strategy Coverage Report\n")
            f_cov.write("==================================================\n")
            f_cov.write(f"Total Unique Minima Found: {len(unique_groups)}\n\n")

            for task_folder in sorted(task_folders):
                covered_indices = strategy_coverage[task_folder]
                hit_rate = len(covered_indices) / len(unique_groups) * 100 if unique_groups else 0

                f_cov.write(f"Strategy: {task_folder}\n")
                f_cov.write(f"  Coverage: {len(covered_indices)}/{len(unique_groups)} ({hit_rate:.1f}%)\n")
                f_cov.write(f"  Captured UNIQUE IDs: {covered_indices}\n\n")

        print(f"Aggregation complete. Found {len(unique_groups)} unique states.")
        print(f"Results and coverage report saved in '{output_dir}'.")
