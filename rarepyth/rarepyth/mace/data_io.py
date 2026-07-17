# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:43:54 2026

@author: Wang Junhao
"""

import os
import math
import random

import ase.io
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm

from rarepyth.utils import make_target_dir


def _read_and_standardize_xyz(file_path):
    """
    Read an extxyz file and standardize its energy and forces properties
    to ensure compatibility with MACE.
    """
    atoms_list = ase.io.read(file_path, index=':')
    standardized_list = []

    energy_keys = ['energy', 'REF_energy', 'Energy', 'dft_energy']
    force_keys = ['forces', 'REF_forces', 'Forces', 'dft_forces']
    stress_keys = ['stress', 'REF_stress', 'virial']

    for atoms in atoms_list:
        energy = None
        forces = None
        stress = None

        # 1. Try to extract from existing calculator
        if atoms.calc is not None:
            try:
                energy = atoms.get_potential_energy()
            except Exception:
                pass
            try:
                forces = atoms.get_forces()
            except Exception:
                pass
            try:
                stress = atoms.get_stress()
            except Exception:
                pass

        # 2. Extract from info/arrays if calculator failed or was absent
        if energy is None:
            for k in energy_keys:
                if k in atoms.info:
                    energy = float(atoms.info[k])
                    break

        if forces is None:
            for k in force_keys:
                if k in atoms.arrays:
                    forces = atoms.arrays[k]
                    break

        if stress is None:
            for k in stress_keys:
                if k in atoms.info:
                    stress = atoms.info[k]
                    break

        # 3. Clean up old non-standard keys to prevent duplication in extxyz
        for k in energy_keys:
            atoms.info.pop(k, None)
        for k in force_keys:
            if k in atoms.arrays:
                del atoms.arrays[k]
        for k in stress_keys:
            atoms.info.pop(k, None)

        # 4. Bind standard calculator
        calc_kwargs = {}
        if energy is not None:
            calc_kwargs['energy'] = energy
        if forces is not None:
            calc_kwargs['forces'] = forces
        if stress is not None:
            calc_kwargs['stress'] = stress

        if calc_kwargs:
            atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)

        standardized_list.append(atoms)

    return standardized_list


def build_from_vasp(
    vaspjob_dir,
    output_dir=None,
    output_name=None,
    tags=None,
    sampling_interval=0,
    maximum_sample=None,
    readfile='OUTCAR'
):
    vaspjob_dir = os.path.realpath(vaspjob_dir)
    output_dir = os.path.realpath(output_dir) if output_dir else os.path.realpath('.')

    if readfile == 'OUTCAR':
        atoms_list = ase.io.read(os.path.join(vaspjob_dir, 'OUTCAR'), index=':', format='vasp-out')

        if sampling_interval:
            sampled_atoms_list = [0] * math.ceil(len(atoms_list) / (sampling_interval + 1))
            for idx in range(len(sampled_atoms_list)):
                sampled_atoms_list[-idx - 1] = atoms_list[-1 - (sampling_interval + 1) * idx]
            atoms_list = sampled_atoms_list

        if maximum_sample and len(atoms_list) > maximum_sample:
            atoms_list = atoms_list[:maximum_sample]
    else:
        atoms_list = [ase.io.read(os.path.join(vaspjob_dir, readfile))]

    if not output_name:
        output_name = f"{os.path.basename(vaspjob_dir)}.xyz"
    output_path = os.path.join(output_dir, output_name)

    if tags is not None:
        tags_array = np.array(tags, dtype=int)

    valid_atoms = []
    for atoms in tqdm(atoms_list, desc="Processing VASP output"):
        if tags is not None:
            if len(tags) == len(atoms):
                atoms.arrays['tags'] = tags_array
            else:
                atoms.info['tags'] = tags_array
        valid_atoms.append(atoms)

    if valid_atoms:
        ase.io.write(output_path, valid_atoms, format='extxyz')


def merge_xyz(
    input_paths,
    output_dir,
    output_name=None
):
    output_dir = os.path.realpath(output_dir)
    make_target_dir(output_dir)
    merged_atoms = []

    if isinstance(input_paths, str) and os.path.isdir(input_paths):
        input_dir = os.path.realpath(input_paths)
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.xyz'):
                    file_path = os.path.join(root, file)
                    merged_atoms.extend(_read_and_standardize_xyz(file_path))
        if not output_name:
            output_name = f"{os.path.basename(input_dir)}_merged.xyz"

    elif isinstance(input_paths, (list, tuple)):
        for file in input_paths:
            file_path = os.path.realpath(file)
            if file_path.endswith('.xyz'):
                merged_atoms.extend(_read_and_standardize_xyz(file_path))
        if not output_name:
            output_name = f"{os.path.splitext(os.path.basename(input_paths[0]))[0]}_merged.xyz"

    else:
        raise ValueError("input_paths must be a directory path or a list of file paths.")

    db_out_path = os.path.join(output_dir, output_name)
    ase.io.write(db_out_path, merged_atoms, format='extxyz')


def split_xyz(
    input_path,
    output_dir,
    method='ratio',
    value=0.8,
    seed=0
):
    input_path = os.path.realpath(input_path)
    output_dir = os.path.realpath(output_dir)
    make_target_dir(output_dir)

    output_prefix = os.path.basename(input_path)[:-4] if input_path.endswith('.xyz') else os.path.basename(input_path)

    # Use the standardization function when reading
    atoms_list = _read_and_standardize_xyz(input_path)

    random.seed(seed)
    random.shuffle(atoms_list)
    _list = []

    if method == 'ratio':
        idx_split = int(value * len(atoms_list))

        file_path_0 = os.path.join(output_dir, f"{output_prefix}_0.xyz")
        ase.io.write(file_path_0, atoms_list[:idx_split], format='extxyz')
        _list.append(file_path_0)

        file_path_1 = os.path.join(output_dir, f"{output_prefix}_1.xyz")
        ase.io.write(file_path_1, atoms_list[idx_split:], format='extxyz')
        _list.append(file_path_1)

    elif method == 'length':
        length = int(value)
        total_pieces = math.ceil(len(atoms_list) / length)
        for piece in range(total_pieces):
            file_path = os.path.join(output_dir, f"{output_prefix}_{piece}.xyz")
            _list.append(file_path)
            start_idx = piece * length
            end_idx = min((piece + 1) * length, len(atoms_list))
            ase.io.write(file_path, atoms_list[start_idx:end_idx], format='extxyz')

    elif method == 'folds':
        folds = int(value)
        fold_size = len(atoms_list) / folds
        for k in range(folds):
            file_path = os.path.join(output_dir, f"{output_prefix}_{k}.xyz")
            _list.append(file_path)
            start_idx = int(k * fold_size)
            end_idx = int((k + 1) * fold_size)
            ase.io.write(file_path, atoms_list[start_idx:end_idx], format='extxyz')

    else:
        raise ValueError("Method must be 'ratio', 'length', or 'folds'.")

    return _list
