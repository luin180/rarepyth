# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:47:14 2024

@author: Wang Junhao
"""

import itertools
import numpy as np
from ase.io import read
from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from rarepyth.utils import get_nearest_atoms

Adsorbates = {
    'H': Molecule('H', [[0, 0, 0]]),
    'CO': Molecule('CO', [[0, 0, 0], [0, 0, 1.17]]),
    'CO2': Molecule('COO', [[0, 0, 0], [0, 1.19, 0.38], [0, -1.19, 0.38]]),
    'COOH': Molecule('COOH', [[0, 0, 0], [0, 1.19, 0.38], [0, -1.19, 0.38], [0, -1.19, 1.36]]),
    'OCHO': Molecule('COOH', [[0, 0, 0.38], [0, 1.19, 0], [0, -1.19, 0], [0, 0, 1.49]]),
    'H2O': Molecule('OHH', [[0, 0, 0], [0, 0.78, 0.57], [0, -0.78, 0.57]]),
    'OH': Molecule('OH', [[0, 0, 0], [0, 0, 0.98]]),
    'NH3': Molecule('NHHH', [[0, 0, 0], [0.5, 0.83, 0.38], [-1.0, 0.0, 0.38], [0.5, -0.83, 1.36]]),
}

Adsorption_avail_elements = {'Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca',
                             'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                             'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru',
                             'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'La',
                             'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                             'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                             'Au', 'Hg', 'Tl', 'Pb', 'Bi'}


def slab_from_file(filename):
    from pymatgen.core import Structure
    from pymatgen.core.surface import SlabGenerator
    from rarepyth.slab.builder import adjust_vacuum_layer

    structure = Structure.from_file(filename)
    gen = SlabGenerator(structure, (0, 0, 1), 1, 0, in_unit_planes=True)
    return adjust_vacuum_layer(gen.get_slab(), 0)


def find_filtered_adsorption_sites(slab, filtered_atoms=Adsorption_avail_elements, tol=0.01, find_args=None):
    find_args = find_args or {}
    ASF_slab = AdsorbateSiteFinder(slab)
    ads_sites = ASF_slab.find_adsorption_sites(**find_args)
    filtered_sites = []
    for site in ads_sites['all']:
        for atom in get_nearest_atoms(site, slab, coords_are_cartesian=True, tol=tol):
            if atom.label in filtered_atoms:
                filtered_sites.append(site)
                break
    return filtered_sites


def generate_adsorption_structures_by_sites(slab, adsorbate, sites, repeat=None, min_lw=10.0, translate=True, reorient=True):
    if repeat is None:
        xrep = np.ceil(min_lw / np.linalg.norm(slab.lattice.matrix[0]))
        yrep = np.ceil(min_lw / np.linalg.norm(slab.lattice.matrix[1]))
        repeat = [xrep, yrep, 1]

    ASF_slab = AdsorbateSiteFinder(slab)
    sites_list = sites['all'] if isinstance(sites, dict) else sites
    return [ASF_slab.add_adsorbate(
        adsorbate, coords, repeat=repeat, translate=translate, reorient=reorient
    ) for coords in sites_list]


def load_molecule_from_file(filename: str) -> Molecule:
    atoms = read(filename)
    symbols = atoms.get_chemical_symbols()
    coords = atoms.get_positions()
    return Molecule(symbols, coords)


def align_molecule_z_to_zero(mol: Molecule) -> Molecule:
    coords = mol.cart_coords.copy()
    min_z = np.min(coords[:, 2])
    coords[:, 2] -= min_z
    return Molecule(mol.species, coords)


def generate_rotated_molecules(mol: Molecule, step_degrees: float = 45.0) -> list[Molecule]:
    coords = mol.cart_coords
    geometric_center = np.mean(coords, axis=0)
    centered_coords = coords - geometric_center

    angles = np.arange(0, 360, step_degrees)
    euler_combinations = list(itertools.product(angles, angles, angles))
    rotated_molecules = []

    for euler_angles in euler_combinations:
        rot = Rotation.from_euler('xyz', euler_angles, degrees=True)
        rotated_coords = rot.apply(centered_coords)
        rotated_mol = Molecule(mol.species, rotated_coords)
        aligned_mol = align_molecule_z_to_zero(rotated_mol)
        rotated_molecules.append(aligned_mol)

    return rotated_molecules


def get_permutation_invariant_rmsd(mol1: Molecule, mol2: Molecule) -> float:
    total_sq_diff = 0.0
    elements = set(site.species_string for site in mol1)

    for el in elements:
        coords1 = np.array([site.coords for site in mol1 if site.species_string == el])
        coords2 = np.array([site.coords for site in mol2 if site.species_string == el])
        dist_matrix = cdist(coords1, coords2, metric='sqeuclidean')
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        total_sq_diff += np.sum(dist_matrix[row_ind, col_ind])

    return np.sqrt(total_sq_diff / len(mol1))


def remove_duplicate_molecules_robust(molecules: list[Molecule], rmsd_tol: float = 0.1) -> list[Molecule]:
    unique_mols = []
    for new_mol in molecules:
        is_duplicate = False
        for exist_mol in unique_mols:
            rmsd = get_permutation_invariant_rmsd(new_mol, exist_mol)
            if rmsd < rmsd_tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_mols.append(new_mol)
    return unique_mols


if __name__ == "__main__":
    from pymatgen.core import Structure
    slab = Structure.from_file('structure.vasp')
    adsorbate = Adsorbates['CO']
    sites = find_filtered_adsorption_sites(
        slab, 'Cu', find_args={'positions': 'ontop'}
    )
    generate_adsorption_structures_by_sites(
        slab, adsorbate, sites
    )[0].to('ads.vasp', fmt='poscar')
