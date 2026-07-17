# -*- coding: utf-8 -*-
"""
Ultimate thermodynamic and topological equivalence evaluator.
- Chemisorption: Evaluated via 1.25x continuous binding profile.
- Physisorption: Evaluated via Z-axis projection density (Z-Profile), perfectly immune to XY translation/rotation.
"""

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.data import covalent_radii


def get_mic_distance_matrix(p1: np.ndarray, p2: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    diff = p1[:, np.newaxis, :] - p2[np.newaxis, :, :]
    diff_flat = diff.reshape(-1, 3)
    mic_diff_flat, _ = find_mic(diff_flat, cell, pbc)
    return np.linalg.norm(mic_diff_flat.reshape(p1.shape[0], p2.shape[0], 3), axis=-1)


def detect_hydrogen_bonds(
    atoms: Atoms,
    ads_idx: list[int],
    sub_idx: list[int],
    donor_elements: tuple[str, ...] = ('O', 'N', 'F', 'C'),
    acceptor_elements: tuple[str, ...] = ('O', 'N', 'F'),
    max_h_bond_dist: float = 2.5,
    min_angle_deg: float = 130.0
) -> dict[str, list[float]]:
    symbols = np.array(atoms.get_chemical_symbols())
    positions, cell, pbc = atoms.positions, atoms.cell.array, atoms.pbc
    radii = np.array([covalent_radii[z] for z in atoms.numbers])

    h_indices = [i for i in ads_idx if symbols[i] == 'H']
    d_indices = [i for i in ads_idx + sub_idx if symbols[i] in donor_elements]
    a_indices = [i for i in sub_idx if symbols[i] in acceptor_elements]

    hb_profile = {}
    if not h_indices or not d_indices or not a_indices:
        return hb_profile

    cos_theta_min = np.cos(np.radians(min_angle_deg))

    for h_idx in h_indices:
        h_pos = positions[h_idx]

        # Donor Check
        d_diff = positions[d_indices] - h_pos
        d_mic, _ = find_mic(d_diff, cell, pbc)
        d_dists = np.linalg.norm(d_mic, axis=1)
        if len(d_dists) == 0 or np.min(d_dists) > 1.2:
            continue
        vec_dh = -d_mic[np.argmin(d_dists)]

        # Acceptor Check
        a_diff = positions[a_indices] - h_pos
        a_mic, _ = find_mic(a_diff, cell, pbc)
        a_dists = np.linalg.norm(a_mic, axis=1)

        valid_a_mask = (a_dists > 0.5) & (a_dists <= max_h_bond_dist)
        for i, a_idx in enumerate(np.array(a_indices)[valid_a_mask]):
            vec_ha = a_mic[valid_a_mask][i]
            norm_product = np.linalg.norm(vec_dh) * a_dists[valid_a_mask][i]

            if norm_product > 1e-8 and (np.dot(vec_dh, vec_ha) / norm_product) >= cos_theta_min:
                norm_dist = a_dists[valid_a_mask][i] / (radii[h_idx] + radii[a_idx])
                key = f"HBond-{symbols[a_idx]}"
                hb_profile.setdefault(key, []).append(norm_dist)

    return hb_profile


def extract_signatures(
    atoms: Atoms,
    ads_idx: list[int],
    sub_idx: list[int],
    intra_tol: float = 1.4,
    chem_tol: float = 1.25
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Extracts:
    1. Internal Adsorbate Graph
    2. Chemisorption Profile (<= 1.25x cutoff + H-Bonds)
    3. Z-Profile (Heights of adsorbate atoms relative to the surface plane)
    """
    radii = np.array([covalent_radii[z] for z in atoms.numbers])
    symbols = np.array(atoms.symbols)

    ads_pos, ads_radii, ads_sym = atoms.positions[ads_idx], radii[ads_idx], symbols[ads_idx]
    sub_pos, sub_radii, sub_sym = atoms.positions[sub_idx], radii[sub_idx], symbols[sub_idx]

    # 1. Internal Topology
    radii_sum_ads = ads_radii[:, np.newaxis] + ads_radii[np.newaxis, :]
    dist_ads = get_mic_distance_matrix(ads_pos, ads_pos, atoms.cell.array, atoms.pbc)
    adj_matrix = (dist_ads < (radii_sum_ads * intra_tol)).astype(int)
    np.fill_diagonal(adj_matrix, 0)

    # 2. Chemisorption Profile
    radii_sum_inter = ads_radii[:, np.newaxis] + sub_radii[np.newaxis, :]
    dist_inter = get_mic_distance_matrix(ads_pos, sub_pos, atoms.cell.array, atoms.pbc)
    norm_dist = dist_inter / radii_sum_inter

    chem_profile = {}
    for i, a_sym in enumerate(ads_sym):
        for j, s_sym in enumerate(sub_sym):
            nd = norm_dist[i, j]
            if nd <= chem_tol:
                pair_key = f"{a_sym}-{s_sym}"
                chem_profile.setdefault(pair_key, []).append(nd)

    hb_profile = detect_hydrogen_bonds(atoms, ads_idx, sub_idx)
    for k, v in hb_profile.items():
        chem_profile.setdefault(k, []).extend(v)

    for k in chem_profile:
        chem_profile[k] = np.sort(chem_profile[k])

    # 3. Physisorption Z-Profile (Height distribution)
    # Estimate the top surface layer by taking the 90th percentile of Z coordinates
    # to be robust against slight surface corrugation/relaxation.
    top_surface_z = np.percentile(sub_pos[:, 2], 90)
    z_profile = np.sort(ads_pos[:, 2] - top_surface_z)

    return adj_matrix, chem_profile, z_profile


def compare_chem_profiles(prof1: dict[str, np.ndarray], prof2: dict[str, np.ndarray], dev_tol: float, cutoff: float) -> bool:
    """Compares chemisorption distance profiles using relative deviation."""
    for key in set(prof1.keys()).union(set(prof2.keys())):
        d1 = prof1.get(key, np.array([]))
        d2 = prof2.get(key, np.array([]))
        min_l = min(len(d1), len(d2))

        if min_l > 0:
            rel_diff = np.abs(d1[:min_l] - d2[:min_l]) / np.maximum(np.minimum(d1[:min_l], d2[:min_l]), 1e-5)
            if np.any(rel_diff > dev_tol):
                return False

        safe_lower_bound = cutoff * (1.0 - dev_tol)
        if len(d1) > min_l and np.any(d1[min_l:] < safe_lower_bound):
            return False
        if len(d2) > min_l and np.any(d2[min_l:] < safe_lower_bound):
            return False

    return True


def compare_z_profiles(z1: np.ndarray, z2: np.ndarray, z_tol: float = 0.5) -> bool:
    """
    Compares the Z-height distribution of two physisorbed molecules.
    A simple absolute difference check (e.g., within 0.5 Angstroms).
    """
    if len(z1) != len(z2):
        return False
    return np.max(np.abs(z1 - z2)) <= z_tol


def is_equivalent_state(
    atoms1: Atoms,
    atoms2: Atoms,
    energy1: float,
    energy2: float,
    ads_idx: list[int],
    sub_idx: list[int],
    energy_tol: float = 0.1,
    intra_tol: float = 1.1,
    chem_tol: float = 1.5,
    rel_dev_tol: float = 0.08,
    z_profile_tol: float = 0.4
) -> bool:
    """
    The unified classification workflow.
    """
    # 1. Thermodynamic Energy Check
    if abs(energy1 - energy2) > energy_tol:
        return False

    # Extract all signatures
    adj1, chem_prof1, z_prof1 = extract_signatures(atoms1, ads_idx, sub_idx, intra_tol, chem_tol)
    adj2, chem_prof2, z_prof2 = extract_signatures(atoms2, ads_idx, sub_idx, intra_tol, chem_tol)

    # 2. Internal Molecular Topology Check
    if not np.array_equal(adj1, adj2):
        return False

    has_chem1 = len(chem_prof1) > 0
    has_chem2 = len(chem_prof2) > 0

    if has_chem1 != has_chem2:
        return False  # Divergent adsorption states

    # 3. Binding Mode / Symmetry Check
    if has_chem1 and has_chem2:
        # Chemisorption: Evaluate specific anchoring interactions.
        return compare_chem_profiles(chem_prof1, chem_prof2, rel_dev_tol, chem_tol)
    else:
        # Physisorption: Evaluate macroscopic posture via Z-Profile.
        return compare_z_profiles(z_prof1, z_prof2, z_profile_tol)
