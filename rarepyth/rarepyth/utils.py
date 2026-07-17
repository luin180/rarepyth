# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:36:27 2024

@author: Wang Junhao
"""
import os
import random
import numpy as np


def make_target_dir(path):
    if os.path.exists(path):
        return
    else:
        try:
            os.mkdir(path)
            return
        except OSError:
            make_target_dir(os.path.dirname(path))
            os.mkdir(path)
            return


def get_monoto_score(
    series
):
    _series = np.array(series).reshape((-1, 1))
    pos = neg = 0
    for i in range(_series.shape[0] - 1):
        if _series[i + 1][0] > _series[i][0]:
            pos = pos + 1
        else:
            neg = neg + 1
    return abs(pos - neg) / max(pos, neg)


def get_range_score(
    series,
    extreme_ratio=0.05
):
    _series = np.array(series).flatten()
    n = _series.shape[0]
    x_axis = np.arange(n)
    slope, intercept = np.polyfit(x_axis, _series, 1)
    
    predicted_diff = abs(slope * n)
    sorted_series = np.sort(_series)
    upper_bound_idx = int((1 - extreme_ratio) * n) - 1
    lower_bound_idx = int(extreme_ratio * n) + 1
    
    mean_upper = np.mean(sorted_series[upper_bound_idx:])
    mean_lower = np.mean(sorted_series[:lower_bound_idx])
    
    return predicted_diff / abs(mean_upper - mean_lower)


def get_periodic_distance(
    coord_1,
    coord_2,
    matrix,
    coords_are_cartesian=False
):
    if coords_are_cartesian:
        coord_1 = np.dot(coord_1, np.linalg.inv(matrix))
        coord_2 = np.dot(coord_2, np.linalg.inv(matrix))
    distance = np.inf
    for trans_a in range(-1, 2):
        for trans_b in range(-1, 2):
            for trans_c in range(-1, 2):
                trans = np.array([trans_a, trans_b, trans_c])
                distance = min(
                    distance,
                    np.linalg.norm(np.dot(coord_2 - coord_1 + trans, matrix))
                )
    return distance


def get_nearest_atoms(
    site,
    structure,
    tol=0.01,
    coords_are_cartesian=False
):
    nearest_distance = np.inf
    nearest_atoms = []
    for periodic_site in structure.sites:
        if coords_are_cartesian:
            distance = get_periodic_distance(
                site,
                periodic_site.coords,
                structure.lattice.matrix,
                coords_are_cartesian=True
            )
        else:
            distance = get_periodic_distance(
                site,
                periodic_site.frac_coords,
                structure.lattice.matrix
            )
        if distance <= nearest_distance + tol:
            if distance <= nearest_distance - tol:
                nearest_atoms = [periodic_site]
                nearest_distance = distance
            else:
                nearest_atoms.append(periodic_site)
    return nearest_atoms


def fix_atoms_by_num(
    structure,
    selected_atoms,
    directions=[False, False, False],
    relax_the_rest=True,
    in_place=True
):
    """Fix atoms in lattice by given order.

    Parameters
    ----------
    structure: pymatgen Structure
        Initial input structure.
    selected_atoms: list, range or slice
        Which atoms should be fixed. Examples:\n
        Assuming your structure has 9 atoms.\n
        i. To fix the first 5 atoms, input [0,1,2,3,4], range(5) or
        slice(None, 5, None).\n
        ii. To fix last 4 atoms, input [5,6,7,8], [-4,-3,-2,-1], range(5,9),
        range(-4,0), slice(5, None, None) or slice(-4, None, None).\n
        iii. To fix all atoms except the last 4, input [0,1,2,3,4], range(9-4)
        or slice(None, -4, None). However, if you have many atoms in your
        structure or just forget the number, use slice is the best option.
    directions: array-like, optional, default=[False, False, False]
        Which direction of selected atoms should be fixed. Its meaning is the
        same as in POSCAR files.
    relax_the_rest: bool, optional, default=True
        Whether to relax the unselected atoms.
    in_place: bool, optional, default=True
        Whether to change the input object directly.

    Returns
    -------
    new_structure : pymatgen Structure
        Fixed input structure.

    """

    new_structure = structure if in_place else structure.copy()
    index_list = list(range(len(structure)))
    selected_list = []
    if isinstance(selected_atoms, slice):
        selected_list = index_list[selected_atoms]
    else:
        for num in selected_atoms:
            selected_list.append(index_list[num])
    for num in index_list:
        if num in selected_list:
            new_structure.sites[num].properties[
                'selective_dynamics'] = np.array(directions)
        elif relax_the_rest:
            new_structure.sites[num].properties[
                'selective_dynamics'] = np.array([True] * 3)
    return new_structure


def fix_atoms_by_height(
    structure,
    z_min,
    z_max,
    coords_are_cartesian=False,
    directions=[False, False, False],
    relax_the_rest=True,
    in_place=True
):
    new_structure = structure if in_place else structure.copy()
    if coords_are_cartesian:
        coords = structure.cart_coords
    else:
        coords = structure.frac_coords
    for i, atom in enumerate(coords):
        if atom[2] >= z_min and atom[2] <= z_max:
            new_structure.sites[i].properties[
                'selective_dynamics'] = np.array(directions)
        elif relax_the_rest:
            new_structure.sites[i].properties[
                'selective_dynamics'] = np.array([True] * 3)
    return new_structure


def fix_atoms_by_center_and_radius(
    structure,
    center,
    radius,
    coords_are_cartesian=False,
    directions=[False, False, False],
    relax_the_rest=True,
    in_place=True
):
    new_structure = structure if in_place else structure.copy()
    if coords_are_cartesian:
        coords = structure.cart_coords
    else:
        coords = structure.frac_coords
    for i, atom in enumerate(coords):
        if get_periodic_distance(
                atom,
                center,
                structure.lattice.matrix,
                coords_are_cartesian=coords_are_cartesian
        ) > radius:
            new_structure.sites[i].properties[
                'selective_dynamics'] = np.array(directions)
        elif relax_the_rest:
            new_structure.sites[i].properties[
                'selective_dynamics'] = np.array([True] * 3)
    return new_structure


