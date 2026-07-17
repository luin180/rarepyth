# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:10:47 2024

@author: Wang Junhao
"""

import re
from math import sqrt
from copy import deepcopy

import numpy as np
from pymatgen.core import Structure


class Oszicar:
    def __init__(self, ionic_steps, electronic_steps):
        self.ionic_steps = deepcopy(ionic_steps)
        self.electronic_steps = deepcopy(electronic_steps)

    @classmethod
    def read(cls, OSZICAR='OSZICAR'):
        ionic_steps = []
        electronic_steps = []
        with open(OSZICAR, 'r') as file:
            for line in file:
                if '       N' in line:
                    ionic_steps.append({})
                    electronic_steps.append([])
                if '=' in line:
                    F = r"[\d\s]?F=\s*(-?\d*\.\d*E*[+-]*\d*)"
                    ionic_steps[-1]['F'] = float(re.findall(F, line)[0])
                    E0 = r"[\d\s]?E0=\s*(-?\d*\.\d*E*[+-]*\d*)"
                    ionic_steps[-1]['E0'] = float(re.findall(E0, line)[0])
                    if 'd E' in line:
                        dE = r"[\d\s]?d\sE\s=\s*(-?\d*\.\d*E*[+-]*\d*)"
                        ionic_steps[-1]['dE'] = float(re.findall(dE, line)[0])
                    pattern = r"([A-Za-z]+\(*[A-Za-z]*\)*)\s*=\s*(-?\d*\.\d*E*[+-]*\d*)"
                    matches = re.findall(pattern, line)
                    for key, value in matches:
                        ionic_steps[-1][key] = float(value)
                if 'DAV' in line or 'RMM' in line or 'CG' in line or 'SDA' in line:
                    pattern = r"(-?[\d\.]*\d+E?[+-]?\d?\d?)"
                    matches = re.findall(pattern, line.strip().split(':')[1])
                    electronic_steps[-1].append([
                        float(matches[1]),
                        float(matches[2]),
                        float(matches[3]),
                        float(matches[4]),
                        float(matches[5])
                    ])
                    try:
                        electronic_steps[-1][-1].append(float(matches[6]))
                    except IndexError:
                        electronic_steps[-1][-1].append(np.nan)

        if ionic_steps and not ionic_steps[-1]:
            ionic_steps.pop()
        return cls(ionic_steps, electronic_steps)

    def copy(self):
        return type(self)(self.ionic_steps, self.electronic_steps)

    @property
    def num_ionic_steps(self):
        return len(self.ionic_steps)

    @property
    def num_elect_steps(self):
        _list = []
        for step in self.electronic_steps:
            _list.append(len(step))
        return _list

    @property
    def ionic_energies(self):
        E0 = []
        for step in self.ionic_steps:
            E0.append(step['E0'])
        return E0

    @property
    def ionic_energies_diff(self):
        dE = []
        for step in self.ionic_steps:
            dE.append(step['dE'])
        return dE

    @property
    def elect_energies(self):
        E = []
        for loop in self.electronic_steps:
            E.append([])
            for step in loop:
                E[-1].append(step[0])
        return E

    @property
    def elect_energies_diff(self):
        dE = []
        for loop in self.electronic_steps:
            dE.append([])
            for step in loop:
                dE[-1].append(step[1])
        return dE

    @property
    def eps_energies_diff(self):
        deps = []
        for loop in self.electronic_steps:
            deps.append([])
            for step in loop:
                deps[-1].append(step[2])
        return deps


class Outcar:
    def __init__(self, content):
        self.content = content

    @classmethod
    def read(cls, OUTCAR='OUTCAR'):
        with open(OUTCAR, 'r') as file:
            content = file.readlines()
        return cls(content)

    def copy(self):
        return type(self)(self.content)

    @property
    def ions_per_type(self):
        ions_per_type = []
        for line in self.content:
            if 'ions per type' in line:
                for i in line.split('=')[1].split():
                    ions_per_type.append(int(i))
                break
        return ions_per_type

    @property
    def num_atoms(self):
        num_atoms = 0
        for ion in self.ions_per_type:
            num_atoms = num_atoms + ion
        return num_atoms

    @property
    def ions_species(self):
        ions_species = []
        for line in self.content:
            if 'VRHFIN' in line:
                ions_species.append(line.split('=')[1].split(':')[0].strip())
                if len(ions_species) == len(self.ions_per_type):
                    break
        return ions_species

    @property
    def elect_converged(self):
        if self.num_elect_inconvergence:
            return False
        else:
            return True

    @property
    def num_elect_inconvergence(self):
        count = 0
        for line in self.content:
            if 'aborting loop EDIFF was not reached' in line:
                count += 1
        return count

    @property
    def num_ionic_steps(self):
        num_ionic_steps = 0
        for line in self.content:
            if 'POSITION' in line.split():
                num_ionic_steps = num_ionic_steps + 1
        return num_ionic_steps

    @property
    def ionic_converged(self):
        for line in self.content:
            if 'reached required accuracy' in line:
                return True
        else:
            return False

    @property
    def list_relaxtion_structures(self):
        # scaling_factor = 1.0
        species = []
        for i in range(len(self.ions_species)):
            species = species + [self.ions_species[i]] * self.ions_per_type[i]

        lattice_matrices = []
        read = []
        ions_positions = []
        for i, line in enumerate(self.content):
            if 'VOLUME and BASIS-vectors are now' in line:
                a = self.content[i + 5].split()
                b = self.content[i + 6].split()
                c = self.content[i + 7].split()
                lattice_matrices.append(
                    [[float(a[0]), float(a[1]), float(a[2])],
                     [float(b[0]), float(b[1]), float(b[2])],
                     [float(c[0]), float(c[1]), float(c[2])]]
                )
            if 'POSITION' in line:
                read = list(range(i + 2, i + 2 + self.num_atoms))
                ions_positions.append([])
            if i in read:
                ions_positions[-1].append([])
                for j in range(3):
                    ions_positions[-1][-1].append(float(line.split()[j]))
        assert len(lattice_matrices) == len(ions_positions)

        list_relaxtion_structures = []
        for i in range(len(lattice_matrices)):
            list_relaxtion_structures.append(Structure(
                lattice_matrices[i],
                species,
                ions_positions[i],
                coords_are_cartesian=True
            ))
        return list_relaxtion_structures

    @property
    def is_normal_terminated(self):
        for line in self.content:
            if 'General timing and accounting informations' in line:
                return True
        else:
            return False

    @property
    def error_info(self):
        for i, line in enumerate(self.content):
            if "EEEEEEE  R " in line:
                return self.content[i + 2][1:-1].strip()
        else:
            return None

    @property
    def total_charge(self):
        for i, line in enumerate(self.content):
            if line == ' total charge\n':
                charge_content = self.content[i + 4: i + self.num_atoms + 4]
                break
        else:
            return None

        charge_list = []
        for line in charge_content:
            line_list = line.strip().split()[1:]
            for i, chg in enumerate(line_list):
                line_list[i] = float(chg)
            charge_list.append(line_list)
        return charge_list

    def get_filtered_forces_list(self, slt_dynamics=None):
        if not slt_dynamics:
            slt_dynamics = [[1] * 3] * self.num_atoms

        read = []
        forces = []
        for i, line in enumerate(self.content):
            if 'POSITION' in line.split():
                read = list(range(i + 2, i + 2 + self.num_atoms))
                forces.append([])
                num = 0
            if i in read:
                forces[-1].append(
                    sqrt((slt_dynamics[num][0] * float(line.split()[3])) ** 2 +
                         (slt_dynamics[num][1] * float(line.split()[4])) ** 2 +
                         (slt_dynamics[num][2] * float(line.split()[5])) ** 2)
                )
                num = num + 1
        return forces

    def get_max_filtered_forces(self, **kwargs):
        return [max(step) for step in self.get_filtered_forces_list(**kwargs)]
