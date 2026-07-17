# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:39:10 2024

@author: Wang Junhao
"""

import os
import time
import datetime
import sys
import re
import shutil
import tempfile
import tarfile
import subprocess as sp
from collections import OrderedDict
from copy import deepcopy
from fnmatch import fnmatch

import numpy as np
from monty.io import zopen
from pymatgen.core import Structure

from rarepyth.base import _mswindows, Message
from rarepyth.utils import get_monoto_score, get_range_score
from rarepyth.vasp.output import Oszicar, Outcar
import rarepyth.vasp.errorhandle as eh

if not _mswindows:
    from rarepyth.base import runcmd, checkcmd

DICT_RCMD = {'H': 'H', 'He': 'He', 'Li': 'Li_sv', 'Be': 'Be', 'B': 'B',
             'C': 'C', 'N': 'N', 'O': 'O', 'F': 'F', 'Ne': 'Ne', 'Na': 'Na_pv',
             'Mg': 'Mg', 'Al': 'Al', 'Si': 'Si', 'P': 'P', 'S': 'S',
             'Cl': 'Cl', 'Ar': 'Ar', 'K': 'K_sv', 'Ca': 'Ca_sv', 'Sc': 'Sc_sv',
             'Ti': 'Ti_sv', 'V': 'V_sv', 'Cr': 'Cr_pv', 'Mn': 'Mn_pv',
             'Fe': 'Fe', 'Co': 'Co', 'Ni': 'Ni', 'Cu': 'Cu', 'Zn': 'Zn',
             'Ga': 'Ga_d', 'Ge': 'Ge_d', 'As': 'As', 'Se': 'Se', 'Br': 'Br',
             'Kr': 'Kr', 'Rb': 'Rb_sv', 'Sr': 'Sr_sv', 'Y': 'Y_sv',
             'Zr': 'Zr_sv', 'Nb': 'Nb_sv', 'Mo': 'Mo_sv', 'Tc': 'Tc_pv',
             'Ru': 'Ru_pv', 'Rh': 'Rh_pv', 'Pd': 'Pd', 'Ag': 'Ag', 'Cd': 'Cd',
             'In': 'In_d', 'Sn': 'Sn_d', 'Sb': 'Sb', 'Te': 'Te', 'I': 'I',
             'Xe': 'Xe', 'Cs': 'Cs_sv', 'Ba': 'Ba_sv', 'La': 'La', 'Ce': 'Ce',
             'Pr': 'Pr_3', 'Nd': 'Nd_3', 'Pm': 'Pm_3', 'Sm': 'Sm_3',
             'Eu': 'Eu_2', 'Gd': 'Gd_3', 'Tb': 'Tb_3', 'Dy': 'Dy_3',
             'Ho': 'Ho_3', 'Er': 'Er_3', 'Tm': 'Tm_3', 'Yb': 'Yb_2',
             'Lu': 'Lu_3', 'Hf': 'Hf_pv', 'Ta': 'Ta_pv', 'W': 'W_sv',
             'Re': 'Re', 'Os': 'Os', 'Ir': 'Ir', 'Pt': 'Pt', 'Au': 'Au',
             'Hg': 'Hg', 'Tl': 'Tl_d', 'Pb': 'Pb_d', 'Bi': 'Bi_d',
             'Po': 'Po_d', 'At': 'At', 'Rn': 'Rn', 'Fr': 'Fr_sv',
             'Ra': 'Ra_sv', 'Ac': 'Ac', 'Th': 'Th', 'Pa': 'Pa', 'U': 'U',
             'Np': 'Np', 'Pu': 'Pu', 'Am': 'Am', 'Cm': 'Cm', 'Cf': 'Cf'}
DICT_F = {'H': 'H', 'He': 'He', 'Li': 'Li_sv', 'Be': 'Be', 'B': 'B', 'C': 'C',
          'N': 'N', 'O': 'O', 'F': 'F', 'Ne': 'Ne', 'Na': 'Na_pv', 'Mg': 'Mg',
          'Al': 'Al', 'Si': 'Si', 'P': 'P', 'S': 'S', 'Cl': 'Cl', 'Ar': 'Ar',
          'K': 'K_sv', 'Ca': 'Ca_sv', 'Sc': 'Sc_sv', 'Ti': 'Ti_sv',
          'V': 'V_sv', 'Cr': 'Cr_pv', 'Mn': 'Mn_pv', 'Fe': 'Fe', 'Co': 'Co',
          'Ni': 'Ni', 'Cu': 'Cu', 'Zn': 'Zn', 'Ga': 'Ga_d', 'Ge': 'Ge_d',
          'As': 'As', 'Se': 'Se', 'Br': 'Br', 'Kr': 'Kr', 'Rb': 'Rb_sv',
          'Sr': 'Sr_sv', 'Y': 'Y_sv', 'Zr': 'Zr_sv', 'Nb': 'Nb_sv',
          'Mo': 'Mo_sv', 'Tc': 'Tc_pv', 'Ru': 'Ru_pv', 'Rh': 'Rh_pv',
          'Pd': 'Pd', 'Ag': 'Ag', 'Cd': 'Cd', 'In': 'In_d', 'Sn': 'Sn_d',
          'Sb': 'Sb', 'Te': 'Te', 'I': 'I', 'Xe': 'Xe', 'Cs': 'Cs_sv',
          'Ba': 'Ba_sv', 'La': 'La', 'Ce': 'Ce', 'Pr': 'Pr', 'Nd': 'Nd',
          'Pm': 'Pm', 'Sm': 'Sm', 'Eu': 'Eu', 'Gd': 'Gd', 'Tb': 'Tb',
          'Dy': 'Dy', 'Ho': 'Ho', 'Er': 'Er', 'Tm': 'Tm', 'Yb': 'Yb',
          'Lu': 'Lu', 'Hf': 'Hf_pv', 'Ta': 'Ta_pv', 'W': 'W_sv', 'Re': 'Re',
          'Os': 'Os', 'Ir': 'Ir', 'Pt': 'Pt', 'Au': 'Au', 'Hg': 'Hg',
          'Tl': 'Tl_d', 'Pb': 'Pb_d', 'Bi': 'Bi_d', 'Po': 'Po_d', 'At': 'At',
          'Rn': 'Rn', 'Fr': 'Fr_sv', 'Ra': 'Ra_sv', 'Ac': 'Ac', 'Th': 'Th',
          'Pa': 'Pa', 'U': 'U', 'Np': 'Np', 'Pu': 'Pu', 'Am': 'Am',
          'Cm': 'Cm', 'Cf': 'Cf'}
DICT_GW = {'H': 'H_GW', 'He': 'He_GW', 'Li': 'Li_sv_GW', 'Be': 'Be_sv_GW',
           'B': 'B_GW', 'C': 'C_GW', 'N': 'N_GW', 'O': 'O_GW', 'F': 'F_GW',
           'Ne': 'Ne_GW', 'Na': 'Na_sv_GW', 'Mg': 'Mg_sv_GW', 'Al': 'Al_GW',
           'Si': 'Si_GW', 'P': 'P_GW', 'S': 'S_GW', 'Cl': 'Cl_GW',
           'Ar': 'Ar_GW', 'K': 'K_sv_GW', 'Ca': 'Ca_sv_GW', 'Sc': 'Sc_sv_GW',
           'Ti': 'Ti_sv_GW', 'V': 'V_sv_GW', 'Cr': 'Cr_sv_GW',
           'Mn': 'Mn_sv_GW', 'Fe': 'Fe_sv_GW', 'Co': 'Co_sv_GW',
           'Ni': 'Ni_sv_GW', 'Cu': 'Cu_sv_GW', 'Zn': 'Zn_sv_GW',
           'Ga': 'Ga_d_GW', 'Ge': 'Ge_d_GW', 'As': 'As_GW', 'Se': 'Se_GW',
           'Br': 'Br_GW', 'Kr': 'Kr_GW', 'Rb': 'Rb_sv_GW', 'Sr': 'Sr_sv_GW',
           'Y': 'Y_sv_GW', 'Zr': 'Zr_sv_GW', 'Nb': 'Nb_sv_GW',
           'Mo': 'Mo_sv_GW', 'Tc': 'Tc_sv_GW', 'Ru': 'Ru_sv_GW',
           'Rh': 'Rh_sv_GW', 'Pd': 'Pd_sv_GW', 'Ag': 'Ag_sv_GW',
           'Cd': 'Cd_sv_GW', 'In': 'In_d_GW', 'Sn': 'Sn_d_GW', 'Sb': 'Sb_d_GW',
           'Te': 'Te_GW', 'I': 'I_GW', 'Xe': 'Xe_GW', 'Cs': 'Cs_sv_GW',
           'Ba': 'Ba_sv_GW', 'La': 'La_GW', 'Ce': 'Ce_GW', 'Hf': 'Hf_sv_GW',
           'Ta': 'Ta_sv_GW', 'W': 'W_sv_GW', 'Re': 'Re_sv_GW',
           'Os': 'Os_sv_GW', 'Ir': 'Ir_sv_GW', 'Pt': 'Pt_sv_GW',
           'Au': 'Au_sv_GW', 'Hg': 'Hg_sv_GW', 'Tl': 'Tl_d_GW',
           'Pb': 'Pb_d_GW', 'Bi': 'Bi_d_GW', 'Po': 'Po_d_GW', 'At': 'At_d_GW',
           'Rn': 'Rn_d_GW'}
F_BLOCK = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
           'Er', 'Tm', 'Yb', 'Lu'}
D_BLOCK = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr',
           'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W',
           'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'}
U_VALUES = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2,
            'V': 3.25, 'W': 6.2,  # From the Materials Project, also used in OC22
            'La': 4.3, 'Ce': 5.8, 'Pr': 5.2, 'Nd': 4.8, 'Sm': 3.4, 'Eu': 6.0,
            'Gd': 4.1, 'Tb': 2.9, 'Dy': 3.4, 'Ho': 3.4, 'Er': 6.5, 'Tm': 6.4,
            'Yb': 7.8, 'Lu': 0.9  # From Linear Reaction Fitting
            }

if not _mswindows:

    class rarepyth_system:
        def __init__(self, func):
            self.func = func
            self.path = {}
            try:
                self.path['lda'] = checkcmd(
                    'grep LDA_PATH ~/.rarepyth').split('=')[1].split()[0]
            except sp.CalledProcessError:
                try:
                    self.path['lda'] = checkcmd(
                        'grep LDA_PATH ~/.vaspkit').split('=')[1].split()[0]
                except sp.CalledProcessError:
                    Message("Can't find potcar path").exitm()

            try:
                self.path['pbe'] = checkcmd(
                    'grep PBE_PATH ~/.rarepyth').split('=')[1].split()[0]
            except sp.CalledProcessError:
                try:
                    self.path['pbe'] = checkcmd(
                        'grep PBE_PATH ~/.vaspkit').split('=')[1].split()[0]
                except sp.CalledProcessError:
                    Message("Can't find potcar path").exitm()

        def __call__(self, *args, **kwargs):
            return self.func(*args,
                             self.path,
                             **kwargs)


class Incar(OrderedDict):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_file(cls, INCAR):
        def convert_str(_str):
            try:
                values = int(_str)
            except ValueError:
                try:
                    values = float(_str)
                except ValueError:
                    values = _str
            return values
        _object = cls()
        with open(INCAR, 'r') as file:
            for line in file:
                if line and line[0] not in ('#', '!', '('):
                    try:
                        tag = line.split('=')[0].strip()
                        values = (line.split('=')[1].split('#')[0]
                                  .split('!')[0].split('(')[0].strip())
                        if values == '.TRUE.':
                            values = True
                        elif values == '.FALSE.':
                            values = False
                        else:
                            if len(values.split()) > 1:
                                values_list = []
                                for val in values.split():
                                    values_list.append(convert_str(val))
                                values = values_list
                            else:
                                values = convert_str(values)
                        _object[tag] = values
                    except IndexError:
                        continue
        return _object

    def to_file(self, INCAR):
        seq = []
        with open(INCAR, 'w') as file:
            for tag in self.keys():
                if isinstance(self[tag], bool):
                    if self[tag]:
                        values = '.TRUE.'
                    else:
                        values = '.FALSE.'
                elif isinstance(self[tag], list):
                    values = ''
                    for val in self[tag]:
                        values = values + str(val) + ' '
                else:
                    values = self[tag]
                seq.append("{} = {}\n".format(tag, values))
            file.writelines(seq)

    def erase(self, key):
        if key in self.keys():
            del self[key]


class Poscar(Structure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.find_layers()
        self.tags = [0] * self.num_sites

    def copy(self):
        _object = super().copy()
        _object.layer_dict = self.layer_dict
        _object.tags = self.tags
        return _object

    def find_layers(self, error_tol=0.01):
        label_z_coords = np.array([list(range(len(self.cart_coords[:, 2]))),
                                   self.cart_coords[:, 2]])
        label_z_coords = label_z_coords[:, np.argsort(label_z_coords[1])]

        gap_list = []
        for idx in range(self.num_sites - 1):
            gap_list.append(label_z_coords[1][idx + 1] - label_z_coords[1][idx])
        gap_list.sort(reverse=True)
        for idx, gap in enumerate(gap_list):
            if gap_list[0] - gap > 2 * error_tol:
                layer_gap = np.average(gap_list[:idx])
                break
        self.layer_dict = {}
        layer = 0
        for idx in range(self.num_sites):
            if idx and label_z_coords[1][idx] - label_z_coords[1][idx - 1] >= layer_gap - error_tol:
                layer += 1
            self.layer_dict[int(label_z_coords[0][idx])] = layer

    def get_tags_by_layers(self):
        tags = []
        max_layer = max(self.layer_dict.values())
        for idx in range(self.num_sites):
            tags.append(int(self.layer_dict[idx] == max_layer))
        self.tags = tags

    def read_tags_from_file(self, TAG='TAG'):
        with open(TAG, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                self.tags[int(line.split()[0]) - 1] = int(line.split()[1])

    def edit_tags(self, idxes, values):
        """Set values of tags of selected atoms.

        Parameters
        ----------
        idxes : list or string
            As list, should be the indexes of selected atoms, but start from 1.
            As string, will be read like vaspkit's input, e.g. "1,3,6-9" equals to [1,3,6,7,8,9].
        values : int or list
            As int, all selected atoms will use this value.
            As list, each atom will use corresponding value.

        Returns
        -------
        No returns.

        """

        if isinstance(idxes, str):
            list_idxes = []
            for substr in idxes.split(','):
                try:
                    list_idxes.append(int(substr))
                except ValueError:
                    list_idxes += list(range(
                        int(substr.split('-')[0]), int(substr.split('-')[1]) + 1))
            idxes = list_idxes
        if isinstance(values, int):
            values = [values] * len(idxes)
        for j, idx in enumerate(idxes):
            self.tags[idx - 1] = values[j]

    def fix_atoms_by_tags(self, fixed_tags=[0], directions=[False, False, False]):
        for idx in range(self.num_sites):
            if self.tags[idx] in fixed_tags:
                self.sites[idx].properties[
                    'selective_dynamics'] = np.array(directions)
            else:
                self.sites[idx].properties[
                    'selective_dynamics'] = np.array([True] * 3)


class Kpoints:
    def __init__(
        self,
        comment,
        scheme,
        subdivisions,
        shift
    ):
        self.comment = comment
        self.scheme = scheme
        self.subdivisions = subdivisions
        self.shift = shift

    @classmethod
    def from_str(cls, _str):
        if (_str[2].strip()[0] == 'G' or
                _str[2].strip()[0] == 'g'):
            scheme = 'Gamma'
        elif (_str[2].strip()[0] == 'M' or
                _str[2].strip()[0] == 'm'):
            scheme = 'Monkhorst-Pack'
        else:
            Message("Invalid KPOINTS scheme").exitm()
        subdivisions = []
        shift = []
        for i in [0, 1, 2]:
            subdivisions.append(int(_str[3].strip().split()[i]))
            shift.append(float(_str[4].strip().split()[i]))
        return cls(
            _str[0],
            scheme,
            subdivisions,
            shift
        )

    @classmethod
    def from_file(cls, KPOINTS):
        with open(KPOINTS, 'r') as file:
            contents = file.readlines()
        return cls.from_str(contents)

    def to_file(self, KPOINTS):
        with open(KPOINTS, 'w') as file:
            file.writelines([
                self.comment,
                '0\n',
                self.scheme + '\n',
                '   {0[0]}   {0[1]}   {0[2]}\n'.format(self.subdivisions),
                '{0[0]}  {0[1]}  {0[2]}\n'.format(self.shift)
            ])

    def copy(self):
        return type(self)(self.comment,
                          self.scheme,
                          self.subdivisions.copy(),
                          self.shift.copy())


class Potcar():
    def __init__(self, _list):
        self._list = deepcopy(_list)

    @classmethod
    def from_file(cls, POTCAR):
        _object = cls([])
        with open(POTCAR, 'r') as file:
            for line in file:
                try:
                    if line[:8] == '   TITEL':
                        _object._list.append(line[12:].split()[0:2])
                except IndexError:
                    pass
        return _object

    def copy(self):
        return type(self)(self._list)

    if not _mswindows:
        @rarepyth_system
        def _to_file(self, POTCAR, path):
            potcar_path = ''
            for pot in self._list:
                if pot[0] == 'PAW_LDA':
                    potcar_path = potcar_path + '{}/{}/POTCAR '.format(
                        path['lda'], pot[1])
                elif pot[0] == 'PAW_PBE':
                    potcar_path = potcar_path + '{}/{}/POTCAR '.format(
                        path['pbe'], pot[1])
                else:
                    Message("Invalid functional type").exitm()
            runcmd('cat {}> {}'.format(potcar_path, POTCAR))

        def to_file(self, POTCAR):
            self._to_file(self, POTCAR)


class VaspJob:
    def __init__(
        self,
        incar,
        poscar,
        kpoints,
        potcar,
        chgcar_path=None,
        wavecar_path=None,
        oszicar=None,
        outcar=None
    ):
        """Create a vasp job.

        Parameters
        ----------
        incar: rarepyth Incar object or None\n
        poscar: pymatgen Structure object or None\n
        kpoints: rarepyth Kpoints object or None\n
        potcar: rarepyth Potcar object or None\n
        chgcar: path to CHGCAR file or None\n
        wavecar: path to WAVECAR file or None\n
        oszicar: rarepyth Oszicar object or None\n
        outcar: rarepyth Outcar object or None\n

        """
        self._cache_path = tempfile.mkdtemp(prefix='cache_', dir='.')

        self.incar = incar.copy() if isinstance(incar, Incar) else None
        self.poscar = poscar.copy() if isinstance(poscar, Structure) else None
        self.kpoints = kpoints.copy() if isinstance(kpoints, Kpoints) else None
        self.potcar = potcar.copy() if isinstance(potcar, Potcar) else None
        if chgcar_path and os.path.exists(chgcar_path):
            self.readfile(CHGCAR=chgcar_path)
        else:
            self.chgcar_path = None
        if wavecar_path and os.path.exists(wavecar_path):
            self.readfile(WAVECAR=wavecar_path)
        else:
            self.wavecar_path = None
        self.oszicar = oszicar.copy() if isinstance(oszicar, Oszicar) else None
        self.outcar = outcar.copy() if isinstance(outcar, Outcar) else None

    def __del__(self):
        shutil.rmtree(self._cache_path)

    @classmethod
    def from_files(
        cls,
        **kwargs
    ):
        object_ = cls(None, None, None, None)
        object_.readfile(**kwargs)
        return object_

    def readfile(self, **kwargs):
        if 'INCAR' in kwargs:
            self.incar = Incar.from_file(kwargs['INCAR'])

        if 'POSCAR' in kwargs:
            with zopen(kwargs['POSCAR'], "rt", errors="replace") as file:
                contents = file.read()
            fname = os.path.basename(kwargs['POSCAR'])
            if fnmatch(fname.lower(), "*.cif*") or fnmatch(fname.lower(), "*.mcif*"):
                self.poscar = Structure.from_str(contents, fmt="cif")
            else:
                self.poscar = Structure.from_str(contents, fmt="poscar")

        if 'KPOINTS' in kwargs:
            with open(kwargs['KPOINTS'], 'r') as file:
                contents = file.readlines()
            if not contents[1].strip().split()[0] == '0':
                self.kpoints = contents
            else:
                self.kpoints = Kpoints.from_str(contents)

        if 'POTCAR' in kwargs:
            self.potcar = Potcar.from_file(kwargs['POTCAR'])

        if 'OSZICAR' in kwargs:
            self.oszicar = Oszicar.read(OSZICAR=kwargs['OSZICAR'])

        if 'OUTCAR' in kwargs:
            self.outcar = Outcar.read(OUTCAR=kwargs['OUTCAR'])

        if 'CHGCAR' in kwargs:
            self.chgcar_path = f'{self._cache_path}/CHGCAR'
            shutil.copy(kwargs['CHGCAR'], self.chgcar_path)

        if 'WAVECAR' in kwargs:
            self.wavecar_path = f'{self._cache_path}/WAVECAR'
            shutil.copy(kwargs['WAVECAR'], self.wavecar_path)

    def writefile(self, **kwargs):
        if 'INCAR' in kwargs:
            self.incar.to_file(kwargs['INCAR'])

        if 'POSCAR' in kwargs:
            self.poscar.to(kwargs['POSCAR'], fmt='poscar')

        if 'KPOINTS' in kwargs:
            if isinstance(self.kpoints, str):
                with open(kwargs['KPOINTS'], 'w') as file:
                    file.write(self.kpoints)
            else:
                self.kpoints.to_file(kwargs['KPOINTS'])

        if 'CHGCAR' in kwargs:
            shutil.copy(self.chgcar_path, kwargs['CHGCAR'])

        if 'WAVECAR' in kwargs:
            shutil.copy(self.wavecar_path, kwargs['WAVECAR'])

        if not _mswindows:
            if 'POTCAR' in kwargs:
                self.potcar.to_file(kwargs['POTCAR'])

    @classmethod
    def from_checkpoint(cls, checkpoint_path=None):
        if checkpoint_path is None:
            cwd = os.getcwd()
            newest = 0
            for root, dirs, files in os.walk(cwd):
                for file in files:
                    if os.path.splitext(file)[-1] == '.vjz' or '.VJZ':
                        now = os.stat(file).st_mtime
                        if now > newest:
                            newest = now
                            checkpoint_path = os.path.join(root, file)
                else:
                    Message('No VJZ File Found.').exitm()
        else:
            assert os.path.splitext(checkpoint_path)[-1] == '.vjz' or '.VJZ'
        object_ = cls(None, None, None, None, None, None, None, None)
        object_.load_checkpoint(checkpoint_path)
        return object_

    def load_checkpoint(self, checkpoint_path, tmp_file_dir='tmp_checkpoint'):
        if os.path.exists(tmp_file_dir):
            shutil.rmtree(tmp_file_dir)
        os.mkdir(tmp_file_dir)
        with tarfile.open(checkpoint_path, 'r') as tar:
            tar.extractall(path=tmp_file_dir)
        for root, dirs, files in os.walk(tmp_file_dir):
            for file in files:
                if file == 'INCAR':
                    self.readfile(INCAR=os.path.join(root, file))
                if file == 'POSCAR':
                    self.readfile(POSCAR=os.path.join(root, file))
                if file == 'KPOINTS':
                    self.readfile(KPOINTS=os.path.join(root, file))
                if file == 'POTCAR':
                    self.readfile(POTCAR=os.path.join(root, file))
                if file == 'CHGCAR':
                    self.readfile(CHGCAR=os.path.join(root, file))
                if file == 'WAVECAR':
                    self.readfile(WAVECAR=os.path.join(root, file))
        shutil.rmtree(tmp_file_dir)

    def save_checkpoint(self, checkpoint_path=None, tmp_file_dir='tmp_checkpoint'):
        if os.path.exists(tmp_file_dir):
            shutil.rmtree(tmp_file_dir)
        os.mkdir(tmp_file_dir)
        if checkpoint_path is None:
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            checkpoint_path = os.path.join(os.getcwd(), timestamp_str + '.vjt')

        if self.incar:
            self.writefile(INCAR=os.path.join(tmp_file_dir, 'INCAR'))
        if self.poscar:
            self.writefile(POSCAR=os.path.join(tmp_file_dir, 'POSCAR'))
        if self.kpoints:
            self.writefile(KPOINTS=os.path.join(tmp_file_dir, 'KPOINTS'))
        if self.potcar:
            self.writefile(POTCAR=os.path.join(tmp_file_dir, 'POTCAR'))
        if self.chgcar_path:
            shutil.copy2(self.chgcar_path, os.path.join(tmp_file_dir, 'CHGCAR'))
        if self.wavecar_path:
            shutil.copy2(self.wavecar_path, os.path.join(tmp_file_dir, 'WAVECAR'))

        with tarfile.open(checkpoint_path, 'w') as tar:
            tar.add(tmp_file_dir, arcname='')

        shutil.rmtree(tmp_file_dir)
        return checkpoint_path

    def gene_potcar(self, option='recommended', spc_list=None):
        """Generate PBE POTCAR info according to given option.
        Ref: https://www.vasp.at/wiki/index.php/Choosing_pseudopotentials

        Parameters
        ----------
        option: str, optional, default='recommended'
            'recommended': Use vasp wiki recommended potentials as vaspkit do.\n
            'f-electron': For lanthanides, use their f-electron versions.\n
            'GW': Use vasp wiki recommended GW potentials.\n
            'specified': Use potentials in spc_list.

        Returns
        -------
        None.

        """

        _list = []
        if option[0] == 'r' or option[0] == 'R':
            for ele in self.ion_species:
                _list.append(['PAW_PBE', DICT_RCMD[ele]])
        elif option[0] == 'f' or option[0] == 'F':
            for ele in self.ion_species:
                _list.append(['PAW_PBE', DICT_F[ele]])
        elif option[0] == 'g' or option[0] == 'G':
            for ele in self.ion_species:
                _list.append(['PAW_PBE', DICT_GW[ele]])
        elif option[0] == 's' or option[0] == 'S':
            if not spc_list or not len(spc_list) == len(self.ion_species):
                Message('Invalid specified POTCAR list.').exitm()
            for pot in spc_list:
                _list.append(['PAW_PBE', pot])
        else:
            Message('Invalid POTCAR generate option.').exitm()

        self.potcar = Potcar(_list)

    def set_calc_details(
        self,
        start_option='intWAVECHG',
        dipole_correction='incar',
        vdw='incar',
        dft_u='incar',
        u_values_dict={}
    ):
        """Set variant details used in calculation.


        Parameters
        ----------
        start_option : str or int, optional, default='intWAVECHG'
            Valid values: 'intWAVECHG', 'intWAVE', 'intCHG', 'extWAVECHG', 'extWAVE', 'extCHG', 'incar', 'none'\n
            Only in linux system, code will examine CHGCAR and WAVECAR in the object and writefile them correspondingly.\n
        dipole_correction : str, int or bool, optional, default='incar'
            As str, the first letter will determine how to perform dipole correction. 'x', 'y', 'z', 'a' corresponds to IDIPOL=1, 2, 3, 4, while 'n' corresponds to LDIPOL=.FALSE. Other words will keep the setting in INCAR.\n
            As int, integers between 1~4 corresponds to the value of IDIPOL, but only -1 correspond to LDIPOL=.FALSE. Other value will also keep the setting in INCAR.\n
            As bool, you can only choose to keep the setting in INCAR (as True) or not use dipole corrections (as False).\n
            This flag should be used for slab calculations, with the surface normal being the direction in which the IDIPOL is set. IDIPOL=4 should be used for calculations for isolated molecules.
        vdw : str, int or bool, optional, default='incar'
            As str, the following method will be enabled according to the keywords in string: 'bj' refer to DFT-D3(BJ), 'd3' refer to DFT-D3, 'd2' refer to DFT-D2(not suggested). If none of them are found and the first letter is 'n', IVDW will be set to 0, otherwides the setting in INCAR will be kept.\n
            As int, valid value(refer to https://www.vasp.at/wiki/index.php/IVDW) will be set as IVDW, except for 0, which refer to keep the setting in INCAR. Use -1 if you want to disable VDW corrections.\n
            As bool, you can only choose to keep the setting in INCAR (as True) or not use VDW corrections (as False).\n
        dft_u : str, int or bool, optional, default='incar'
            As str, only words like 'enable', 'true', 'yes' will enable LDA+U correction, while 'disable', 'false' or 'no' will disable it(only the first letter determines). Other words will keep the setting in INCAR.\n
            As int, positive value refer to enable, negative value refer to disable, and 0 refer to keep the setting in INCAR.\n
            As bool, you can only choose to enable (as True) or disable LDA+U corrections (as False).\n
        u_values_dict : dict, optional, default={}
            DESCRIPTION.


        """

        # Start parameters
        if start_option == 'intWAVECHG' or start_option == 1:
            if not _mswindows:
                if self.wavecar_path:
                    self.writefile(WAVECAR='WAVECAR')
                    if self.chgcar_path:
                        self.writefile(CHGCAR='CHGCAR')
                        start_option = 11
                    else:
                        start_option = 12
                else:
                    if self.chgcar_path:
                        self.writefile(CHGCAR='CHGCAR')
                        start_option = 13
                    else:
                        start_option = -1
            else:
                start_option = 11
        elif start_option == 'intWAVE' or start_option == 2:
            if not _mswindows:
                if self.wavecar_path:
                    self.writefile(WAVECAR='WAVECAR')
                    start_option = 12
                else:
                    start_option = -1
            else:
                start_option = 12
        elif start_option == 'intCHG' or start_option == 3:
            if not _mswindows:
                if self.chgcar_path:
                    self.writefile(CHGCAR='CHGCAR')
                    start_option = 13
                else:
                    start_option = -1
            else:
                start_option = 13

        if start_option == 'extWAVECHG' or start_option == 11:
            self.incar['ISTART'] = 1
            self.incar['ICHARG'] = 1
        elif start_option == 'extWAVE' or start_option == 12:
            self.incar['ISTART'] = 1
            self.incar['ICHARG'] = 0
        elif start_option == 'extCHG' or start_option == 13:
            self.incar['ISTART'] = 0
            self.incar['ICHARG'] = 1
        elif start_option == 'none' or start_option == -1:
            self.incar['ISTART'] = 0
            self.incar['ICHARG'] = 2
        elif start_option == 'incar' or start_option == 0:
            pass
        else:
            Message("Invalid start option.").exitm()

        # Dipole Corrections
        if not isinstance(dipole_correction, int):
            if isinstance(dipole_correction, str):
                dipole_para_dict = {'x': 1, 'y': 2, 'z': 3, 'a': 4, 'n': -1,
                                    'X': 1, 'Y': 2, 'Z': 3, 'A': 4, 'N': -1}
                try:
                    dipole_correction = dipole_para_dict[dipole_correction[0]]
                except KeyError:
                    dipole_correction = 0
            elif dipole_correction:
                dipole_correction = 0
            else:
                dipole_correction = -1
        if dipole_correction in [1, 2, 3, 4]:
            self.incar['LDIPOL'] = True
            self.incar['IDIPOL'] = dipole_correction
        elif dipole_correction < 0:
            self.incar['LDIPOL'] = False

        # Van der Waals Corrections
        if not isinstance(vdw, int):
            if isinstance(vdw, str):
                if 'bj' in vdw or 'BJ' in vdw:
                    vdw = 12
                elif 'd3' in vdw or 'D3' in vdw:
                    vdw = 11
                elif 'd2' in vdw or 'D2' in vdw:  # NOT SUGGESTED
                    vdw = 10
                elif vdw[0] in ('n', 'N'):
                    vdw = -1
                else:
                    vdw = 0
            elif vdw:
                vdw = 0
            else:
                vdw = -1
        if vdw in [10, 11, 12, 13]:
            self.incar['IVDW'] = vdw
        elif vdw < 0:
            self.incar['IVDW'] = 0

        # LDA+U Corrections
        if not isinstance(dft_u, int):
            if isinstance(dft_u, str):
                if dft_u[0] in ('e', 'E', 't', 'T', 'y', 'Y'):
                    dft_u = 1
                elif dft_u[0] in ('d', 'D', 'f', 'F', 'n', 'N'):
                    dft_u = -1
                else:
                    dft_u = 0
            elif dft_u:
                dft_u = 1
            else:
                dft_u = -1
        if dft_u > 0:
            self.incar['LDAU'] = True
            self.incar['LDAUTYPE'] = 2
            self.incar['LMAXMIX'] = 0
            self.incar['LDAUL'] = []
            self.incar['LDAUU'] = []
            self.incar['LDAUJ'] = []

            for ele in self.ion_species:
                if ele in F_BLOCK:
                    self.incar['LDAUL'].append(3)
                    self.incar['LMAXMIX'] = 6
                elif ele in D_BLOCK:
                    self.incar['LDAUL'].append(2)
                    if self.incar['LMAXMIX'] < 6:
                        self.incar['LMAXMIX'] = 4
                else:
                    self.incar['LDAUL'].append(-1)

                try:
                    self.incar['LDAUU'].append(u_values_dict[ele])
                except KeyError:
                    try:
                        self.incar['LDAUU'].append(U_VALUES[ele])
                    except KeyError:
                        self.incar['LDAUU'].append(0.0)
                finally:
                    self.incar['LDAUJ'].append(0.0)
        elif dft_u < 0:
            self.incar['LDAU'] = False
            self.incar.erase('LDAUTYPE')
            self.incar.erase('LDAUL')
            self.incar.erase('LDAUU')
            self.incar.erase('LDAUJ')

    @property
    def ion_species(self):
        if not self.poscar:
            Message('POSCAR is needed to get species of ions.').exitm()
        ion_species = []
        for site in self.poscar.sites:
            if not ion_species or not site.species_string == ion_species[-1]:
                ion_species.append(site.species_string)
        return ion_species

    @property
    def ion_numbers(self):
        if not self.poscar:
            Message('POSCAR is needed to get numbers of ions.').exitm()
        ion_numbers = []
        last_site = ''
        for site in self.poscar.sites:
            if not ion_numbers or not site.species_string == last_site:
                last_site = site.species_string
                ion_numbers.append(0)
            ion_numbers[-1] = ion_numbers[-1] + 1
        return ion_numbers

    def check_error(self):
        assert self.incar and self.outcar

        if not self.outcar.elect_converged:
            raise eh.NotConvergedError(count=self.outcar.num_elect_inconvergence)

        if self.outcar.is_normal_terminated:
            return None
        elif self.outcar.error_info:
            if 'ZBRENT' in self.outcar.error_info:
                raise eh.ZbrentError
            elif 'ZHEGV' in self.outcar.error_info:
                raise eh.ZhegvError
            elif 'FEXCP' in self.outcar.error_info:
                raise eh.FexcpError
            elif 'reading plane wave' in self.outcar.error_info:
                raise eh.ReadWavecarError
            else:
                raise eh.VaspJobError(self.outcar.error_info)
        else:
            raise eh.VaspInterruptError

    if not _mswindows:

        @rarepyth_system
        def __num_electrons(self, path):
            if not self.potcar:
                Message(
                    'POTCAR is needed to get number of valence electrons.'
                ).exitm()
            num_electrons = 0
            for i, pot in enumerate(self.potcar._list):
                assert pot[1].split('_')[0] == self.ion_species[i]
                if pot[0] == 'PAW_LDA':
                    path_src = path['lda']
                elif pot[0] == 'PAW_PBE':
                    path_src = path['pbe']
                else:
                    Message("Invalid functional type").exitm()
                pattern = r'ZVAL\s*=\s*(\d+\.\d+)'
                matches = re.findall(
                    pattern, runcmd('grep ZVAL {}/{}/POTCAR'
                                    .format(path_src, pot[1])))
                zval = int(float(matches[0]))
                num_electrons = num_electrons + zval * self.ion_numbers[i]
            return num_electrons

        @property
        def num_electrons(self):
            return self.__num_electrons(self)

        def gene_kpoints(self,
                         KMesh_resolved_value,
                         scheme='MP'):
            # NOT FINISHED
            self.poscar.to('{}/POSCAR'.format(self._cache_path), fmt='poscar')
            if scheme[0] in ('M', 'm'):
                runcmd(
                    "(echo 102; echo 1; echo {}) | vaspkit > /dev/null"
                    .format(KMesh_resolved_value), cwd=self._cache_path)
            elif scheme[0] in ('G', 'g'):
                runcmd(
                    "(echo 102; echo 2; echo {}) | vaspkit > /dev/null"
                    .format(KMesh_resolved_value), cwd=self._cache_path)
            else:
                Message("Unrecongnised KPOINTS scheme").exitm()
            self.readfile(KPOINTS='{}/KPOINTS'.format(self._cache_path))

        def run(
            self,
            popen=False,
            check_error=True,
            gamma_only=False,
            KPAR=8,
            NCORE=4,
            refresh_hosts=True,
            **kwargs
        ):
            """Run a vasp job using paras and structure of the object.


            Parameters
            ----------
            popen: bool, optional, default=False
                Whether return sp.Popen object or wait until job completes.
            check_SCF: bool, optional, default=True
                Whether to check SCF is converged or not. Won't work when
                popen=True. If enabled, raise MessageError when not converged.

            Returns
            -------
            subprocess.Popen object if popen=True,
            Nothing if popen=False.

            """

            if not (self.incar and self.poscar
                    and self.kpoints and self.potcar):
                Message("Necessary file not found").exitm()

            try:
                num_cpus = int(os.environ['SLURM_CPUS_PER_TASK']) * int(os.environ['SLURM_TASKS_PER_NODE'])
            except KeyError:
                num_cpus = int(re.findall(
                    r'NumCPUs=(\d*)',
                    runcmd("scontrol show job {} | grep NumCPUs"
                           .format(os.getenv('SLURM_JOB_ID')))
                )[0])

            if gamma_only:
                vasp = 'vasp_gam'
                self.incar['KPAR'] = 1
                self.incar['NCORE'] = num_cpus
            else:
                vasp = 'vasp_std'
                kpoints_count = self.kpoints.subdivisions[0] * self.kpoints.subdivisions[1] * self.kpoints.subdivisions[2]
                # Calculate KPAR: It MUST be a divisor of num_cpus
                ideal_kpar = min(KPAR, kpoints_count)
                actual_kpar = 1
                for i in range(ideal_kpar, 0, -1):
                    if num_cpus % i == 0:
                        actual_kpar = i
                        break

                self.incar['KPAR'] = actual_kpar

                # Calculate NCORE: It MUST be a divisor of tasks_per_group
                tasks_per_group = num_cpus // actual_kpar
                ideal_ncore = min(NCORE, tasks_per_group)
                actual_ncore = 1
                for i in range(ideal_ncore, 0, -1):
                    if tasks_per_group % i == 0:
                        actual_ncore = i
                        break

                self.incar['NCORE'] = actual_ncore

            self.writefile(
                INCAR='INCAR',
                POSCAR='POSCAR',
                KPOINTS='KPOINTS',
                POTCAR='POTCAR'
            )

            if popen:
                return sp.Popen("mpirun -n {} {}".format(num_cpus, vasp),
                                shell=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)

            else:
                runcmd("mpirun -n {} {}".format(num_cpus, vasp), print_error=False)

                self.oszicar = Oszicar.read()
                self.outcar = Outcar.read()

                if check_error:
                    self.check_error()

        def try_scf(
            self,
            **kwargs
        ):
            """Try perform SCF calculation using paras and structure of the
                object. WILL change INCAR parameters of the object. Raise
                NotConvergedError when all attempts failed.


            Parameters
            ----------
            NELM: int, optional, deafult=60
                Electronic steps taken between twice SCF checks.
            read_initio_calculate: bool, optional, default=False
                Whether to read CHGCAR and WAVECAR info of the object.
            cost: str, optional, default='medium'
                How many sets of parameters, should be tested to achieve
                convergence.\n
                'low': 7 sets\n
                'medium': 11 sets\n
                'high': 20 sets

            Returns
            -------
            No returns.

            """

            ScfCalculator(self,
                          **kwargs).start()

        def try_relax(
            self,
            **kwargs
        ):
            OptimizeCalculator(self,
                               **kwargs).start()

        def get_free_energy(
            self,
            **kwargs
        ):
            FreeEnergyCalculator(self,
                                 **kwargs).start()


class ScfCalculator:
    def __init__(
        self,
        vaspjob,
        NELM_per_try=60,
        print_level=2,
        pre_set='moderate',
        read_initio_calculate=False,
        gamma_only=False,
        KPAR=8,
        NCORE=4,
        refresh_hosts=True
    ):
        self.job = vaspjob
        if read_initio_calculate:
            self.read_ini_calc = 'intWAVECHG'
        else:
            self.read_ini_calc = 'none'

        # Tune INCAR sets into proper condition
        self.job.incar['ADDGRID'] = True
        self.job.incar['NSW'] = 0

        self.job.incar.erase('IBRION')
        self.job.incar.erase('NFREE')
        if set(self.job.poscar.labels) & F_BLOCK:
            self.job.incar['NBANDS'] = int(max(
                2 * self.job.poscar.num_sites + self.job.num_electrons / 2,
                0.6 * self.job.num_electrons))
        else:
            self.job.incar.erase('NBANDS')

        # Record target SIGMA
        self.target_sigma = (self.job.incar['SIGMA']
                             if 'SIGMA' in self.job.incar.keys() else 0.2)

        # Generate Parameters List
        if 'ALGO' in self.job.incar.keys():
            if (self.job.incar['ALGO'][0] in ('N', 'n')):
                self.job.incar['ALGO'] = 'Normal'
            if (self.job.incar['ALGO'][0] in ('F', 'f')):
                self.job.incar['ALGO'] = 'Fast'
            if (self.job.incar['ALGO'][0] in ('V', 'v')):
                self.job.incar['ALGO'] = 'VeryFast'
            if (self.job.incar['ALGO'][0] in ('C', 'c', 'A', 'a')):
                self.job.incar['ALGO'] = 'Conjugate'
        else:
            self.job.incar['ALGO'] = 'Normal'
        if 'AMIX' not in self.job.incar.keys():
            self.job.incar['AMIX'] = 0.4
        if 'BMIX' not in self.job.incar.keys():
            self.job.incar['BMIX'] = 1.0

        para_set_moderate = [
            ('Fast', 0.4, 1.0),       # Standard Pulay mixing (fastest, bulk non-magnetic)
            ('Fast', 0.2, 0.0001),    # Kerker mixing (good for magnetic/metals)
            ('Normal', 0.4, 1.0),     # Davidson with Pulay (standard robust fallback)
            ('Normal', 0.2, 0.0001),  # Davidson with Kerker (reliable for slabs/surfaces)
            ('Normal', 0.08, 0.0001)  # Damped Kerker (for difficult magnetic configurations)
        ]

        para_set_extreme = [
            ('Normal', 0.08, 0.0001),  # The statistical champion for rare-earth surface systems
            ('Normal', 0.04, 0.0001),  # Stronger dampening for severe sloshing
            ('Normal', 0.4, 1.0),
            ('Normal', 0.2, 0.0001),  # Standard Kerker (if the system happens to be well-behaved)
            ('Normal', 0.02, 0.0001),  # Extreme dampening
            ('Conjugate', 0.08, 0.0001),  # Conjugate gradient paired with Kerker mixing
            ('Conjugate', 0.04, 0.0001)  # Maximum stability, highest computational cost
        ]

        if pre_set[0] in ('m', 'M'):
            para_set = para_set_moderate[:11]
        elif pre_set[0] in ('e', 'E'):
            para_set = para_set_extreme[:7]
        else:
            Message("Invalid cost parameter.").exitm()
        if read_initio_calculate:
            para_set_ini = (self.job.incar['ALGO'],
                            self.job.incar['AMIX'],
                            self.job.incar['BMIX'])
            if para_set_ini in para_set:
                para_set.remove(para_set_ini)
            para_set.insert(0, para_set_ini)
        self.para_list = para_set

        # Set variants used in calculation
        self.tmp_vjz_list = []
        self.NELM = NELM_per_try
        self.scf_converged = False
        self._well_ini_found = False
        self._best_diff = np.inf
        self.tmp_vjz_list.append(self.job.save_checkpoint())
        self.print_level = print_level
        self.gamma_only = gamma_only
        self.KPAR = KPAR
        self.NCORE = NCORE
        self.refresh_hosts = refresh_hosts

    def __del__(self):
        for file in self.tmp_vjz_list:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    def scf_with_NELM_check(self, read_ini_calc, tol_left=0):
        self.job.incar['NELM'] = 3 * self.NELM
        stop = False
        tol = tol_left
        if not self._well_ini_found:
            self.job.set_calc_details(start_option=read_ini_calc)
        else:
            self.job.set_calc_details(start_option='intWAVE')
        runcmd(["rm", "OUTCAR", "OSZICAR", "STOPCAR"], print_error=False)
        pop = self.job.run(gamma_only=self.gamma_only,
                           popen=True,
                           KPAR=self.KPAR,
                           NCORE=self.NCORE,
                           refresh_hosts=self.refresh_hosts)
        electronic_step = 0
        time.sleep(3)
        while pop.poll() is None:
            time.sleep(3)
            try:
                self.job.oszicar = Oszicar.read()
                current_step = self.job.oszicar.num_elect_steps[-1]
            except Exception:
                continue
            if current_step > electronic_step:
                electronic_step = current_step
                # Check for severely bad results
                if current_step >= 0.5 * self.NELM:
                    if self.job.oszicar.elect_energies[-1][-1] > 0:
                        stop = True
                if current_step >= self.NELM:
                    mean_dE = np.mean(np.log10(np.abs(self.job.oszicar.elect_energies_diff[-1][-5:])))
                    if mean_dE > 2.0:
                        stop = True
                    if mean_dE > 0.0 and self.job.oszicar.elect_energies[-1][-1] / self.job.poscar.num_sites > -3.0:
                        stop = True
                # Check for converging trends
                if current_step > self.NELM and not (current_step % self.NELM):
                    energies = np.array(self.job.oszicar.elect_energies[-1])[-self.NELM:].flatten()
                    x = np.arange(self.NELM)
                    slope_abs = np.polyfit(x, np.abs(energies), 1)[0]
                    slope_raw = np.polyfit(x, energies, 1)[0]
                    if slope_abs >= 0.0 and slope_raw >= 0.0:
                        tol += 1
                    if (get_monoto_score(energies) < 0.8 and get_range_score(energies) < 0.5):
                        tol += 1
                if stop or tol > 3:
                    Message(
                        "Severe trend detected, abort the SCF."
                    ).printm()
                    with open("STOPCAR", mode='w') as file:
                        file.write("LABORT = .TRUE.")
                    pop.wait()
        # Check errors and choose next operation
        if self.print_level > 1:
            print(runcmd(['cat', 'OSZICAR']))
            sys.stdout.flush()
        if stop or self.job.oszicar.elect_energies[-1][-1] > 0:
            self.scf_converged = False
            return
        else:
            self.job.outcar = Outcar.read()
            try:
                self.job.check_error()
            except eh.ReadWavecarError:
                self.scf_with_NELM_check('intCHG')
            except eh.NotConvergedError:
                pass
            else:
                self.scf_converged = True
                self._well_ini_found = True
                self.job.readfile(CHGCAR='CHGCAR', WAVECAR='WAVECAR')
                self.tmp_vjz_list.append(self.job.save_checkpoint())
                return
        # readfile or update any well initio guess
        if not self._well_ini_found:
            self._well_ini_found = True
            self.job.readfile(CHGCAR='CHGCAR', WAVECAR='WAVECAR')
            self.tmp_vjz_list.append(self.job.save_checkpoint())
        elif mean_dE < self._best_diff:
            self._best_diff = mean_dE
            self.job.readfile(CHGCAR='CHGCAR', WAVECAR='WAVECAR')
            self.tmp_vjz_list.append(self.job.save_checkpoint())
        if tol <= 3:
            self.scf_with_NELM_check('intWAVE', tol_left=tol)

    def start(self):
        if self.print_level:
            Message("Start performing supervised SCF calculation").printm()

        for stage in range(2):
            if self.print_level:
                if stage:
                    Message("Found a good initio guess, increase accruacy.").printm()
                else:
                    Message("Try to find a good initio guess.").printm()

            for para in self.para_list:
                if self.print_level:
                    Message("Using ALGO = {0[0]}, AMIX = {0[1]}, BMIX = {0[2]}"
                            .format(para)).printm()
                self.job.incar['ALGO'] = para[0]
                self.job.incar['AMIX'] = para[1]
                self.job.incar['BMIX'] = para[2]

                try:
                    sigma = self.target_sigma + 2 * 0.2
                    self.job.incar['SIGMA'] = sigma
                    if self.print_level:
                        Message("Using SIGMA = {}".format(sigma)).printm()
                    self.scf_with_NELM_check(self.read_ini_calc)
                    if not self.scf_converged:
                        continue
                    while sigma > 0.2:
                        sigma = round(sigma - 0.2, 2)
                        self.job.incar['SIGMA'] = sigma
                        if self.print_level:
                            Message("Using SIGMA = {}".format(sigma)).printm()
                        self.scf_with_NELM_check('extWAVECHG')
                        if not self.scf_converged:
                            break
                except eh.VaspJobError:
                    self.scf_converged = False
                else:
                    if self.scf_converged:
                        if self.print_level:
                            Message("Supervised SCF success.").printm()
                        self.job.readfile(CHGCAR='CHGCAR', WAVECAR='WAVECAR')
                        return self.job
                    elif not stage and self._well_ini_found:
                        break
                    else:
                        self.load_checkpoint()

            if not self._well_ini_found:
                break

        for file in self.tmp_vjz_list:
            os.remove(file)
        raise eh.NotConvergedError


class OptimizeCalculator:
    def __init__(
        self,
        vaspjob,
        NSW_per_try=50,
        NELM_per_step=200,
        relax_directly=False,
        read_initio_calculate=True,
        start_from_gamma=False,
        KPAR=8,
        NCORE=4,
        refresh_hosts=True
    ):
        self.job = vaspjob
        self.NSW = NSW_per_try
        self.NELM_per_step = NELM_per_step
        self.bug_tol = 3

        para_set = [(2, 0.5), (2, 0.2), (1, 0.2)]
        if 'IBRION' in self.job.incar.keys():
            if self.job.incar['IBRION'] not in (1, 2):
                self.job.incar.erase('IBRION')
                self.job.incar.erase('POTIM')
            if 'POTIM' in self.job.incar.keys():
                para_set = [(self.job.incar['IBRION'],
                             self.job.incar['POTIM'])]
                if self.job.incar['IBRION'] == 1:
                    if self.job.incar['POTIM'] > 0.2:
                        para_set.append((1, 0.2))
                if self.job.incar['IBRION'] == 2:
                    if self.job.incar['POTIM'] > 0.5:
                        para_set.append((2, 0.5))
                    if self.job.incar['POTIM'] > 0.2:
                        para_set.append((2, 0.2))
                    if self.job.incar['POTIM'] < 0.2:
                        para_set.append((1, self.job.incar['POTIM']))
                    else:
                        para_set.append((1, 0.2))
        self.para_list = para_set
        self.job.incar.erase('NFREE')

        self.relax_directly = relax_directly
        self.read_initio_calculate = read_initio_calculate
        self.start_from_gamma = start_from_gamma
        if start_from_gamma:
            self.kpoints_target = self.job.kpoints.copy()
        self.KPAR = KPAR
        self.NCORE = NCORE
        self.refresh_hosts = refresh_hosts
        self.list_energy = []
        self.list_forces_max = []

    def relax_with_SCF_check(self, IBRION, POTIM):
        ionic_steps = 0
        while ionic_steps < self.NSW:
            self.job.incar['NSW'] = self.NSW - ionic_steps
            self.job.incar['IBRION'] = IBRION
            self.job.incar['POTIM'] = POTIM
            self.job.incar['NELM'] = self.NELM_per_step
            self.job.set_calc_details(start_option='intWAVE')  # intWAVECHG may cause bug
            runcmd(["rm", "OUTCAR", "OSZICAR"], print_error=False)
            pop = self.job.run(popen=True,
                               KPAR=self.KPAR,
                               NCORE=self.NCORE,
                               refresh_hosts=self.refresh_hosts)
            while pop.poll() is None:
                time.sleep(10)
                try:
                    self.job.outcar = Outcar.read()
                except FileNotFoundError:
                    continue

                current_step = self.job.outcar.num_ionic_steps
                if current_step > ionic_steps + self.job.incar['NSW'] - self.NSW:
                    print(current_step)
                    sys.stdout.flush()
                    ionic_steps = current_step + self.NSW - self.job.incar['NSW']

                    try:
                        self.job.check_error()
                    except eh.NotConvergedError as e:
                        if e.count >= 3:
                            Message(
                                "SCF convergence failed, aborting loop"
                            ).printm()
                            with open("STOPCAR", mode='w') as file:
                                file.write("LABORT = .TRUE.")
                            pop.wait()
                    except (eh.VaspInterruptError, eh.ZbrentError):
                        pass

            self.job.outcar = Outcar.read()
            self.list_energy += Oszicar.read().ionic_energies
            self.list_forces_max += Outcar.read().get_max_filtered_forces(
                slt_dynamics=self.job.poscar.site_properties.get('selective_dynamics', None))
            self.job.readfile(CHGCAR='CHGCAR', WAVECAR='WAVECAR')
            last_struct = self.job.outcar.list_relaxtion_structures[-1]
            self.job.poscar = Structure(last_struct.lattice,
                                        last_struct.species,
                                        last_struct.frac_coords,
                                        site_properties=self.job.poscar.site_properties,
                                        labels=self.job.poscar.labels)
            try:
                self.job.check_error()
            except (eh.VaspInterruptError, eh.ZbrentError) as e:
                if self.bug_tol:
                    self.bug_tol -= 1
                else:
                    raise e
            except eh.NotConvergedError:
                pass

            if self.job.outcar.ionic_converged:
                return True
            elif ionic_steps < self.NSW:
                ionic_steps -= 1
                self.list_energy.pop()
                self.list_forces_max.pop()
                ScfCalculator(self.job,
                              read_initio_calculate=True,
                              KPAR=self.KPAR,
                              NCORE=self.NCORE,
                              refresh_hosts=self.refresh_hosts).start()
            else:
                return False

    def relax_with_kpoints(self):
        if self.relax_directly and self.job.wavecar_path:
            pass
        else:
            ScfCalculator(self.job,
                          read_initio_calculate=self.read_initio_calculate,
                          print_level=1,
                          KPAR=self.KPAR,
                          NCORE=self.NCORE,
                          refresh_hosts=self.refresh_hosts).start()

        for IBRION, POTIM in self.para_list:
            Message("Try to relax with IBRION={}, POTIM={}"
                    .format(IBRION, POTIM)).printm()

            while not self.relax_with_SCF_check(IBRION, POTIM):
                x = np.arange(self.NSW)
                energies = np.array(self.list_energy[-self.NSW:]).flatten()
                slope_abs = np.polyfit(x, np.abs(energies), 1)[0]
                slope_raw = np.polyfit(x, energies, 1)[0]
                if slope_abs >= 0.0 and slope_raw >= 0.0:
                    break
                if (get_monoto_score(energies) < 0.8 and get_range_score(energies) < 0.5):
                    break
            else:
                return True
        else:
            return False

    def start(self):
        # forces check not used
        if self.start_from_gamma:
            self.job.gene_kpoints(0)
            Message(
                "Performing supervised ionic relaxation calculation on gamma point"
            ).printm()
            if not self.relax_with_kpoints():
                Message("Ionic steps converge failed.").exitm()
            else:
                Message("Gamma-only relaxation success.").printm()
                self.job.poscar.to('POSCAR_RELAXED_GAM.vasp', fmt='poscar')
                self.job.kpoints = self.kpoints_target.copy()

        Message(
            "Start performing supervised ionic relaxation calculation"
        ).printm()

        if not self.relax_with_kpoints():
            Message("Ionic steps converge failed.").exitm()
        else:
            Message("Supervised ionic relaxation success.").printm()
            self.job.poscar.to('POSCAR_RELAXED.vasp', fmt='poscar')


class FreeEnergyCalculator:
    def __init__(
        self,
        vaspjob,
        read_initio_calculate=True,
        model='adsorbate',
        temprature=298,
        pressure=1.0,
        spin_multiplicity=1,
        NELM_per_step=300,
        refresh_hosts=True
    ):
        self.job = vaspjob
        self.read_initio_calculate = read_initio_calculate
        self.model = model
        self.temprature = temprature
        self.pressure = pressure
        self.spin_multiplicity = spin_multiplicity
        self.NELM_per_step = NELM_per_step
        self.refresh_hosts = refresh_hosts

    def get_electonic_energy(self):
        if 'EDIFF' not in self.job.incar.keys() or self.job.incar['EDIFF'] > 1e-06:
            self.job.incar['EDIFF'] = 1e-06
        Message('Calculating Electronic energy.').printm()
        ScfCalculator(self.job,
                      read_initio_calculate=self.read_initio_calculate,
                      print_level=0,
                      refresh_hosts=self.refresh_hosts).start()
        self.job.readfile(CHGCAR='CHGCAR',
                          WAVECAR='WAVECAR')
        self.electonic_energy = Oszicar.read().ionic_energies[-1]
        Message('Electronic energy: {} eV'.format(self.electonic_energy)).printm()
        return self.electonic_energy

    def start(self):
        self.get_electonic_energy()
        self.job.incar['ADDGRID'] = False
        self.job.incar['NELM'] = self.NELM_per_step
        self.job.incar['NSW'] = 1
        self.job.incar['IBRION'] = 5
        self.job.incar['POTIM'] = 0.015
        self.job.incar['NFREE'] = 2
        self.job.incar.erase('EDIFFG')
        self.job.set_calc_details(start_option='intWAVE')
        freq_calculated = 0
        runcmd(["rm", "OSZICAR"], print_error=False)
        runcmd(["rm", "OUTCAR"], print_error=False)

        pop = self.job.run(KPAR=1,
                           NCORE=1,
                           popen=True,
                           refresh_hosts=self.refresh_hosts)
        while pop.poll() is None:
            try:
                self.job.outcar = Outcar.read()
            except FileNotFoundError:
                time.sleep(5)
                continue

            freq_calculated_now = self.job.outcar.num_ionic_steps
            if freq_calculated_now > freq_calculated:
                print(freq_calculated_now)
                sys.stdout.flush()
                freq_calculated = freq_calculated_now
            time.sleep(5)
        self.job.outcar = Outcar.read()
        try:
            self.job.check_error()
        except eh.NotConvergedError:
            Message('SCF inconvergence detected, please check.').printm()
        Message("Vibration analysis success.").printm()

        if self.model[0] in ('a', 'A'):
            self.thermal_correction = float(re.findall(r'(-?\d+.\d*)\seV', runcmd(
                "(echo 501; echo {}) | vaspkit | grep 'to G(T)'"
                .format(self.temprature)))[0])
        elif self.model[0] in ('g', 'G'):
            self.thermal_correction = float(re.findall(r'(-?\d+.\d*)\seV', runcmd(
                "(echo 502; echo {}; echo {}; echo {}) | vaspkit | grep -a 'to G(T)'"
                .format(self.temprature, self.pressure, self.spin_multiplicity)))[0])
        else:
            Message('Unrecongnised Thermal Model.').exitm()

        Message('Gibbs free energy: {} + {} = {}(eV)'.format(
            format(self.electonic_energy, '.6f'),
            format(self.thermal_correction, '.6f'),
            format(self.electonic_energy + self.thermal_correction, '.6f'),
        )).printm()
        return self.electonic_energy + self.thermal_correction


class DOSCalculator:
    def __init__(
        self,
        vaspjob,
        option='element_separate',
        projected_atoms=None,
        projected_orbitals=None
    ):
        self.job = vaspjob

    def start(self):
        # Use tetrahedron method and Gamma-Centered KPOINTS
        self.job.incar['NELM'] = 200
        self.job.incar['ISMEAR'] = -5
        pass
