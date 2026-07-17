# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 08:43:10 2024

@author: Wang Junhao
"""

import os
import re
import math
import random
from pathlib import Path

import lmdb
import yaml
import pickle
import torch
from tqdm import tqdm
from pymatgen.core import Structure
from ase.io import vasp
from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from fairchem.core.preprocessing import AtomsToGraphs

from rarepyth.utils import make_target_dir
from rarepyth.vasp.output import Outcar

PERODIC_TABLE = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                 'Tl', 'Pb', 'Bi']

NON_METAL = {'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Si', 'P',
             'S', 'Cl', 'Ar', 'As', 'Se', 'Br', 'Kr', 'Sb', 'Te', 'I', 'Xe'}


class extended_pyg_data:
    def __init__(self, pyg_data):
        self.data = pyg_data
        species = []
        for atom in pyg_data['atomic_numbers']:
            species.append(PERODIC_TABLE[int(atom) - 1])
        self.struct = Structure(
            pyg_data['cell'],
            species,
            pyg_data['pos'],
            coords_are_cartesian=True)

        self.massive_atoms = {}
        self.few_atoms = {}

        pattern = r"([A-Za-z]+)([0-9]+)"
        matches = re.findall(pattern, self.struct.formula)
        for key, value in matches:
            if int(value) >= 8 or key not in ['C', 'H', 'O', 'N']:
                self.massive_atoms[key] = int(value)
            else:
                self.few_atoms[key] = int(value)

        assert (not self.massive_atoms == {}), f"{self.struct.formula}"

        self.base_eles = set(self.massive_atoms.keys())
        self.adsorbate_eles = set(self.few_atoms.keys())

        if math.gcd(*tuple(self.massive_atoms.values())) < 3:
            for key in self.massive_atoms:
                if key in ['C', 'H', 'O', 'N']:
                    self.adsorbate_eles.add(key)

        if 'N' in self.adsorbate_eles:
            if 'C' in self.adsorbate_eles:
                self.adsorbate_type = 'CNHO'
            else:
                self.adsorbate_type = 'NHO'
        elif 'C' in self.adsorbate_eles:
            self.adsorbate_type = 'CHO'
        else:
            self.adsorbate_type = 'OH'

        for element in self.base_eles:
            if element in NON_METAL:
                self.base_type = 'compound'
                break
        else:
            self.base_type = 'alloy'


def build_from_vasp(
    vaspjob_dir,
    output_dir=None,
    output_name=None,
    tags=[],
    sid=None,
    test_set=False,
    sampling_interval=0,
    maximum_sample=None,
    readfile='OUTCAR'
):
    vaspjob_dir = os.path.realpath(vaspjob_dir)
    output_dir = os.path.realpath(output_dir) if output_dir else os.path.realpath('.')
    assert tags
    if readfile == 'OUTCAR':
        max_index = Outcar.read(OUTCAR=os.path.join(vaspjob_dir, 'OUTCAR')).num_ionic_steps
        atoms_list = []
        for index in range(max_index):
            atoms_list.append(vasp.read_vasp_out('OUTCAR', index=index))
        if sampling_interval:
            sampled_atoms_list = [0] * math.ceil(len(atoms_list) / (sampling_interval + 1))
            for idx in range(len(sampled_atoms_list)):
                sampled_atoms_list[-idx - 1] = atoms_list[-1 - (sampling_interval + 1) * idx]
            atoms_list = sampled_atoms_list
        if maximum_sample and len(atoms_list) > maximum_sample:
            atoms_list = atoms_list[:maximum_sample]
    else:
        max_index = 1
        atoms_list = [vasp.read_vasp(os.path.join(vaspjob_dir, readfile))]

    if not output_name:
        output_name = "{}.lmdb".format(os.path.basename(vaspjob_dir))

    db = lmdb.open(
        os.path.join(output_dir, output_name),
        map_size=1048576 * max_index,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    data_objects = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=not test_set,
        r_forces=not test_set,
        r_distances=False,
        r_fixed=True,
    ).convert_all(atoms_list, disable_tqdm=True)

    if not sid:
        sid = torch.LongTensor([int(10000000 * random())])
    elif not isinstance(sid, torch.LongTensor):
        sid = torch.LongTensor([sid])
    for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
        data.sid = sid
        data.fid = torch.LongTensor([fid])
        data.tags = torch.LongTensor(tags)
        # Filter data if necessary
        # FAIRChem filters adsorption energies > |10| eV and forces > |50| eV/A
        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            print("no neighbors")
            continue

        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
    txn.commit()

    db.sync()
    db.close()


def merge_specified_lmdb(
    input_files,
    output_dir,
    output_name=None
):
    output_dir = os.path.realpath(output_dir)
    make_target_dir(output_dir)
    ldses = []
    total_length = 0
    size = 0
    for file in input_files:
        assert os.path.splitext(os.path.realpath(file))[-1] == '.lmdb'
        size += os.path.getsize(os.path.realpath(file))
        lds = LmdbDataset({'src': os.path.realpath(file)})
        ldses.append(lds)
        total_length += len(lds)
    if not output_name:
        output_name = "{}_merged.lmdb".format(
            os.path.splitext(os.path.basename(file))[0])
    db_out_path = os.path.join(output_dir, output_name)
    if os.path.exists(db_out_path):
        os.remove(db_out_path)
    db_out = lmdb.open(
        db_out_path,
        map_size=(int(1.02 * size / 1048576) + 4) * 1048576,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    idx = 0
    for lds in ldses:
        for data in lds:
            txn = db_out.begin(write=True)
            txn.put(f'{idx}'.encode('ascii'), pickle.dumps(data, protocol=-1))
            txn.commit()
            db_out.sync()
            idx += 1
    db_out.close()


def merge_lmdb(
    input_path,
    output_dir,
    output_name=None
):
    input_path = os.path.realpath(input_path)
    output_dir = os.path.realpath(output_dir)
    make_target_dir(output_dir)
    ldses = []
    total_length = 0
    size = 0
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if os.path.splitext(os.path.join(root, file))[-1] == '.lmdb':
                size += os.path.getsize(os.path.join(root, file))
                lds = LmdbDataset({'src': os.path.join(root, file)})
                ldses.append(lds)
                total_length += len(lds)
    if not output_name:
        output_name = "{}_merged.lmdb".format(
            os.path.splitext(os.path.basename(input_path))[0])
    db_out_path = os.path.join(output_dir, output_name)
    if os.path.exists(db_out_path):
        os.remove(db_out_path)
    db_out = lmdb.open(
        db_out_path,
        map_size=(int(1.02 * size / 1048576) + 4) * 1048576,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    idx = 0
    for lds in ldses:
        for data in lds:
            txn = db_out.begin(write=True)
            txn.put(f'{idx}'.encode('ascii'), pickle.dumps(data, protocol=-1))
            txn.commit()
            db_out.sync()
            idx += 1
    db_out.close()


def read_lmdb_formulas(path):
    return [extended_pyg_data(data).struct.formula for data in tqdm(
        LmdbDataset({'src': path}))]


def _preparer(
        input_path,
        output_dir,
        seed
):
    random.seed(seed)
    make_target_dir(output_dir)
    size = 0
    if os.path.isfile(input_path):
        size = os.path.getsize(input_path)
        output_prefix = os.path.basename(input_path)[:-5]
    else:
        output_prefix = os.path.basename(input_path)
        for db_path in Path(input_path).glob('*.lmdb'):
            size = size + os.path.getsize(db_path)
    lds = LmdbDataset({'src': input_path})
    idxs = list(range(len(lds)))
    random.shuffle(idxs)

    return size, output_prefix, lds, idxs


def classify_lmdb_by_chem(
        input_path,
        output_dir,
        length=1000,
        seed=0
):
    input_path = os.path.realpath(input_path)
    output_dir = os.path.realpath(output_dir)
    size, output_prefix, lds, idxs = _preparer(input_path,
                                               output_dir,
                                               seed)

    db_keys = {
        'alloy_CNHO', 'alloy_NHO', 'alloy_CHO', 'alloy_OH',
        'compound_CNHO', 'compound_NHO', 'compound_CHO', 'compound_OH'
    }

    _list = []
    db_dict = {}
    for db_key in db_keys:
        _list.append(os.path.join(output_dir, "{}_{}_0.lmdb"
                     .format(output_prefix, db_key)))
        db_dict[db_key] = {
            'i': 0,
            'count': 0,
            'db': lmdb.open(
                _list[-1],
                map_size=(int(1.02 * size * length / len(lds) /
                              1048576) + 4) * 1048576,
                subdir=False,
                meminit=False,
                map_async=True,
            )}

    for idx in tqdm(idxs):
        epd = extended_pyg_data(lds[idx])
        db_key = f'{epd.base_type}_{epd.adsorbate_type}'

        if db_dict[db_key]['i'] == length:
            db_dict[db_key]['i'] = 0
            db_dict[db_key]['count'] = db_dict[db_key]['count'] + 1
            db_dict[db_key]['db'].close()
            _list.append(os.path.join(output_dir, "{}_{}_{}.lmdb"
                         .format(output_prefix,
                                 db_key, db_dict[db_key]['count'])))
            db_dict[db_key]['db'] = lmdb.open(
                _list[-1],
                map_size=(int(1.02 * size * length / len(lds) /
                              1048576) + 4) * 1048576,
                subdir=False,
                meminit=False,
                map_async=True,
            )

        txn = db_dict[db_key]['db'].begin(write=True)
        txn.put(f"{db_dict[db_key]['i']}".encode("ascii"),
                pickle.dumps(epd.data, protocol=-1))
        txn.commit()
        db_dict[db_key]['db'].sync()
        db_dict[db_key]['i'] = db_dict[db_key]['i'] + 1

    for db_key in db_keys:
        db_dict[db_key]['db'].close()

    return _list


def devide_lmdb_by_length(
        input_path,
        output_dir,
        length=1000,
        seed=0
):
    input_path = os.path.realpath(input_path)
    output_dir = os.path.realpath(output_dir)
    size, output_prefix, lds, idxs = _preparer(input_path,
                                               output_dir,
                                               seed)

    _list = []
    for piece in range(int(len(lds) / length) + 1):
        _list.append(os.path.join(output_dir,
                     "{}_{}.lmdb".format(output_prefix, piece)))
        db = lmdb.open(
            _list[-1],
            map_size=(int(1.02 * size * length / len(lds) /
                          1048576) + 4) * 1048576,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        for i, idx in enumerate(tqdm(idxs[
            int(piece * length):min(int((piece + 1) * length), len(lds))
        ])):
            txn = db.begin(write=True)
            txn.put(f'{i}'.encode('ascii'), pickle.dumps(lds[idx], protocol=-1))
            txn.commit()
            db.sync()
        db.close()

    return _list


def devide_lmdb_by_folds(
        input_path,
        output_dir,
        folds=5,
        seed=0
):
    input_path = os.path.realpath(input_path)
    output_dir = os.path.realpath(output_dir)
    size, output_prefix, lds, idxs = _preparer(input_path,
                                               output_dir,
                                               seed)

    _list = []
    for k in range(folds):
        _list.append(os.path.join(output_dir,
                     "{}_{}.lmdb".format(output_prefix, k)))
        db = lmdb.open(
            _list[-1],
            map_size=(int(1.02 * size / folds / 1048576) + 4) * 1048576,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        for i, idx in enumerate(tqdm(idxs[
            int(k * len(lds) / folds):
            int((k + 1) * len(lds) / folds)
        ])):
            txn = db.begin(write=True)
            txn.put(f'{i}'.encode('ascii'), pickle.dumps(lds[idx], protocol=-1))
            txn.commit()
            db.sync()
        db.close()

    return _list


def cut_lmdb(
        input_path,
        output_dir,
        split_point=0.8,
        seed=0
):
    input_path = os.path.realpath(input_path)
    output_dir = os.path.realpath(output_dir)
    size, output_prefix, lds, idxs = _preparer(input_path,
                                               output_dir,
                                               seed)

    idx_split = int(split_point * len(lds))

    _list = []
    _list.append(os.path.join(output_dir,
                 "{}_0.lmdb".format(output_prefix)))
    db = lmdb.open(
        _list[-1],
        map_size=(int(1.02 * size * split_point / 1048576) + 4) * 1048576,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    for i, idx in enumerate(tqdm(idxs[:idx_split])):
        txn = db.begin(write=True)
        txn.put(f'{i}'.encode('ascii'), pickle.dumps(lds[idx], protocol=-1))
        txn.commit()
        db.sync()
    db.close()

    _list.append(os.path.join(output_dir,
                 "{}_1.lmdb".format(output_prefix)))
    db = lmdb.open(
        _list[-1],
        map_size=(int(1.02 * size * (1 - split_point) / 1048576) + 4) * 1048576,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    for i, idx in enumerate(tqdm(idxs[idx_split:])):
        txn = db.begin(write=True)
        txn.put(f'{i}'.encode('ascii'), pickle.dumps(lds[idx], protocol=-1))
        txn.commit()
        db.sync()
    db.close()

    return _list


def build_sid_dict(
    dft_files_path,
    output_path=None,
    output_name=None
):
    dft_files_path = os.path.realpath(dft_files_path)
    if output_path:
        output_path = os.path.realpath(output_path)
    else:
        output_path = dft_files_path
    if not output_name:
        output_name = f'{os.path.basename(dft_files_path)}_sid.yml'
    sid_dict = {}
    for root, dirs, files in os.walk(dft_files_path):
        for struct_dir in dirs:
            if os.path.exists(os.path.join(root, struct_dir, 'SID')):
                with open(os.path.join(root, struct_dir, 'SID'), mode='r') as file:
                    sid = int(file.readline())
                sid_dict[sid] = os.path.join(root, struct_dir)
            elif os.path.exists(os.path.join(root, struct_dir, 'POSCAR')):
                sid = int(10000000 * random.random())
                sid_dict[sid] = os.path.join(root, struct_dir)
                with open(os.path.join(root, struct_dir, 'SID'), mode='w') as file:
                    file.write(str(sid))
    with open(os.path.join(output_path, output_name), mode='w') as file:
        yaml.dump(sid_dict, file)
