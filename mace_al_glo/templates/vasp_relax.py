# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:21:22 2025

@author: Wang Junhao
"""

import os
from rarepyth.base import runcmd
from rarepyth.vasp.compute import Kpoints, Poscar, VaspJob
from rarepyth.mace.data_io import build_from_vasp

if __name__ == "__main__":

    vj = VaspJob.from_files(INCAR='INCAR.ini')

    vj.poscar = Poscar.from_file('POSCAR.vasp')
    vj.poscar.get_tags_by_layers()
    vj.poscar.read_tags_from_file(TAG='TAG')
    vj.poscar.fix_atoms_by_tags(fixed_tags=[0, 1])  # Only relax adsorped molecules

    vj.incar['ISPIN'] = 2
    vj.incar['EDIFF'] = 1e-05
    vj.kpoints = Kpoints('comment\n', 'Gamma',
                         [max(int(30 / vj.poscar.lattice.a), 1),
                          max(int(30 / vj.poscar.lattice.b), 1),
                          1],
                         [0.0, 0.0, 0.0])
    vj.gene_potcar(option='f-electron')
    vj.set_calc_details(dft_u='true')
    vj.try_scf(pre_set='extreme')

    vj.incar['NSW'] = 250
    vj.incar['IBRION'] = 2
    vj.incar['POTIM'] = 0.5
    vj.run(check_error=False)

    # Extract tags from TAG file to include in the extxyz dataset
    tags_list = None
    if os.path.exists('TAG'):
        with open('TAG', 'r', encoding='utf-8') as f:
            tags_list = [int(line.split()[1]) for line in f.readlines() if line.strip()]

    # Replace singularity container call with direct mace_data_io method
    build_from_vasp(
        vaspjob_dir='.',
        output_dir='.',
        output_name='traj.xyz',
        tags=tags_list,
        sampling_interval=4,
        maximum_sample=50,
    )

    runcmd('rm CHG CHGCAR WAVECAR *.vjt', print_error=False)
