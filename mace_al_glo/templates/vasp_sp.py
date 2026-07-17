# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:37:51 2025

@author: Wang Junhao
"""

from rarepyth.base import runcmd
from rarepyth.vasp.compute import Kpoints, Poscar, VaspJob
from rarepyth.vasp.output import Oszicar

vj = VaspJob.from_files(INCAR='INCAR.ini')
vj.poscar = Poscar.from_file('POSCAR.vasp')
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

with open('MINIMA_DFT_EE', 'a') as file:
    file.write(f'{Oszicar.read().ionic_energies[-1]}\n')

runcmd('rm CHG CHGCAR WAVECAR *.vjt', print_error=False)
