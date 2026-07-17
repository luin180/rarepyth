#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:50:11 2024

@author: Wang Junhao
"""

import os
from rarepyth.base import runcmd
from rarepyth.vasp.compute import VaspJob
from rarepyth.utils import fix_atoms_by_height


if __name__ == "__main__":

    vj = VaspJob.from_files(INCAR='INCAR.ini',
                            POSCAR='POSCAR_RELAXED.vasp',
                            CHGCAR='CHGCAR',
                            WAVECAR='WAVECAR')

    vj.incar['SYSTEM'] = runcmd("scontrol show job {} | grep JobName"
                  .format(os.getenv('SLURM_JOB_ID'))).split()[1].split('=')[1]

    vj.gene_kpoints(0.04)
    vj.gene_potcar()
    
    fix_atoms_by_height(vj.poscar, 0, 0.40)
    vj.try_relax()
    
    fix_atoms_by_height(vj.poscar, 0, 0.40)
    vj.get_free_energy(model='adsorbate')
