#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=wzacnormal04
#SBATCH -J name
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1

useScript=true
scriptName="job.py"
envName="rarepyth"

source ~/software/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $envName

export MKL_DEBUG_CPU_TYPE=5
export MKL_CBWR=AVX2
export I_MPI_PIN_DOMAIN=numa

module purge
source ~/apprepo/vasp/6.4.3-optcell_intelmpi2017_wannier90_libbeef_dftd4/scripts/env.sh

if $useScript ; then
    python $scriptName
    else
    mpirun -np $SLURM_NTASKS vasp_std > log
fi
