#!/bin/bash
#SBATCH --job-name=LATTICE_GEN
#SBATCH --ntasks 16
#SBATCH --nodes 1
#SBATCH --time 1:00:00

#SBATCH --partition sched_mit_rodrigof_r8
#SBATCH --mem-per-cpu=8000

source ~/.bashrc
conda activate lammps_env

for i in "bcc" "fcc" "sc"
do
    lmp -in ${i}.lmp
done