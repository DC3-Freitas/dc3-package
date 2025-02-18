# script for running lattices in parallel (on local)
import subprocess
for i in range(10):
    subprocess.Popen(f"py -m ml_dataset.data_gen_single fcc lattice/lammps_lattices/data/fcc.gz {i}")
for i in range(10):
    subprocess.Popen(f"py -m ml_dataset.data_gen_single bcc lattice/lammps_lattices/data/bcc.gz {i}")
for i in range(10):
    subprocess.Popen(f"py -m ml_dataset.data_gen_single sc lattice/lammps_lattices/data/sc.gz {i}")