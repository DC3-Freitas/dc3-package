import json
from md.lammps_writer import LammpsInput

with open("md/experiments_list.json") as f:
    x = f.read()
    print(x)
    data = json.loads(x)
    for exp in data:
        lammps_input = LammpsInput(**exp)
        lammps_input.write(f"md/{exp['exp_name']}.in")