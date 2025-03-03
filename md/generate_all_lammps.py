import numpy as np
import json
from md.lammps_writer import LammpsInput

SIM_TEMPERATURE_FRACTIONS = np.round(np.arange(0.04, 1.60 + 0.04, 0.04), 6)

with open("md/experiments_list.json") as f:
    x = f.read()
    print(x)
    data = json.loads(x)

    for exp in data:
        # Generate data at each simulation temperature fraction
        lammps_input = LammpsInput(**exp)

        for frac in SIM_TEMPERATURE_FRACTIONS:
            lammps_input.write(frac, f"md/lammps_scripts/{exp['exp_name']}/{exp['exp_name']}_{frac}.in")