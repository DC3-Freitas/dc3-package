import os
from textwrap import dedent
import subprocess
import sys

runner_template = dedent("""\
    #!/bin/bash
    #SBATCH --job-name={file_name}
    #SBATCH --ntasks-per-node=1
    #SBATCH --nodes 1
    #SBATCH --output=md/data/logs/{file_name}_job_out.log
    #SBATCH --error=md/data/logs/{file_name}_job_err.log
    #SBATCH --time 00:30:00
    #SBATCH --partition sched_mit_rodrigof_r8
    
    # Load any necessary modules
    source ~/.bashrc
    # Activate your Conda environment
    conda activate lammps_env
        
    mpirun lmp -in {file_path} \\
               -log md/data/logs/{file_name}_lammps.log \\
               -var potentials_dir md/potentials
    """)

selected_experiments = None

if len(sys.argv) > 1:
    selected_experiments = sys.argv[1:]
    
for exp_name in os.listdir("md/lammps_scripts"):
    if selected_experiments and exp_name not in selected_experiments:
        continue
    for file_name in os.listdir(f"md/lammps_scripts/{exp_name}"):
        file_path = f"md/lammps_scripts/{exp_name}/{file_name}"
        runner_content = runner_template.format(file_name=file_name, file_path=file_path)

        try:
            result = subprocess.run(["sbatch"], input=runner_content, text=True, capture_output=True, check=True)
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("Error submitting job:")
            print(e.stderr)
