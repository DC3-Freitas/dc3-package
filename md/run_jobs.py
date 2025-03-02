import os
from textwrap import dedent

runner_template = dedent("""
    #SBATCH --job-name={file_name}
    #SBATCH --ntasks-per-node=1
    #SBATCH --nodes 1
    #SBATCH --output=$md/data/logs/job_out_{file_name}.log
    #SBATCH --error=$md/data/logs/job_err_{file_name}.log
    #SBATCH --time 00:30:00
    #SBATCH --partition sched_mit_rodrigof_r8
                         
    module --force purge
    ml load cpu slurm gcc openmpi
                         
    mpirun -in {file_path} \\
           -log md/data/logs/{file_name}.log \\
           -var potentials_dir $md/potentials \\
           -var RANDOM ${{RANDOM}}
    """)

for exp_name in os.listdir("md/lammps_scripts"):
    for file_name in os.listdir(f"md/lammps_scripts/{exp_name}"):
        file_path = f"md/{exp_name}/{file_name}"
        runner_content = runner_template.format(file_name=file_name, file_path=file_path)
        print(runner_content)
        print("-----------------------------")

