import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from ml_dataset.data_gen_single import create


def run_sim(data):
    create(*data)


if __name__ == "__main__":
    runs = []
    structs = ["fcc", "bcc", "sc", "hd", "cd", "hcp"]
    temps = np.linspace(0.01, 0.25, 10)

    for struct in structs:
        for temp in temps:
            runs.append((struct, f"lattice/md_results/{struct}.gz", temp))

    num_workers = min(
        int(os.getenv("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count())), len(runs)
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(run_sim, runs)
