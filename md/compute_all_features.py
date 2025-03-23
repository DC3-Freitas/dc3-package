from features.compute_all import compute_feature_vectors
import ovito
import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys

N_USE = 17000

def compute_feature_for_simulation(file_info):
    # Random seed
    np.random.seed(42)

    # Get information about the file
    exp_name, file_name = file_info

    # Get T/T_m value
    sim_temp_id = file_name[5:-3]

    # Get lattices
    file_path = f"md/data/{exp_name}/{file_name}"
    pipeline = ovito.io.import_file(file_path)

    # Iterate over each saved lattice
    num_frames = pipeline.source.num_frames
    features = []

    for frame in range(num_frames):
        lattice = pipeline.compute(frame)
        features.append(compute_feature_vectors(lattice))

    features = np.vstack(features)

    # Shuffle features and take first N_USE (17000)
    np.random.shuffle(features)
    os.makedirs(f"md/features/{exp_name}", exist_ok=True)
    np.savetxt(f"md/features/{exp_name}/feature_{sim_temp_id}.gz", features[:N_USE])

    if len(features) < N_USE:
        print(f"{file_path} has less than {N_USE} features!")


if __name__ == '__main__':
    # Only proccess selected experiments
    selected_experiments = None

    if len(sys.argv) > 1:
        selected_experiments = sys.argv[1:]
    
    # Get all (exp name, file name) that we want to proccess
    sims_to_process = []

    for exp_name in os.listdir("md/data"):
        if selected_experiments and exp_name not in selected_experiments:
            continue
        for file_name in os.listdir(f"md/data/{exp_name}"):
            # Don't use relaxed data or logs
            if "relaxed" in file_name or exp_name == "logs":
                continue
            # Add to list of things we want to proccess
            sims_to_process.append((exp_name, file_name))

    # Parallelize
    num_workers = min(
        int(os.getenv('SLURM_CPUS_ON_NODE', multiprocessing.cpu_count())), 
        len(sims_to_process)
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(compute_feature_for_simulation, sims_to_process)