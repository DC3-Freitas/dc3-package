import numpy as np
from detection_model import compute_distance 

#defining the reference vector
def compute_ref_vec(synthetic_data, lattices):
  #mean feature vector for each lattice
  return np.mean(synthetic_data, axis = 1)    

#defining the 99th percentile of synthetic data
def compute_delta_99(synthetic_data, ref_vectors, lattices):
  delta_99 = np.zeros(len(lattices))

  for label in range(len(lattices)):
    #synthetic feature vectors
    feature_vectors = synthetic_data[label]

    #compute distances to reference vector
    distances = np.linalg.norm(feature_vectors - ref_vectors[label], axis = 1)

    #compute 99th percentile
    delta_99[label] = np.percentile(distances, 99)
    
  return delta_99
