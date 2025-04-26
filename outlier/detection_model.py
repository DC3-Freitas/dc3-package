import numpy as np 
import torch

from features.compute_all import compute_feature_vectors
from coherence import calculate_all_coherence
from ml.model import MLP_Model
from synthetic_features import compute_ref_vectors, compute_delta_99

# Import trained network

network = MLP_Model
network.eval()

#lattices
lattices = ["FCC", "BCC", "HCP", "CD", "HD", "SC"]


#distance between the feature vector and the reference vector 
def compute_distance(x_i, x_ref):
  #distance between feature vector and special feature vector
  return np.linalg.norm(x_i - x_ref)

#predict the label for an atom

def predict_label(x_i, a, a_cut, delta_99, ref_vectors):

  # Determining structure
  if a > a_cut:
    # Converting to pytorch tensor
    f_tensor = torch.tensor(x_i).float() 
    
    #make prediction
    y_pred = network(f_tensor)
    label = torch.argmax(y_pred).item()

        #ref_vec
    x_star = ref_vectors[label]

        #calculate distance 
    delta = compute_distance(x_i, x_star)

        #99th percentile detection
    if delta <= delta_99[label]:
      predicted_label = lattices[label]
    else: 
      predicted_label = "unknown structure"
  else:
    predicted_label = "liquid or amorphous"

  return predicted_label


def outlier_det(data, synthetic_data, l, N_neigh, a_cut):

    #coherence factor and feature vectors for all atoms. 
  coh_fac = calculate_all_coherence(data, l, N_neigh)
  feature_vectors = compute_feature_vectors(data, None)
  predicted_labels = []

  #ref_vector and delta_99 from synthetic
  ref_vectors = compute_ref_vectors(synthetic_data, lattices)
  delta_99 = compute_delta_99(synthetic_data, ref_vectors)

  for atom in range(len(coh_fac)):
    a = coh_fac[atom]
    x_i = feature_vectors[atom]

    predicted_label = predict_label(x_i, a, a_cut, delta_99, ref_vectors)

        #storing the last label predicted
    predicted_labels.append(predicted_label)
    
  return predicted_labels
            

