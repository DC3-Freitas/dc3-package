import torch
import numpy as np
from ml.model import MLP_Model
from ovito.data import DataCollection
from compute_features.compute_all import compute_feature_vectors
from outlier.coherence import calculate_amorphous
from outlier.outlier_cutoffs import compute_ref_vec, compute_delta_cutoff

class DC3:
    def __init__(
        self,
        model_path: str,
        label_map: dict[str, int],
        ref_vec_folder: str,
        synthetic_data_folder: str,
    ) -> None:
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP_Model()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        # Pull normalization values out from model
        self.means = self.model.means.cpu().numpy()
        self.stds = self.model.stds.cpu().numpy()

        # Mapping
        self.label_str_to_int = label_map
        self.label_int_to_str = {v: k for k, v in label_map.items()}

        # Reference vectors and delta cutoffs
        self.ref_vecs = compute_ref_vec(ref_vec_folder, self.means, self.stds)
        self.delta_cutoffs = compute_delta_cutoff(synthetic_data_folder, self.ref_vecs, self.means, self.stds)


    def calculate(self, lattice: DataCollection) -> np.ndarray:
        features = compute_feature_vectors(lattice)
        amorphous = calculate_amorphous(lattice)
        results = []

        with torch.no_grad():
            preds = self.model(torch.from_numpy(features).float().to(self.device)).argmax(dim=1).cpu().numpy()

        for i in range(len(features)):
            if not amorphous[i]:
                ref_vec = self.ref_vecs[self.label_int_to_str[preds[i]]]
                normalized_feature = (features[i] - self.means) / (self.stds + 1e-6)
                dist = np.linalg.norm(normalized_feature - ref_vec)

                if dist >= self.delta_cutoffs[self.label_int_to_str[preds[i]]]:
                    results.append("unknown")
                else:
                    results.append(self.label_int_to_str[preds[i]])
            else:
                results.append("amorphous")
        
        return results


tester = DC3("ml/models/model_2025-04-26_23-04-46.pt", 
             {"bcc": 0, "cd": 1, "fcc": 2, "hcp": 3, "hd": 4, "sc": 5}, 
             "lattice/features", "ml_dataset/features")

print("Done initializing")

import ovito
pipeline = ovito.io.import_file("dump_1.44_relaxed.gz") # mg hcp
lattice = pipeline.compute(0)
# calculate_amorphous(lattice)
# tester.calculate(lattice)
print(tester.calculate(lattice))