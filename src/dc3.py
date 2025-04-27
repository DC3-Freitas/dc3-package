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
        self.means = self.model.means.cpu()
        self.stds = self.model.stds.cpu()

        # Mapping
        self.str_to_label = label_map
        self.label_to_str = {v: k for k, v in label_map.items()}

        # Reference vectors and delta cutoffs
        self.ref_vecs = compute_ref_vec(ref_vec_folder, self.means, self.stds)
        self.delta_cutoffs = compute_delta_cutoff(synthetic_data_folder, self.ref_vecs, self.means, self.stds)


    def calculate(self, lattice: DataCollection) -> np.ndarray:
        features = compute_feature_vectors(lattice)
        amorphous = calculate_amorphous(lattice)
        results = []

        with torch.no_grad():
            preds = self.model(features.to(self.device)).argmax(dim=1).cpu()

        for i in range(len(features)):
            if not amorphous[i]:
                ref_vec = self.ref_vecs[self.label_to_str[preds[i]]]
                normalized_feature = (features[i] - self.means) / (self.stds + 1e-6)
                dist = np.linalg.norm(normalized_feature - ref_vec)

                if dist >= self.delta_cutoffs[self.label_to_str[preds[i]]]:
                    results.append("unknown")
                else:
                    results.append(preds[i])
            else:
                results.append("amorphous")
