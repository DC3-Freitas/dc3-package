import torch
import numpy as np
import os
import json
from DC3.constants import SAVED_FULL_MODEL_PATH
from DC3.ml.model import MLPModel
from ovito.data import DataCollection
from DC3.compute_features.compute_all import compute_feature_vectors
from DC3.outlier.coherence import calculate_amorphous
from DC3.outlier.outlier_cutoffs import compute_ref_vec, compute_delta_cutoff
from DC3.data_handler import DataHandler
from DC3.ml_dataset.dataset import CrystalDataset
from DC3.ml.train import train


class DC3:
    def __init__(
        self,
        model: MLPModel,
        label_map: dict[str, int],
        ref_vecs: dict[str, np.ndarray],
        delta_cutoffs: dict[str, float],
    ) -> None:
        """
        Initializes a DC3 model for structure classification.

        Args:
            model: trained PyTorch MLPModel used for classification
            label_map: dictionary mapping structure names to integer class labels
            ref_vecs: dictionary mapping structure names to mean feature vectors
            delta_cutoffs: dictionary mapping structure names to distance thresholds for outlier rejection
        """
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        # Pull normalization values out of model
        self.means = self.model.means.cpu().detach().numpy()
        self.stds = self.model.stds.cpu().detach().numpy()

        # Mapping (include outlier)
        self.label_to_number = {k: v for k, v in label_map.items()}
        self.number_to_label = {v: k for k, v in label_map.items()}

        amorphous_num = max(label_map.keys()) + 1
        unknown_num = amorphous_num + 1

        self.label_to_number["amorphous"] = amorphous_num
        self.label_to_number["unknown"] = unknown_num

        self.number_to_label[amorphous_num] = "amorphous"
        self.number_to_label[unknown_num] = "unknown"

        # Reference vectors and delta cutoffs
        self.ref_vecs = ref_vecs
        self.delta_cutoffs = delta_cutoffs

    def calculate(self, lattice: DataCollection) -> np.ndarray:
        """
        Classifies atoms in the input lattice as <structure>, amorphous, or unknown.

        Args:
            lattice: OVITO DataCollection representing the atomic structure
        Returns:
            Array of structure labels for each atom (as strings)
        """
        features = compute_feature_vectors(lattice)
        amorphous = calculate_amorphous(lattice)
        results = []

        with torch.no_grad():
            preds = (
                self.model(torch.from_numpy(features).float().to(self.device))
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

        for i in range(len(features)):
            if not amorphous[i]:
                ref_vec = self.ref_vecs[self.number_to_label[preds[i]]]
                normalized_feature = (features[i] - self.means) / (self.stds + 1e-6)
                dist = np.linalg.norm(normalized_feature - ref_vec)

                if dist >= self.delta_cutoffs[self.number_to_label[preds[i]]]:
                    results.append("unknown")
                else:
                    results.append(self.number_to_label[preds[i]])
            else:
                results.append("amorphous")

        return results

    def save(self, model_name: str, file_dir: str) -> None:
        """
        Saves the entire DC3 model and metadata to disk.
        Includes the model weights and associated label map, reference vectors, and cutoffs.

        Args:
            model_name: filename prefix for the saved model
            file_dir: directory to save the model into
        """
        # Encode into tensor
        meta_json = json.dumps(
            {
                "label_map": self.label_to_number,
                "ref_vecs": {k: v.tolist() for k, v in self.ref_vecs.items()},
                "delta_cutoffs": {k: float(v) for k, v in self.delta_cutoffs.items()},
            }
        ).encode("utf-8")

        meta_tensor = torch.tensor(list(meta_json), dtype=torch.uint8)

        # Save
        torch.save(
            {"state_dict": self.model.cpu().state_dict(), "metadata": meta_tensor},
            os.path.join(file_dir, f"{model_name}.pth"),
        )


def create_model(structure_map: str | dict[str, str | None] | None) -> DC3:
    """
    Creates a DC3 model from scratch or loads a saved model from disk.

    Args:
        structure_map: determines how the model is created.

                       If it's a string then it must be a path to a full DC3 model which is then loaded

                       If it's a dict then it will contain key value pairs in the form
                       {<structure name>, path to OVITO file of perfect lattice of structure | None}

                       If the value is None then it defaults to the saved perfect lattices
                       associated with <structure name> (currently there is only support for
                       bcc, cd, fcc, hcp, hd, sc). Note that <structure_name> should never be
                       amorphous or unknown.
    Returns:
        A DC3 instance ready for classification
    """
    if isinstance(structure_map, dict):
        # Data
        data_handler = DataHandler(structure_map)
        dataset = CrystalDataset(data_handler.get_synthetic_data())

        # Model
        model = MLPModel(len(dataset.label_map), dataset.means, dataset.stds)
        train(model, dataset)

        # Outlier
        ref_vecs = compute_ref_vec(
            data_handler.get_perfect_lattice_data(), dataset.means, dataset.stds
        )
        delta_cutoffs = compute_delta_cutoff(
            data_handler.get_synthetic_data(), ref_vecs, dataset.means, dataset.stds
        )

        # Turn everything into a DC3 model
        return DC3(model, dataset.label_map, ref_vecs, delta_cutoffs)

    else:
        dc3_path = structure_map if structure_map is not None else SAVED_FULL_MODEL_PATH
        assert os.path.isfile(dc3_path), f"DC3 model at {dc3_path} does not exist"

        # Load
        dc3_loaded = torch.load(dc3_path, map_location="cpu", weights_only=False)

        # Decode
        metadata_bytes = bytes(dc3_loaded["metadata"].tolist())
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        ref_vecs = {k: np.array(v) for k, v in metadata["ref_vecs"].items()}

        # Model
        model = MLPModel(classes=len(metadata["label_map"]))
        model.load_state_dict(dc3_loaded["state_dict"], strict=True)

        return DC3(model, metadata["label_map"], ref_vecs, metadata["delta_cutoffs"])
