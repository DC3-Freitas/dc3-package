from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from DC3.dc3 import DC3
from traits.api import *


class DC3Modifier(ModifierInterface):
    """
    DC3Modifier is a custom modifier for the OVITO pipeline that uses the DC3 model to classify crystal structures.
    It modifies the data collection in place based on the classification results.
    """

    model_path = Str()
    label_map = Dict(Str, Int)
    ref_vec_path = Str()
    delta_cutoff_path = Str()

    def __init__(self):
        """
        Initialize the DC3Modifier with the model path and label mapping.

        Args:
            model_path (str): Path to the trained model file.
            label_map (dict[str, int]): Mapping from structure names to integer labels.
            ref_vec_path (str): Path to the reference vectors CSV file.
            delta_cutoff_path (str): Path to the delta cutoffs CSV file.
        """
        # need to figure out how this works with the pipeline
        if not self.model_path:
            self.model_path = "ml/models/model_2025-04-26_23-04-46.pt"
        print("Initializing DC3Modifier")
        self.dc3 = DC3(
            self.model_path, self.label_map, self.ref_vec_path, self.delta_cutoff_path
        )

    def modify(self, data: DataCollection, frame: int, **kwargs):
        """
        Modify the data collection in place.
        This method is called for each frame of the pipeline.

        Args:
            data (DataCollection): The data collection to modify.
            frame (int): The current frame number.
            **kwargs: Additional keyword arguments.
        """
        # need to research how kwargs works in the pipeline
        # print("Reinitializing DC3Modifier")
        # self.dc3 = DC3(model_path, label_map, ref_vec_path, delta_cutoff_path)
        print("Calculating structure types")
        data.particles_.create_property("Structure_Type", data=self.dc3.calculate(data))
