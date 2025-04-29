from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from DC3.dc3 import create_model
from traits.api import *


class DC3Modifier(ModifierInterface):
    """
    DC3Modifier is a custom modifier for the OVITO pipeline that uses the DC3 model to classify crystal structures.
    It modifies the data collection in place based on the classification results.
    """

    structure_map = Any()

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
        # if not self.model_path:
        #     self.model_path = "ml\models\model_2025-04-26_19-26-39.pt"
        # print("Initializing DC3Modifier")
        # self.dc3 = DC3(
        #     self.model_path, self.label_map, self.ref_vec_path, self.delta_cutoff_path
        # )

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
        self.dc3 = create_model(self.structure_map)
        
        print("Calculating structure types")
        result = self.dc3.calculate(data)

        # Convert
        for i in range(len(result)):
            if result[i] in self.label_map:
                result[i] = self.label_map[result[i]]
            elif result[i] == "amorphous":
                result[i] = len(self.label_map)
            elif (result[i] == "unknown"):
                result[i] = len(self.label_map) + 1
            else:
                raise ValueError("Unknown reult")

        data.particles_.create_property("Structure_Type", data=result)
