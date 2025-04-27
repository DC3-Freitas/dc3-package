from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from dc3 import DC3


class DC3Modifier(ModifierInterface):
    """
    DC3Modifier is a custom modifier for the OVITO pipeline that uses the DC3 model to classify crystal structures.
    It modifies the data collection in place based on the classification results.
    """

    def __init__(
        self,
        model_path: str,
        label_map: dict[str, int],
        ref_vec_path: str,
        delta_cutoff_path: str,
    ):
        """
        Initialize the DC3Modifier with the model path and label mapping.

        Args:
            model_path (str): Path to the trained model file.
            label_map (dict[str, int]): Mapping from structure names to integer labels.
            ref_vec_path (str): Path to the reference vectors CSV file.
            delta_cutoff_path (str): Path to the delta cutoffs CSV file.
        """
        # need to figure out how this works with the pipeline
        self.dc3 = DC3(model_path, label_map, ref_vec_path, delta_cutoff_path)

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
        self.dc3.calculate(data)
