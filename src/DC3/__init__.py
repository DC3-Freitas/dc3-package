"""
OVITO modifier for crystal structure classification using the DC3 model.
"""

from DC3.dc3 import create_model
from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from traits.api import Any, Bool


class DC3Modifier(ModifierInterface):
    """
    DC3Modifier is a custom modifier for the OVITO pipeline that uses the
    DC3 model to classify crystal structures. It modifies the data collection in
    place based on the classification results.
    """

    model_input = Any()
    run = Bool(False, help="Check to start DC3 classification")

    def __init__(self) -> None:
        """
        Initialize the DC3Modifier with the model path and label mapping.
        Accepts no arguments.
        """
        super().__init__()

        # Initial values
        self.dc3 = None
        self.number_to_label = {}
        self.label_to_number = {}

    def initialize_info_from_input(self) -> None:
        """
        Initializes values for dc3 and supporting label maps using the
        informaation contained in model_input.
        """

        # Make sure model input is valid
        assert isinstance(
            self.model_input, (str, dict, type(None))
        ), "model_input must be str, dict, or none"

        if isinstance(self.model_input, dict):
            assert (
                "amorphous" not in self.model_input
                and "unknown" not in self.model_input
            ), "Amorphous / unknown should not be name of any structures"

        # Create
        self.dc3 = create_model(self.model_input)

        # Defensively copy both label maps
        self.label_to_number = dict(self.dc3.label_to_number)
        self.number_to_label = dict(self.dc3.number_to_label)

        # Note that the label map used in dc3 and dataset must not include amorphous and unknown
        amorphous_num = max(self.label_to_number.values()) + 1
        unknown_num = amorphous_num + 1

        self.label_to_number["amorphous"] = amorphous_num
        self.label_to_number["unknown"] = unknown_num

        self.number_to_label[amorphous_num] = "amorphous"
        self.number_to_label[unknown_num] = "unknown"

    def modify(self, data: DataCollection, frame: int, **kwargs) -> None:
        """
        Modify the data collection in place.
        This method is called for each frame of the pipeline.

        Args:
            data (DataCollection): The data collection to modify.
            frame (int): The current frame number.
            **kwargs: Additional keyword arguments.
        """
        # Only run if the run flag is set
        if not self.run:
            return

        # Initialize a new model if we haven't already
        if self.dc3 is None:
            self.initialize_info_from_input()

        # Calculations
        print("\nCalculating structure types")
        result = self.dc3.calculate(data)

        # Set as structure_type
        data.particles_.create_property(
            "structure_type", data=[self.label_to_number[i] for i in result]
        )

    def save_full_model(self, model_name: str, file_dir: str) -> None:
        """
        Saves the entire DC3 model and metadata according to what
        model_input is set to.

        Args:
            model_name: filename prefix for the saved model (excluding .pth)
            file_dir: directory to save the model into (absolute path when called from outside)
        """
        # Initialize a new model if we haven't already
        if self.dc3 is None:
            self.initialize_info_from_input()

        self.dc3.save(model_name, file_dir)
