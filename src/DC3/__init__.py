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
    run = Bool(False, help="Check to start DC3 classification")

    def __init__(self):
        """
        Initialize the DC3Modifier with the model path and label mapping.

        TODO
        """
        super().__init__()

        # Initial values
        self.dc3 = None
        self.number_to_label = {}
        self.label_to_number = {}

    def modify(self, data: DataCollection, frame: int, **kwargs):
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
        
        # Initialize a new model and supporting label maps
        if self.dc3 is None:
            self.dc3 = create_model(self.structure_map)

            # Defensively copy both label maps
            self.label_to_number = dict(self.dc3.label_to_number)
            self.number_to_label = dict(self.dc3.number_to_label)
            
            # Add amorphous and unknown
            amorphous_num = max(self.number_to_label.keys()) + 1
            unknown_num = amorphous_num + 1

            self.label_to_number["amorphous"] = amorphous_num
            self.label_to_number["unknown"] = unknown_num

            self.number_to_label[amorphous_num] = "amorphous"
            self.number_to_label[unknown_num] = "unknown"
        
        if isinstance(self.structure_map, dict):
            assert "amorphous" not in self.structure_map and "unknown" not in self.structure_map, "Amorphous / unknown should not be name of any structures"
        
        # Calculations
        print("Calculating structure types")
        result = self.dc3.calculate(data)

        # Set as Structure_Type
        data.particles_.create_property("Structure_Type", data=[self.label_to_number[i] for i in result])
