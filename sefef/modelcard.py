# built-in
import os
import json
import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelPerformance:
    Sen: Optional[float] = None
    FPR: Optional[float] = None
    TiW: Optional[float] = None
    AUC_TiW: Optional[float] = None
    resolution: Optional[float] = None
    reliability: Optional[float] = None
    BS: Optional[float] = None
    BSS: Optional[float] = None


@dataclasses.dataclass
class ModelDetails:
    name: Optional[str] = None
    date: Optional[str] = None


@dataclasses.dataclass
class ModelTraining:
    dataset: Optional[str] = None
    motivation: Optional[str] = None
    preprocessing: Optional[str] = None


@dataclasses.dataclass
class ModelEvaluation:
    dataset: Optional[str] = None
    motivation: Optional[str] = None
    preprocessing: Optional[str] = None


@dataclasses.dataclass
class ModelMetrics:
    performance: Optional[ModelPerformance] = None
    decision_thr: Optional[float] = None


class ModelCard:
    ''' Creates a model card following [Mitchell2019]_.   

    Attributes
    ---------- 
    details : ModelDetails
        Contains basic information regarding the model version, type and other details.
    training : ModelTraining
        Information regarding the set of data used for training, preferably including preprocessing steps to ensure reproducibility. 
    evaluation : ModelEvaluation
        Same structure as "training".
    metrics : ModelMetrics
        Contains information such as model performance measures, decision thresholds, and approaches to uncertainty.

    Methods
    -------
    update(data) :
        Receives a dictionary with new data to update the ModelCard instance. Only updates the attributes present in the dictionary, up to 2 levels, i.e. the attributes of ModelCard or the attributes of ModelDetails, ModelTraining, ModelEvaluation, or ModelMetrics. 
    to_dict() :
        Recursively serializes ModelCard and its attributes into a dictionary.
    save(folder_path, filename[optional]) :
        Saves the contents of ModelCard as JSON. If "filename" is not provided, ModelCard.ModelDetails.name will be used as the file name, i.e. "folder_path/name.json".
    References
    ----------
    .. [Mitchell2019] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model Cards for Model Reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19). Association for Computing Machinery, New York, NY, USA, 220-229. https://doi.org/10.1145/3287560.3287596
    '''

    def __init__(self, details={}, training={}, evaluation={}, metrics={}):
        self.details = ModelDetails(**details)
        self.training = ModelTraining(**training)
        self.evaluation = ModelEvaluation(**evaluation)
        self.metrics = ModelMetrics(**metrics)

    def update(self, data):
        ''' Receives a dictionary with new data to update the ModelCard instance. Only updates the attributes present in the dictionary, up to 2 levels, i.e. the attributes of ModelCard or the attributes of ModelDetails, ModelTraining, ModelEvaluation, or ModelMetrics. 

        Parameters
        ---------- 
        data : dict
            Keys should only be included if the attributes are meant to be overwritten. 
        '''

        if 'details' in data.keys():
            for key, value in data['details'].items():
                setattr(self.details, key, value)
        if 'training' in data.keys():
            for key, value in data['training'].items():
                setattr(self.training, key, value)
        if 'evaluation' in data.keys():
            for key, value in data['evaluation'].items():
                setattr(self.evaluation, key, value)
        if 'metrics' in data.keys():
            for key, value in data['metrics'].items():
                setattr(self.metrics, key, value)

    def to_dict(self):
        ''' Recursively serialize ModelCard and its attributes into a dictionary. 

        Returns
        -------
        result : dict
            Nested dictionary with ModelCard attributes and respective contents. 
        '''
        def serialize(obj):
            """Helper function to serialize nested objects."""
            if hasattr(obj, "__dict__"):  # Check if the object has attributes
                result = {}
                for key, value in vars(obj).items():
                    if hasattr(value, "__dict__"):  # If it's another object, serialize it
                        result[key] = serialize(value)
                    elif isinstance(value, dict):  # Handle dictionary attributes
                        result[key] = {k: serialize(v) if hasattr(v, "__dict__") else v for k, v in value.items()}
                    elif isinstance(value, (list, tuple)):  # Handle list or tuple attributes
                        result[key] = [serialize(item) if hasattr(item, "__dict__") else item for item in value]
                    else:  # For primitive types
                        result[key] = value
                return result
            elif isinstance(obj, (list, tuple)):  # If it's a list or tuple
                return [serialize(item) if hasattr(item, "__dict__") else item for item in obj]
            else:
                return obj  # Return primitive types as is

        # Start serialization from `self`
        return serialize(self)

    def save(self, folder_path, filename=None):
        ''' Saves the contents of ModelCard as JSON. If "filename" is not provided, ModelCard.ModelDetails.name will be used as the file name, i.e. "folder_path/name.json".

        Parameters
        ---------- 
        folder_path : str
            Path to the folder where the contents of the ModelCard instance should be saved. 
        '''
        data = self.to_dict()
        if filename is None:
            filename = self.details.name
        with open(os.path.join(folder_path, f'{filename}.json'), 'w') as f:
            json.dump(data, f)


def model_card_from_json(filepath):
    ''' Creates a ModelCard from a json-serializable file (nested dictionaries). Keys must be the same as the ModelCard attributes.

    Parameters
    ---------- 
    filepath : str
        Path to the JSON file containing the desired model card data.

    Returns
    -------
    result : ModelCard
    '''
    with open(filepath) as f:
        data = json.load(f)
    return ModelCard(**data)
