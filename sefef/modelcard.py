# built-in
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
    to_json() :
        Description
    save() :
        Description
    References
    ----------
    .. [Mitchell2019] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model Cards for Model Reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19). Association for Computing Machinery, New York, NY, USA, 220-229. https://doi.org/10.1145/3287560.3287596
    '''

    def __init__(self, details=None, training=None, evaluation=None, metrics=None):
        self.details = ModelDetails(**details)
        self.training = ModelTraining(**training)
        self.evaluation = ModelEvaluation(**evaluation)
        self.metrics = ModelMetrics(**metrics)

    def to_json(self):
        pass

    def from_json(self):
        pass

    def save(self):
        pass


def load_from_json():
    pass


# if __name__ == '__main__':
#     js = '''{"details": {"name": "model_name", "date": "06/12/2024"},
#     "training": {"dataset": "Menstuation Dataset", "preprocessing": "Description of preprocessing"},
#     "evaluation": {"dataset": "Menstuation Dataset", "preprocessing": "Description of preprocessing"}}'''

#     j = json.loads(js)
#     print(j)
#     u = ModelCard(**j)
#     print(u)
