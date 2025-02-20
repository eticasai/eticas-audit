"""
base_model.py
=============

Defines a base class for models in the eticas library.
"""

from abc import ABC


class BaseModel(ABC):
    """
    Base class for a model in the eticas library.
    """

    def __init__(
        self,
        model_name: str,
        description: str = None,
        country: str = None,
        state: str = None,
        sensitive_attributes: dict = None,
        distribution_ref: dict = None,
        features: list = None
    ):
        """
        Initialize basic model metadata.

        Parameters
        ----------
        :param model_name : str
            Name of the model (e.g., 'Logistic Regression', 'XGBoost', etc.).
        :param description : str, optional
            A brief description of the model (version, purpose, etc.).
        :param country : str, optional
            The country where the model is primarily used or deployed.
        :param state : str, optional
            The state or region where the model is primarily used.
        :param sensitive_attributes : dict, optional
            The sensitive attributes (e.g., 'gender', 'race') relevant to fairness.
                The column in dataset.
                Under-privileged values.
                {'gender' : {'columns' : [
                                            {
                                            "name": "sex",
                                            "underprivileged": [2]
                                            }
                                        ],
                             'type' : 'simple'},
                 'ethnicity' : {'columns' : [
                                                        {
                                                        "name": "ethnicity",
                                                        "privileged": [1]
                                                        }
                                                    ],
                                         'type' : 'simple'}
                 'gender_ethnicity' : {'groups' : ["gender","ethnicity"],
                                                'type' : 'complex'}}
        :param distribution_ref : dict Expected distribution for underprivileged group in production.
        :param features : list, optional
            The features used by the model (column names in a dataset).
        """
        self.model_name = model_name
        self.description = description
        self.country = country
        self.state = state
        self.sensitive_attributes = sensitive_attributes
        self.features = features
        self.labeled_results = {}
        self.production_results = {}
        self.impacted_results = {}
        self.drift_results = {}
        if distribution_ref is None:
            self.distribution_ref = {}
        else:
            self.distribution_ref = distribution_ref

    def __str__(self):
        return f"{self.__class__.__name__}({self.model_name})"
