"""
base_audit.py
=============

Defines the abstract base class for model audits.
"""

from abc import ABC, abstractmethod


class BaseAudit(ABC):
    """
    Abstract base class for auditing a model.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : BaseModel
            The model instance to be audited.
        """
        self.model = model

    @abstractmethod
    def run_audit(self,
                  dataset_path: str,
                  label_column: str,
                  output_column: str = None,
                  positive_output: list = None):
        """
        :param dataset_path : path to training dataset.
        :param label_column: Name of the column containing the target.
        :param output_column: Name of the column containing the prediction / classification.
        :param positive_output: Values of the column_output consider as positive.


        Returns
        -------
        :return: dict. The result of the training audit.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model.model_name})"
