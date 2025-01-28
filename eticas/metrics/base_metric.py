from abc import ABC, abstractmethod
import pandas as pd


class BaseFairnessMetric(ABC):
    """
    Abstract base class defining the basic structure of a fairness metric.
    """

    @abstractmethod
    def compute(self,
                input_data: pd.DataFrame,
                sensitive_attrs: dict = None,
                input_features: list = None,
                column_output: str = None,
                positive_output: list = [1]):
        """
        Main method to compute the metric.
        Parameters
        ----------
        :param input_data: List or array of ground-truth (true) labels.
        :param sensitive_attrs: List or array of model predictions or scores.
        :param input_features: List of features use as input to get the output.
        :param column_output: Name of the column containing the target.
        :param positive_output: Values of the column_output consider as positive.
        Returns
        -------
        :return: A numeric value or a dictionary containing the metric results.
        """
        raise NotImplementedError("Not implemented.")

    def normalize_value(self, value):
        """
        Main method to normalize the metric between 0 (BAD) to 100 (GOOD).

        :param value: Metric value.

        :return: A numeric value between 0 to 100.
        """
        raise NotImplementedError("Not implemented.")

    def get_bias_level(self, value):
        """
        Main method to get the bias level.

        :param value: Metric value.

        :return:
        """
        return 'Low/no bias' if value > 80 else 'High bias' if value <= 60 else 'Medium bias'

    def validate_parameters(self, **kwargs):
        """
        check input params are null or not.
        """
        for param_name, param_value in kwargs.items():
            if param_value is None:
                raise ValueError(f"You must provide a value for {param_name} to compute {self.__class__.__name__}")

    def __str__(self):
        """
        String representation of the metric.
        """
        return self.__class__.__name__
