import numpy as np
from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask, binarize_predictions
import logging
logger = logging.getLogger(__name__)


class D_parity(BaseFairnessMetric):
    """
    Metric D- Parity

    Distribution of the predicted labels differs between protected and reference groups in the population (DI).
    """

    def compute(self,
                input_data,
                sensitive_attrs=None,
                label_column: str = None,
                positive_output: list = [1]):
        """
        Parameters
        ----------
       :param input_data: Dataset which should contain the columns in param sensitive_attrs.
       :param sensitive_attrs: The sensitive attributes (e.g., 'gender', 'race') relevant to fairness.
       :param label_column: Name of the column containing the target.
       :param positive_output: Values of the column_output consider as positive.
        Returns
        ----------
       :return: A dictionary containing the results for each group
        """

        self.validate_parameters(
            input_data=input_data,
            sensitive_attrs=sensitive_attrs,
            label_column=label_column,
            positive_output=positive_output
        )

        if not input_data[label_column].isin([0, 1]).all():
            input_data[label_column] = binarize_predictions(input_data[label_column].values)
        json_groups = sensitive_attrs
        result_list = {}

        for item in json_groups.items():
            group = item[0]
            if item[1]['type'] == 'simple':
                filters = item[1]['columns']
            else:
                filters = np.concat([json_groups[c]['columns'] for c in item[1]['groups']]).tolist()

            try:
                mask_privileged, mask_underprivileged = get_mask(input_data, filters)
                input_data_underprivileged = input_data[mask_underprivileged].copy()
                input_data_privileged = input_data[mask_privileged].copy()

                mask_underprivileged = input_data_underprivileged[label_column].isin(positive_output)
                mask_privileged = input_data_privileged[label_column].isin(positive_output)
                # item is to convert np float64 to native python float.
                data_underprivileged = mask_underprivileged.sum() / mask_underprivileged.shape[0]
                p_positive = np.round(data_underprivileged*100, 1).item()
                data_privileged = mask_privileged.sum() / mask_privileged.shape[0]
                DI = np.round(data_underprivileged / data_privileged, 2).item()
                normalized_risk = self.normalize_value(DI)
                bias_level = self.get_bias_level(normalized_risk)
                result_list.update({group: {'DI': DI,
                                            '%_positive': p_positive,
                                            'normalized_risk': normalized_risk,
                                            'bias_level': bias_level}})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {'DI': 0,
                                            '%_positive': 0,
                                            'normalized_risk': None,
                                            'bias_level': self.get_bias_level(0),
                                            'error': 'group no present in data.'}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list

    def normalize_value(self, value):
        x = value
        normalized = 0
        if 0.8 < x < 1.2:
            if x == 1:
                normalized = 100
            elif x < 1:
                normalized = (x - 0.8) * (100 - 80) / (1 - 0.8) + 80
            else:
                normalized = (x - 1) * (80 - 100) / (1.2 - 1) + 100
        elif 1.2 <= x < 2:
            normalized = (x - 1.2) * (60 - 80) / (2 - 1.2) + 80
        elif x >= 2:
            normalized = 0
        elif 0.5 < x <= 0.8:
            normalized = (x - 0.5) * (80 - 60) / (0.8 - 0.5) + 60
        elif x <= 0.5:
            normalized = x * (60 - 0) / 0.5
        return normalized
