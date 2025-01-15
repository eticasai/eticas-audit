import numpy as np
from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask, binarize_predictions
import logging
logger = logging.getLogger(__name__)


class D_statisticalparity(BaseFairnessMetric):
    """
    Metric D-Statistical Parity

    Distribution of the predicted labels differs between protected and reference groups in the population (SPD).
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
            # Convert to binary
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
                SPD = np.round(data_underprivileged - data_privileged, 2).item()
                normalized_risk = self.normalize_value(SPD)
                bias_level = self.get_bias_level(normalized_risk)
                result_list.update({group: {'SPD': SPD,
                                            '%_positive': p_positive,
                                            'normalized_risk': normalized_risk,
                                            'bias_level': bias_level}})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {'SPD': 0,
                                            '%_positive': 0,
                                            'normalized_risk': None,
                                            'bias_level': self.get_bias_level(0),
                                            'error': 'group no present in data.'}})

        logger.info(f"Completed: '{self.__str__()}'")
        return result_list

    def normalize_value(self, value):

        abs_SPD = abs(value)
        if abs_SPD < 0.1:
            normalized = 100 - (abs_SPD / 0.1) * 20
        elif 0.1 <= abs_SPD < 0.2:
            normalized = 80 - ((abs_SPD - 0.1) / 0.1) * 20
        else:
            normalized = 60 - ((abs_SPD - 0.2) / 0.8) * 60
        return normalized
