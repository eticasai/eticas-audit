import numpy as np
from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask, binarize_predictions
import logging
logger = logging.getLogger(__name__)


class D_calibrated(BaseFairnessMetric):
    """
    Metric D(A)-positive
    Calculates the distribution of a given group when label is positive.
    """

    def compute(self,
                input_data,
                sensitive_attrs=None,
                label_column: str = None,
                positive_output: list = [1],
                output_column: str = None):
        """
        Parameters
        ----------
        :param input_data: Dataset which should contain the columns in param sensitive_attrs.
        :param sensitive_attrs: The sensitive attributes (e.g., 'gender', 'race') relevant to fairness.
        :param label_column: Name of the column containing the target.
        :param output_column: Name of the column containing the prediction / classification.
        :param positive_output: Values of the column_output consider as positive.
        Returns
        ----------
        :return: A dictionary containing the results for each group
        """
        self.validate_parameters(
            input_data=input_data,
            sensitive_attrs=sensitive_attrs,
            label_column=label_column,
            positive_output=positive_output,
            output_column=output_column
        )
        if not input_data[label_column].isin([0, 1]).all():
            # Convert to binary
            input_data[label_column] = binarize_predictions(input_data[label_column].values)
        if not input_data[output_column].isin([0, 1]).all():
            # Convert to binary
            input_data[output_column] = binarize_predictions(input_data[output_column].values)
        json_groups = sensitive_attrs
        result_list = {}

        for item in json_groups.items():
            group = item[0]

            try:
                if item[1]['type'] == 'simple':
                    filters = item[1]['columns']
                else:
                    filters = np.concat([json_groups[c]['columns'] for c in item[1]['groups']]).tolist()

                mask_privileged, mask_underprivileged = get_mask(input_data, filters)
                data_underprivileged = input_data[mask_underprivileged].copy()
                mask_positive_under = data_underprivileged[label_column].isin(positive_output)
                data_underprivileged_true = data_underprivileged[mask_positive_under]
                data_underprivileged_false = data_underprivileged[~mask_positive_under]
                data_privileged = input_data[mask_privileged].copy()
                mask_positive_privileged = data_privileged[label_column].isin(positive_output)
                data_privileged_true = data_privileged[mask_positive_privileged]
                data_privileged_false = data_privileged[~mask_positive_privileged]
                mask_underprivileged_true = data_underprivileged_true[output_column].isin(positive_output)
                mask_underprivileged_false = data_underprivileged_false[output_column].isin(positive_output)
                mask_privileged_true = data_privileged_true[output_column].isin(positive_output)
                mask_privileged_false = data_privileged_false[output_column].isin(positive_output)
                underprivileged_true = mask_underprivileged_true.sum() / mask_underprivileged_true.shape[0]
                underprivileged_false = mask_underprivileged_false.sum() / mask_underprivileged_false.shape[0]
                privileged_true = mask_privileged_true.sum() / mask_privileged_true.shape[0]
                privileged_false = mask_privileged_false.sum() / mask_privileged_false.shape[0]
                ratio_true = np.round(underprivileged_true / privileged_true, 4).item()
                ratio_false = np.round(underprivileged_false / privileged_false, 4).item()

                result_list.update({group: {
                                        'true_calibrated': {
                                            'underprivileged': np.round(underprivileged_true, 4).item(),
                                            'privileged': np.round(privileged_true, 4).item(),
                                            'ratio_true': ratio_true,
                                            'normalized_risk': np.round(self.normalize_value(ratio_true), 4).item(),
                                            'bias_level': self.get_bias_level(self.normalize_value(ratio_true))
                                                                },
                                        'false_calibrated': {
                                            'underprivileged': np.round(underprivileged_false, 4).item(),
                                            'privileged': np.round(privileged_false, 4).item(),
                                            'ratio_false': ratio_false,
                                            'normalized_risk': np.round(self.normalize_value(ratio_false), 4).item(),
                                            'bias_level': self.get_bias_level(self.normalize_value(ratio_false))
                                                                }
                                                                }})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {
                                            'error': 'group no present in data.',
                                            'true_calibrated': {
                                                'underprivileged': 0,
                                                'privileged': 0,
                                                'ratio_true': 0,
                                                'normalized_risk': None,
                                                'bias_level': self.get_bias_level(0)
                                                                    },
                                            'false_calibrated': {
                                                'underprivileged': 0,
                                                'privileged': 0,
                                                'ratio_false': 0,
                                                'normalized_risk': None,
                                                'bias_level': self.get_bias_level(0)
                                                                    }
                                                                    }})
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
