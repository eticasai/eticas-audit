import numpy as np
from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask, binarize_predictions
import logging
logger = logging.getLogger(__name__)


class Performance(BaseFairnessMetric):
    """
    Metric D(A)-positive
    Calculates the distribution of a given group when label is positive.
    """

    def compute(self,
                input_data,
                sensitive_attrs=None,
                label_column: str = None,
                positive_outcome=[1],
                output_column: str = None):
        """
        Parameters
        ----------
       :param input_data: Dataset which should contain the columns in param sensitive_attrs.
       :param sensitive_attrs: The sensitive attributes (e.g., 'gender', 'race') relevant to fairness.
       :param label_column: Name of the column containing the target.
       :param output_column: Name of the column containing the prediction / classification.
        Returns
        ----------
       :return: A dictionary containing the results for each group
        """
        self.validate_parameters(
            input_data=input_data,
            sensitive_attrs=sensitive_attrs,
            label_column=label_column,
            positive_outcome=positive_outcome,
            output_column=output_column
        )
        np.random.seed(123)
        # Verify training data
        # verify_training(csv_path)
        if not input_data[label_column].isin([0, 1]).all():
            # Convert to binary
            input_data[label_column] = binarize_predictions(input_data[label_column].values)
        if not input_data[output_column].isin([0, 1]).all():
            # Convert to binary
            input_data[output_column] = binarize_predictions(input_data[output_column].values)

        json_groups = sensitive_attrs

        result_list = {}
        TP = ((input_data[label_column].isin(positive_outcome)) &
              (input_data[output_column].isin(positive_outcome))).sum().item()  # True Positives
        FP = ((~input_data[label_column].isin(positive_outcome)) &
              (input_data[output_column].isin(positive_outcome))).sum().item()  # False Positives
        TN = ((~input_data[label_column].isin(positive_outcome)) &
              (~input_data[output_column].isin(positive_outcome))).sum().item()  # True Negatives
        FN = ((input_data[label_column].isin(positive_outcome)) &
              (~input_data[output_column].isin(positive_outcome))).sum().item()  # False Negatives
        accuracy = np.round((TP + TN) / (TP + TN + FP + FN), 4).item()
        precision = np.round(TP / (TP + FP), 4).item() if (TP + FP) > 0 else 0
        recall = np.round(TP / (TP + FN), 4).item() if (TP + FN) > 0 else 0
        f1 = np.round(2 * (precision * recall) / (precision + recall), 4).item() if (precision + recall) > 0 else 0
        largest_class_frequency = np.round(input_data[label_column].value_counts().max() / len(input_data), 4).item()
        poor_performance = np.round(accuracy / largest_class_frequency, 4).item()
        normalized_risk = self.normalize_value(poor_performance)
        result_list.update({'overall': {
                                    'accuracy': np.round(accuracy * 100, 4).item(),
                                    'precision': np.round(precision * 100, 4).item(),
                                    'recall': np.round(recall * 100, 4).item(),
                                    'f1': np.round(f1 * 100, 4).item(),
                                    'lcf': largest_class_frequency,
                                    'TP': int(TP),
                                    'FP': int(FP),
                                    'TN': int(TN),
                                    'FN': int(FN),
                                    'poor_performance': poor_performance,
                                    'normalized_risk': normalized_risk,
                                    'bias_level': self.get_bias_level(normalized_risk)}})
        for item in json_groups.items():

            group = item[0]
            try:
                if item[1]['type'] == 'simple':
                    filters = item[1]['columns']
                else:
                    filters = np.concat([json_groups[c]['columns'] for c in item[1]['groups']]).tolist()

                mask_privileged, mask_underprivileged = get_mask(input_data, filters)

                data = input_data[mask_underprivileged].copy()
                TP = ((data[label_column].isin(positive_outcome)) &
                      (data[output_column].isin(positive_outcome))).sum()  # True Positives
                FP = ((~data[label_column].isin(positive_outcome)) &
                      (data[output_column].isin(positive_outcome))).sum()  # False Positives
                TN = ((~data[label_column].isin(positive_outcome)) &
                      (~data[output_column].isin(positive_outcome))).sum()  # True Negatives
                FN = ((data[label_column].isin(positive_outcome)) &
                      (~data[output_column].isin(positive_outcome))).sum()  # False Negatives
                accuracy = np.round((TP + TN) / (TP + TN + FP + FN), 4).item()
                precision = np.round(TP / (TP + FP), 4).item() if (TP + FP) > 0 else 0
                recall = np.round(TP / (TP + FN), 4).item() if (TP + FN) > 0 else 0
                f1 = 0
                if (precision + recall) > 0:
                    f1 = np.round(2 * (precision * recall) / (precision + recall), 4).item()

                largest_class_frequency = np.round(data['outcome'].value_counts().max() / len(data), 4).item()
                poor_performance = np.round(accuracy / largest_class_frequency, 4).item()
                normalized_risk = self.normalize_value(poor_performance)
                result_list.update({group: {
                                            'accuracy': np.round(accuracy * 100, 4).item(),
                                            'precision': np.round(precision * 100, 4).item(),
                                            'recall': np.round(recall * 100, 4).item(),
                                            'f1': np.round(f1 * 100, 4).item(),
                                            'lcf': largest_class_frequency,
                                            'TP': int(TP),
                                            'FP': int(FP),
                                            'TN': int(TN),
                                            'FN': int(FN),
                                            'poor_performance': np.round(poor_performance, 4).item(),
                                            'normalized_risk': np.round(normalized_risk, 4).item(),
                                            'bias_level': self.get_bias_level(normalized_risk)}})
            except KeyError:
                logger.error(f"Group no present in data: '{group}'")
                # Captura cualquier ValueError lanzado por get_benchmarking
                result_list.update({group: {
                                            'accuracy': 0,
                                            'precision': 0,
                                            'recall': 0,
                                            'f1': 0,
                                            'lcf': 0,
                                            'TP': int(TP),
                                            'FP': int(FP),
                                            'TN': int(TN),
                                            'FN': int(FN),
                                            'poor_performance': 0,
                                            'normalized_risk': None,
                                            'bias_level': self.get_bias_level(0),
                                            'error': 'group no present in data.'}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list

    def normalize_value(self, value):
        scaled_accuracy = value
        if scaled_accuracy < 1:
            return scaled_accuracy * 60 / 1
        elif 1.0 <= scaled_accuracy < 1.3:
            return 60 + (scaled_accuracy - 1) * 20 / 0.3
        elif 1.3 <= scaled_accuracy <= 2:
            return 80 + (scaled_accuracy - 1.3) * 20 / 0.7
        else:
            return 100
