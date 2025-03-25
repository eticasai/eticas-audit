import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask, binarize_predictions

import logging

logger = logging.getLogger(__name__)


class Da_informative(BaseFairnessMetric):
    """
    Metric D(A)-informative
    Indicates that the sensitive data is informative for the label
    """

    def compute(self,
                input_data,
                sensitive_attrs=None,
                label_column=None,
                input_features=None):
        """
        Parameters
        ----------
       :param input_data: Dataset which should contain the columns in param sensitive_attrs.
       :param sensitive_attrs: The sensitive attributes (e.g., 'gender', 'race') relevant to fairness.
       :param label_column: Name of the column containing the target.
       :param input_features: List of features use as input to get the output.
        Returns
        ----------
       :return: A dictionary containing the results for each group
        """
        self.validate_parameters(
            input_data=input_data,
            sensitive_attrs=sensitive_attrs,
            label_column=label_column,
            input_features=input_features,
        )
        np.random.seed(123)
        input_data = input_data.dropna()

        if not input_data[label_column].isin([0, 1]).all():
            # Convert to binary
            input_data[label_column] = binarize_predictions(input_data[label_column].values)

        json_groups = sensitive_attrs

        # Split data (70% train, 30% test) based on the sensitive attribute

        train_columns = [c for c in input_data.columns if c in input_features]

        if len(train_columns) == 0:
            logger.error("Input features are not in dataset.")
            return {'error': "Input features are not in dataset."}
        result_list = {}

        for item in json_groups.items():

            data = input_data.copy()[train_columns + [label_column]]
            group = item[0]
            s_columns = []

            try:

                if item[1]['type'] == 'simple':

                    filters = item[1]['columns']
                    mask_privileged, mask_underprivileged = get_mask(input_data, filters)
                    data[item[1]['columns'][0]['name']] = mask_underprivileged.astype(int)
                    s_columns.append(item[1]['columns'][0]['name'])

                else:
                    filters = np.concat([json_groups[c]['columns'] for c in item[1]['groups']]).tolist()
                    for filter in filters:
                        mask_privileged, mask_underprivileged = get_mask(input_data, [filter])
                        data[filter['name']] = mask_underprivileged.astype(int)
                        s_columns.append(filter['name'])

                # Split data (70% train, 30% test) based on the sensitive attribute
                train_data, test_data = train_test_split(data, test_size=0.3, random_state=123)
                logistic_base = LogisticRegression(random_state=123)
                logistic_base.fit(train_data[train_columns], train_data[label_column])
                predictions_base = logistic_base.predict(test_data[train_columns])
                accuracy_base = np.round(accuracy_score(test_data[label_column], predictions_base), 4).item()

                logistic = LogisticRegression(random_state=123)
                logistic.fit(train_data[train_columns + s_columns], train_data[label_column])
                predictions = logistic.predict(test_data[train_columns + s_columns])
                accuracy = np.round(accuracy_score(test_data[label_column], predictions), 4).item()

                accuracy_rate = accuracy / accuracy_base
                normalized_risk = np.round(self.normalize_value(accuracy_rate), 4).item()
                result_list.update({group: {
                                            'accuracy_base': accuracy_base,
                                            'accuracy_group': accuracy,
                                            'accuracy': accuracy_rate,
                                            'normalized_risk': normalized_risk,
                                            'bias_level': self.get_bias_level(normalized_risk)}})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {
                                            'accuracy_base': 0,
                                            'accuracy_group': 0,
                                            'accuracy': 0,
                                            'normalized_risk': None,
                                            'bias_level': self.get_bias_level(0),
                                            'error': 'group no present in data.'}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list

    def normalize_value(self, value):
        '''
        Normalize the value between 0 - 100
        '''
        accuracy = value - 1
        normalized = 0
        if accuracy <= 0:
            normalized = 100
        elif accuracy < 0.1:
            normalized = 100 - (abs(accuracy) / 0.1) * 20
        elif 0.1 <= accuracy < 0.2:
            normalized = 80 - ((accuracy - 0.1) / 0.1) * 20
        elif accuracy >= 0.2:
            normalized = 60 - ((accuracy - 0.2) / 0.8) * 60
        return normalized
