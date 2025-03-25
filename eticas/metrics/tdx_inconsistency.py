import numpy as np
from .base_metric import BaseFairnessMetric

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd
from ..utils.helpers import get_mask
import logging
logger = logging.getLogger(__name__)


class Tdx_inconsistency(BaseFairnessMetric):
    """
    Metric Drift
    """

    def compute(self, sensitive_attrs: dict = None,
                input_features: list = None,
                input_data_dev: str = None,
                output_column_dev: str = None,
                positive_output_dev: list = [1],
                input_data_prod: str = None,
                output_column_prod: str = None,
                positive_output_prod: list = [1]):
        """
        Parameters
        ----------
        :param sensitive_attrs: The sensitive attributes
                                (e.g., 'gender', 'race') relevant to fairness.
        :param input_features: List of features use as input to get the output.
        :param input_data: Dataset which should contain the sensitive_attrs
                           and input_features.
        :param output_column: Name of the column containing the target.
        :param positive_output: Values of the column_output
                                consider as positive.
        Returns
        ----------
        :return: A dictionary containing the results for each group
        """
        self.validate_parameters(
            sensitive_attrs=sensitive_attrs,
            input_features=input_features,
            input_data_dev=input_data_dev,
            output_column_dev=output_column_dev,
            positive_output_dev=positive_output_dev,
            input_data_prod=input_data_prod,
            output_column_prod=output_column_prod,
            positive_output_prod=positive_output_prod
        )
        column_output = 'inconsistency'
        # Set random seed
        np.random.seed(123)
        input_data_dev[column_output] = 0
        input_data_prod[column_output] = 1
        len_dev = input_data_dev.shape[0]
        len_prod = input_data_prod.shape[0]
        total_len = np.sum([len_dev, len_prod])
        proportion_data = np.max([len_dev, len_prod]) / total_len

        json_groups = sensitive_attrs

        cols_dev = input_data_dev.columns
        cols_prod = input_data_prod.columns
        sensitive_columns = self.extract_sensitive_columns(json_groups)

        data_columns = set(cols_dev) & set(cols_prod)
        sensitive_columns = set(sensitive_columns) & set(data_columns)
        sensitive_columns = list(sensitive_columns)

        train_columns_0 = set(cols_dev) & set(input_features)
        train_columns_1 = set(cols_prod) & set(input_features)

        train_columns = set(train_columns_0) & set(train_columns_1)
        train_columns = list(train_columns)
        result_list = {}

        feat_columns = train_columns + sensitive_columns
        if len(feat_columns) == 0:
            logger.error("length 0 of train and sensitive features is 0.")
            return {'error': 'length 0 of train and sensitive features is 0.'}
        try:
            group = 'overall'
            data_dev = input_data_dev[feat_columns + [column_output]]
            data_prod = input_data_prod[feat_columns + [column_output]]
            data = pd.concat([data_dev, data_prod])
            # Split data (70% train, 30% test) based on the sensitive attribute
            train_data, test_data = train_test_split(data, test_size=0.3,
                                                     stratify=data[column_output],
                                                     random_state=123)
            logistic_base = LogisticRegression(random_state=123)
            logistic_base.fit(train_data[feat_columns], train_data[column_output])
            predictions_base = logistic_base.predict(test_data[feat_columns])
            accuracy_base = np.round(accuracy_score(test_data[column_output],
                                                    predictions_base), 4).item()
            rate = np.round(accuracy_base / proportion_data, 4).item()
            normalized_risk = np.round(self.normalize_value(rate), 4).item()
            result_list.update({group: {
                                'accuracy': np.round(accuracy_base, 4).item(),
                                'proportion': np.round(proportion_data, 4).item(),
                                'rate': rate,
                                'normalized_risk': normalized_risk,
                                'bias_level': self.get_bias_level(normalized_risk)
                                }})
        except ValueError:
            # Captura cualquier ValueError lanzado por get_benchmarking
            logger.error(f"Group no present in data: '{group}'")
            result_list.update({group: {
                                        'accuracy': 0,
                                        'proportion': 0,
                                        'rate': 0,
                                        'normalized_risk': 100,
                                        'bias_level': self.get_bias_level(100),
                                        'error': 'only one value target.'
                                        }})
        for item in json_groups.items():
            data_dev = input_data_dev
            data_prod = input_data_prod
            group = item[0]
            sensitive_columns = []

            try:
                if item[1]['type'] == 'simple':

                    filters = item[1]['columns']
                    mask_privileged, mask_underprivileged = get_mask(input_data_dev, filters)
                    data_dev[item[1]['columns'][0]['name']] = mask_underprivileged.astype(int)
                    mask_privileged_2, mask_underprivileged_2 = get_mask(input_data_prod, filters)
                    data_prod[item[1]['columns'][0]['name']] = mask_underprivileged.astype(int)
                    sensitive_columns.append(item[1]['columns'][0]['name'])

                else:
                    filters = np.concat([json_groups[c]['columns'] for c in item[1]['groups']]).tolist()
                    for filter in filters:
                        mask_privileged, mask_underprivileged = get_mask(input_data_dev, [filter])
                        data_dev[filter['name']] = mask_underprivileged.astype(int)
                        mask_privileged_2, mask_underprivileged_2 = get_mask(input_data_prod, [filter])
                        data_prod[filter['name']] = mask_underprivileged.astype(int)
                        sensitive_columns.append(filter['name'])
                feat_columns = train_columns + sensitive_columns

                data_dev = data_dev[mask_underprivileged][feat_columns + [column_output]]
                data_prod = data_prod[mask_underprivileged_2][feat_columns + [column_output]]
                data = pd.concat([data_dev, data_prod])
                # Split data (70% train, 30% test) based on the sensitive attribute
                train_data, test_data = train_test_split(data, test_size=0.3,
                                                         stratify=data[column_output],
                                                         random_state=123)
                logistic_base = LogisticRegression(random_state=123)
                logistic_base.fit(train_data[feat_columns], train_data[column_output])
                predictions_base = logistic_base.predict(test_data[feat_columns])
                accuracy_base = np.round(accuracy_score(test_data[column_output], predictions_base), 4).item()
                rate = np.round(accuracy_base / proportion_data, 4).item()
                normalized_risk = np.round(self.normalize_value(rate), 4).item()
                result_list.update({group: {
                                            'accuracy': np.round(accuracy_base, 4).item(),
                                            'proportion': np.round(proportion_data, 4).item(),
                                            'rate': rate,
                                            'normalized_risk': normalized_risk,
                                            'bias_level': self.get_bias_level(normalized_risk)
                                             }})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {
                                            'accuracy': 0,
                                            'proportion': 0,
                                            'rate': 0,
                                            'normalized_risk': None,
                                            'bias_level': self.get_bias_level(0),
                                            'error': 'group no present in data.'
                                            }})
            except ValueError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Only one value target: '{group}'")
                result_list.update({group: {
                                            'accuracy': 0,
                                            'proportion': 0,
                                            'rate': 0,
                                            'normalized_risk': 100,
                                            'bias_level': self.get_bias_level(100),
                                            'error': 'only one value target.'
                                            }})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list

    def extract_sensitive_columns(self, json_groups):

        columns = []

        for key, value in json_groups.items():
            column_data = value.get('columns', [])
            # Si los elementos son diccionarios, extraer el 'name'
            if isinstance(column_data, list) and len(column_data) > 0 and isinstance(column_data[0], dict):
                columns.extend([col['name'] for col in column_data])
            # Si son solo strings en una lista, los añadimos directamente
            elif isinstance(column_data, list) and all(isinstance(col, str) for col in column_data):
                columns.extend(column_data)

        return columns

    def normalize_value(self, value):
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
