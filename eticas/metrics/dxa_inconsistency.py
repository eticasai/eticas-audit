import numpy as np
from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask
import logging
logger = logging.getLogger(__name__)


class Dxa_inconsistency(BaseFairnessMetric):
    """
    Metric D(X | A)-inconsistency
    It is possible to predict a sensitive attribute based on the non-sensitive attributes in the data.
    This means that the sensitive attribute (i.e. gender) can be inferred from the other attributes,
    which means that there are proxies in the model.
    """

    def compute(self,
                input_data,
                sensitive_attrs=None,
                input_features=None):
        """
        Parameters
        ----------
       :param input_data: Dataset which should contain the columns in param sensitive_attrs.
       :param sensitive_attrs: The sensitive attributes (e.g., 'gender', 'race') relevant to fairness.
       :param input_features: List of features use as input to get the output.
        Returns
        ----------
       :return: A dictionary containing the results for each group
        """
        self.validate_parameters(
            input_data=input_data,
            sensitive_attrs=sensitive_attrs,
            input_features=input_features,
        )
        np.random.seed(123)
        input_data = input_data.dropna()
        json_groups = sensitive_attrs

        # Split data (70% train, 30% test) based on the sensitive attribute
        train_columns = [c for c in input_data.columns if c in input_features]

        if len(train_columns) == 0:
            logger.error("Input features are not in dataset.")
            return {'error': "Input features are not in dataset."}
        result_list = {}

        for item in json_groups.items():
            group = item[0]

            if item[1]['type'] == 'simple':
                filters = item[1]['columns']
            else:
                filters = np.concat([json_groups[c]['columns'] for c in item[1]['groups']]).tolist()

            try:

                mask_privileged, mask_underprivileged = get_mask(input_data, filters)
                # columns features extract automaticaly quiting sensitive.
                # actual manul.
                data = input_data.copy()[train_columns]
                data['y'] = mask_underprivileged.astype(int)
                proportion_data = np.max([data['y'].sum(), data.shape[0] - data['y'].sum()]) / data['y'].shape[0]

                correlation_matrix = abs(data.corr())

                correlation = correlation_matrix.loc['y'][train_columns].max()
                rate = np.round(correlation, 4).item()
                normalized_risk = np.round(self.normalize_value(rate), 4).item()
                result_list.update({group: {'correlation':  np.round(correlation, 4).item(),
                                            'proportion': np.round(proportion_data, 4).item(),
                                            'rate': rate,
                                            'normalized_risk': normalized_risk,
                                            'bias_level': self.get_bias_level(normalized_risk)}})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")

                result_list.update({group: {'correlation': 0,
                                            'proportion': 0,
                                            'rate': 0,
                                            'normalized_risk': None,
                                            'bias_level': self.get_bias_level(0),
                                            'error': 'group no present in data.'}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list

    def normalize_value(self, value):
        '''
        Normalize the value between 80 - 100 if correlation is lower than 0.4
        Under 80 if value is higher than 0.4
        '''
        if value <= 0.25:
            # Normalización entre 80 y 100
            return np.min([100 - (value / 0.25 * 20), 100]).item()
        else:
            # Normalización entre 80 y 0
            return np.max([80 - ((value - 0.25) / 0.75 * 80), 0]).item()
