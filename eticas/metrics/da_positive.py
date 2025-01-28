import numpy as np
from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask, binarize_predictions
import logging
logger = logging.getLogger(__name__)


class Da_positive(BaseFairnessMetric):
    """
    Metric D(A)-positive
    Calculates the distribution of a given group when label is positive.
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
                mask_underprivileged = mask_underprivileged[input_data[label_column].isin(positive_output)]
                # item is to convert np float64 to native python float.
                data = np.round((mask_underprivileged.sum() / mask_underprivileged.shape[0])*100, 1).item()
                result_list.update({group: {'data': data}})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {'data': None,
                                            'error': 'group no present in data.'}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list
