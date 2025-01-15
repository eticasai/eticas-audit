import numpy as np
from .base_metric import BaseFairnessMetric
from ..utils.helpers import get_mask
import logging
logger = logging.getLogger(__name__)


class Da_inconsistency(BaseFairnessMetric):
    """
    Metric D(A)-inconsistency
    Calculates the distribution of a given group
    """

    def compute(self,
                input_data,
                sensitive_attrs=None):
        """
        Parameters
        ----------
       :param input_data: Dataset which should contain the columns in param sensitive_attrs.
       :param sensitive_attrs: The sensitive attributes (e.g., 'gender', 'race') relevant to fairness.
        Returns
        ----------
       :return: A dictionary containing the results for each group
        """
        self.validate_parameters(
            input_data=input_data,
            sensitive_attrs=sensitive_attrs
        )
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
                data = np.round((mask_underprivileged.sum() / mask_underprivileged.shape[0])*100, 1).item()

                result_list.update({group: {'data': data}})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking

                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {'data': None,
                                            'error': 'group no present in data.'}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list
