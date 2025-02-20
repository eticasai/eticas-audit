"""
production_audit.py
===================

Implements an audit for the production (or testing) phase of a model.
"""


from .base_audit import BaseAudit
from ..data.loaders import load_dataset

from ..metrics.da_inconsistency import Da_inconsistency
from ..metrics.da_positive import Da_positive
from ..metrics.dxa_inconsistency import Dxa_inconsistency
from ..metrics.da_informative import Da_informative
from ..metrics.d_statisticalparity import D_statisticalparity
from ..metrics.d_parity import D_parity
from ..metrics.da_fairness import Da_fairness
from ..metrics.disparate_impact import DisparateImpact
import logging
logger = logging.getLogger(__name__)


class UnlabeledAudit(BaseAudit):
    """
    Audit for the production/test phase of a model.
    """
    def run_audit(self,
                  dataset_path: str,
                  output_column: str = None,
                  positive_output: list = None):
        """
        Performs checks related to the unlabeled / production  dataset and process.

       :param dataset_path: path to production dataset.
       :param output_column: Name of the column containing the prediction / classification.
       :param positive_output: Values of the column_output consider as positive.

        Returns
        -------
       :return: dict. The result of the unlabeled audit.
        """
        # Example: Check for data imbalance, missing values, etc.
        logger.info(f"Running production audit on model '{self.model.model_name}'")
        input_data = load_dataset(dataset_path)
        input_data = input_data.dropna()
        logger.info(f"production data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("production dataset shape is 0.")

        unlabeled_log = {}
        unlabeled_log.update({'da_inconsistency': Da_inconsistency().compute(input_data,
                                                                             self.model.sensitive_attributes)})
        unlabeled_log.update({'da_positive': Da_positive().compute(input_data,
                                                                   self.model.sensitive_attributes,
                                                                   output_column,
                                                                   positive_output)})
        unlabeled_log.update({'dxa_inconsistency': Dxa_inconsistency().compute(input_data,
                                                                               self.model.sensitive_attributes,
                                                                               self.model.features)})
        unlabeled_log.update({'da_informative': Da_informative().compute(input_data,
                                                                         self.model.sensitive_attributes,
                                                                         output_column,
                                                                         self.model.features)})
        unlabeled_log.update({'da_fairness': Da_fairness().compute(input_data,
                                                                   self.model.sensitive_attributes,
                                                                   self.model.features,
                                                                   output_column,
                                                                   positive_output)})
        unlabeled_log.update({'d_statisticalparity': D_statisticalparity().compute(input_data,
                                                                                   self.model.sensitive_attributes,
                                                                                   output_column,
                                                                                   positive_output)})
        unlabeled_log.update({'d_parity': D_parity().compute(input_data,
                                                             self.model.sensitive_attributes,
                                                             output_column,
                                                             positive_output)})
        unlabeled_log.update({'impact_ratio': DisparateImpact().compute(input_data,
                                                                        self.model.sensitive_attributes,
                                                                        output_column,
                                                                        positive_output)})
        return unlabeled_log
