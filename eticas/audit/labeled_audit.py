"""
labeled_audit.py
=================

Implements an audit for the labeled phase of a model.
"""

from .base_audit import BaseAudit
from ..data.loaders import load_dataset

from ..metrics.da_inconsistency import Da_inconsistency
from ..metrics.da_positive import Da_positive
from ..metrics.dxa_inconsistency import Dxa_inconsistency
from ..metrics.da_informative import Da_informative
from ..metrics.d_statisticalparity import D_statisticalparity
from ..metrics.d_parity import D_parity
from ..metrics.performance import Performance
from ..metrics.d_equalodds import D_equalodds
from ..metrics.d_calibrated import D_calibrated
from ..metrics.da_fairness import Da_fairness
from ..metrics.disparate_impact import DisparateImpact
import logging
logger = logging.getLogger(__name__)


class LabeledAudit(BaseAudit):
    """
    Audit for the labeled / labeled phase of a model.
    """

    def run_audit(self,
                  dataset_path: str,
                  label_column: str,
                  output_column: str = None,
                  positive_output: list = None):
        """
        Performs checks related to the labeled / labeled  dataset and process.

       :param dataset_path: path to labeled dataset.
       :param label_column: Name of the column containing the target.
       :param output_column: Name of the column containing the prediction / classification.
       :param positive_output: Values of the column_output consider as positive.

        Returns
        -------
       :return: dict. The result of the labeled audit.
        """
        # Example: Check for data imbalance, missing values, etc.
        logger.info(f"Running labeled audit on model '{self.model.model_name}'")
        input_data = load_dataset(dataset_path)
        input_data = input_data.dropna()
        logger.info(f"labeled data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("labeled dataset shape is 0.")

        labeled_log = {}
        labeled_log.update({'da_inconsistency': Da_inconsistency().compute(input_data,
                                                                           self.model.sensitive_attributes)})
        labeled_log.update({'da_positive': Da_positive().compute(input_data,
                                                                 self.model.sensitive_attributes,
                                                                 label_column,
                                                                 positive_output)})
        labeled_log.update({'dxa_inconsistency': Dxa_inconsistency().compute(input_data,
                                                                             self.model.sensitive_attributes,
                                                                             self.model.features)})
        labeled_log.update({'da_informative': Da_informative().compute(input_data,
                                                                       self.model.sensitive_attributes,
                                                                       label_column,
                                                                       self.model.features)})
        labeled_log.update({'da_fairness': Da_fairness().compute(input_data,
                                                                 self.model.sensitive_attributes,
                                                                 self.model.features,
                                                                 label_column,
                                                                 positive_output)})
        labeled_log.update({'d_statisticalparity': D_statisticalparity().compute(input_data,
                                                                                 self.model.sensitive_attributes,
                                                                                 label_column,
                                                                                 positive_output)})
        labeled_log.update({'d_parity': D_parity().compute(input_data,
                                                           self.model.sensitive_attributes,
                                                           label_column,
                                                           positive_output)})
        labeled_log.update({'poor_performance': Performance().compute(input_data,
                                                                      self.model.sensitive_attributes,
                                                                      label_column,
                                                                      positive_output,
                                                                      output_column)})
        labeled_log.update({'d_equalodds': D_equalodds().compute(input_data,
                                                                 self.model.sensitive_attributes,
                                                                 label_column,
                                                                 positive_output, output_column)})
        labeled_log.update({'d_calibrated': D_calibrated().compute(input_data,
                                                                   self.model.sensitive_attributes,
                                                                   label_column,
                                                                   positive_output, output_column)})
        labeled_log.update({'impact_ratio': DisparateImpact().compute(input_data,
                                                                      self.model.sensitive_attributes,
                                                                      label_column,
                                                                      positive_output)})
        return labeled_log
