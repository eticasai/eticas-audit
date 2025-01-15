"""
production_audit.py
===================

Implements an audit for the production (or testing) phase of a model.
"""


from .base_audit import BaseAudit
from ..data.loaders import load_dataset

from ..metrics.tdx_inconsistency import Tdx_inconsistency
import logging
logger = logging.getLogger(__name__)


class DriftAudit(BaseAudit):
    """
    Drift Audit.
    """
    def run_audit(self,
                  dataset_path_dev: str,
                  output_column_dev: str,
                  positive_output_dev: list,
                  dataset_path_prod: str,
                  output_column_prod: str,
                  positive_output_prod: list):
        """
        Performs checks related to the unlabeled / production  dataset and process.

       :param dataset_path: path to dataset.
       :param output_column: Name of the column containing the prediction / classification.
       :param positive_output: Values of the column_output consider as positive.

        Returns
        -------
       :return: dict. The result of the unlabeled audit.
        """
        # Example: Check for data imbalance, missing values, etc.
        logger.info(f"Running drift audit on model '{self.model.model_name}'")
        input_data_dev = load_dataset(dataset_path_dev)
        input_data_dev = input_data_dev.dropna()
        logger.info(f"DEV data loaded '{input_data_dev.shape}'")
        if input_data_dev.shape[0] == 0:
            raise ValueError("DEV data shape is 0.")

        input_data_prod = load_dataset(dataset_path_prod)
        input_data_prod = input_data_prod.dropna()
        logger.info(f"PROD data loaded '{input_data_prod.shape}'")
        if input_data_prod.shape[0] == 0:
            raise ValueError("PROD data shape is 0.")

        drift_log = {}
        drift_log.update({'tdx_inconsistency': Tdx_inconsistency().compute(self.model.sensitive_attributes,
                                                                           self.model.features,
                                                                           input_data_dev,
                                                                           output_column_dev,
                                                                           positive_output_dev,
                                                                           input_data_prod,
                                                                           output_column_prod,
                                                                           positive_output_prod)})
        return drift_log
