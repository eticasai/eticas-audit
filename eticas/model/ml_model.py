"""
ml_model.py
===========

Provides a concrete implementation of the BaseModel class
that focuses on running audits (labeled, production, impacted).
"""

from .base_model import BaseModel
from eticas.audit.labeled_audit import LabeledAudit
from eticas.audit.unlabeled_audit import UnlabeledAudit
from eticas.audit.drift_audit import DriftAudit
import logging
import warnings
import pandas as pd
from io import StringIO
logger = logging.getLogger(__name__)


class MLModel(BaseModel):
    """
    A generic ML model that extends BaseModel
    Methods to run different audits (labeled, production, impacted).
    """

    def run_labeled_audit(self,
                          dataset_path: str,
                          label_column: str,
                          output_column: str,
                          positive_output: list):
        """
        Runs a labeled audit using the model's metadata.

        Parameters
        ----------
       :param dataset_path: path to labeled dataset.
       :param label_column: Name of the column containing the target.
       :param output_column: Name of the column containing the prediction / classification.
       :param positive_output: Values of the column_output consider as positive.


        Returns
        -------
       :return: dict. The result of the labeled audit.
        """
        logger.info(f"Running labeled audit for model: {self.model_name}")
        audit = LabeledAudit(self)
        self.labeled_results = audit.run_audit(dataset_path,
                                               label_column,
                                               output_column,
                                               positive_output)
        logger.info(f"labeled audit finished for model: {self.model_name}")

    def run_production_audit(self,
                             dataset_path: str,
                             output_column: str,
                             positive_output: list):
        """
        Runs a production audit using the model's metadata.

        Parameters
        ----------
       :param dataset_path: path to production dataset.
       :param output_column: Name of the column containing the prediction / classification.
       :param positive_output: Values of the column_output consider as positive.


        Returns
        -------
       :return: dict. The result of the production audit.
        """
        logger.info(f"Running production audit for model: {self.model_name}")
        audit = UnlabeledAudit(self)
        self.production_results = audit.run_audit(dataset_path,
                                                  output_column,
                                                  positive_output)
        logger.info(f"Production audit finished for model: {self.model_name}")

    def run_impacted_audit(self,
                           dataset_path: str,
                           output_column: str,
                           positive_output: list):
        """
        Runs a impacted / recorded audit using the model's metadata.

        Parameters
        ----------
       :param dataset_path: path to impact dataset.
       :param output_column: Name of the column containing the prediction / classification.
       :param positive_output: Values of the column_output consider as positive.


        Returns
        -------
       :return: dict. The result of the impacted audit.
        """
        logger.info(f"Running Impacted audit for model: {self.model_name}")
        audit = UnlabeledAudit(self)
        self.impacted_results = audit.run_audit(dataset_path,
                                                output_column,
                                                positive_output)
        logger.info(f"Impacted audit finished for model: {self.model_name}")

    def run_drift_audit(self,
                        dataset_path_dev: str,
                        output_column_dev: str,
                        positive_output_dev: list,
                        dataset_path_prod: str,
                        output_column_prod: str,
                        positive_output_prod: list):
        """
        Runs a drift detector between two datasets.

        Parameters
        ----------
       :param dataset_path: path to  dataset.
       :param output_column: Name of the column containing the prediction / classification.
       :param positive_output: Values of the column_output consider as positive.

        Returns
        -------
       :return: dict. The result of the drift audit.
        """
        logger.info(f"Running Drift audit for model: {self.model_name}")
        audit = DriftAudit(self)
        self.drift_results = audit.run_audit(dataset_path_dev,
                                             output_column_dev,
                                             positive_output_dev,
                                             dataset_path_prod,
                                             output_column_prod,
                                             positive_output_prod)
        logger.info(f"Drift audit finished for model: {self.model_name}")

    def impact_ratio(self,
                     stage=None,
                     sensitive_attribute=None):
        """
        Get impact ratio.

        Parameters
        ----------
       :param stage: ['labeled','production','impacted'].
       :param sensitive_attribute: Name of the column include in sensitives attributes..
       :param positive_output: Values of the column_output consider as positive.

        Returns
        -------
       :return: DataFrame. The result of the impact ratio.
        """

        if stage == 'labeled':
            if sensitive_attribute in self.labeled_results['impact_ratio'].keys():
                return pd.read_json(StringIO(
                        self.labeled_results['impact_ratio'][sensitive_attribute]['df_result']),
                        orient='split')
            else:
                raise ValueError("You must provide a correct attribute.")
        elif stage == 'production':
            if sensitive_attribute in self.production_results['impact_ratio'].keys():
                return pd.read_json(StringIO(
                        self.production_results['impact_ratio'][sensitive_attribute]['df_result']),
                        orient='split')
            else:
                raise ValueError("You must provide a correct attribute.")
        elif stage == 'impacted':
            if sensitive_attribute in self.impacted_results['impact_ratio'].keys():
                return pd.read_json(StringIO(
                        self.impacted_results['impact_ratio'][sensitive_attribute]['df_result']),
                        orient='split')
            else:
                raise ValueError("You must provide a correct attribute.")
        else:
            raise ValueError("You must provide a correct stage.")

    def json_results(self, norm_values=True):

        """
        Return the results normalize between 0 (BAD) to 100 (GOOD) or the metric value."""

        if norm_values:
            return self.json_results_norm()
        else:
            return self.json_results_metric()

    def json_results_metric(self):
        """
        Aggregate audit results into json
        Returns
        _______
       :return: json with results.
        """
        protected = [[list(f[k].keys()) for k in f.keys()] for f in [self.labeled_results,
                                                                     self.production_results,
                                                                     self.impacted_results]]
        protected = [p[0] for p in protected if len(p) != 0]
        protected = list(set().union(*protected))
        audit_result = {}
        for p in protected:

            if p != 'error':

                audit_result[p] = {}
                audit_result[p]['benchmarking'] = {

                    **(
                        {} if not self.labeled_results else {
                            'labeled_da_inconsistency': self.labeled_results.get('da_inconsistency',
                                                                                 {}).get(p,
                                                                                         {}).get('data', None),
                            'labeled_da_positive': self.labeled_results.get('da_positive',
                                                                            {}).get(p,
                                                                                    {}).get('data', None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {
                            'operational_da_inconsistency': self.production_results.get('da_inconsistency',
                                                                                        {}).get(p,
                                                                                                {}).get('data', None),
                            'operational_da_positive': self.production_results.get('da_positive',
                                                                                   {}).get(p,
                                                                                           {}).get('data', None),
                        }
                    ),
                    **(
                        {} if not self.impacted_results else {
                            'impact_da_inconsistency': self.impacted_results.get('da_inconsistency',
                                                                                 {}).get(p,
                                                                                         {}).get('data', None),
                            'impact_da_positive': self.impacted_results.get('da_positive',
                                                                            {}).get(p,
                                                                                    {}).get('data', None),
                        }
                    ),
                }

                audit_result[p]['distribution'] = {
                    'ref': 80,
                    **(
                        {} if not self.labeled_results else {
                            'labeled_dxa_inconsistency': self.labeled_results.get('dxa_inconsistency',
                                                                                  {}).get(p,
                                                                                          {}).get('rate', None),
                            'labeled_da_informative': self.labeled_results.get('da_informative',
                                                                               {}).get(p,
                                                                                       {}).get('accuracy', None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {
                            'operational_dxa_inconsistency': self.production_results.get('dxa_inconsistency',
                                                                                         {}).get(p,
                                                                                                 {}).get('rate', None),
                            'operational_da_informative': self.production_results.get('da_informative',
                                                                                      {}).get(p,
                                                                                              {}).get('accuracy', None),
                        }
                    ),
                    **(
                        {} if not self.impacted_results else {
                            'impact_dxa_inconsistency': self.impacted_results.get('dxa_inconsistency',
                                                                                  {}).get(p,
                                                                                          {}).get('rate', None),
                            'impact_da_informative': self.impacted_results.get('da_informative',
                                                                               {}).get(p,
                                                                                       {}).get('accuracy', None),
                        }
                    ),
                }
                audit_result[p]['drift'] = {
                    'ref': 80,
                    **(
                        {} if not self.drift_results else {
                            'drift': self.drift_results.get('tdx_inconsistency',
                                                            {}).get(p,
                                                                    {}).get('accuracy', None),
                        }
                    ),
                }
                audit_result[p]['fairness'] = {
                    'ref': 80,
                    **(
                        {} if not self.labeled_results else {
                            'labeled_equality': self.labeled_results.get('da_fairness',
                                                                         {}).get(p,
                                                                                 {}).get('equality', None),
                            'labeled_equity': self.labeled_results.get('da_fairness',
                                                                       {}).get(p,
                                                                               {}).get('equity', None),
                            'labeled_DI': self.labeled_results.get('d_parity',
                                                                   {}).get(p,
                                                                           {}).get('DI', None),
                            'labeled_SPD': self.labeled_results.get('d_statisticalparity',
                                                                    {}).get(p,
                                                                            {}).get('SPD', None),
                            'labeled_TPR': self.labeled_results.get('d_equalodds',
                                                                    {}).get(p,
                                                                            {}).get('true_positive_rate',
                                                                                    {}).get('ratio_true', None),
                            'labeled_FPR': self.labeled_results.get('d_equalodds',
                                                                    {}).get(p,
                                                                            {}).get('false_positive_rate',
                                                                                    {}).get('ratio_false', None),
                            'labeled_PPV': self.labeled_results.get('d_calibrated',
                                                                    {}).get(p,
                                                                            {}).get('true_calibrated',
                                                                                    {}).get('ratio_true', None),
                            'labeled_PNV': self.labeled_results.get('d_calibrated',
                                                                    {}).get(p,
                                                                            {}).get('false_calibrated',
                                                                                    {}).get('ratio_false', None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {
                            'operational_equality': self.production_results.get('da_fairness',
                                                                                {}).get(p,
                                                                                        {}).get('equality', None),
                            'operational_equity': self.production_results.get('da_fairness',
                                                                              {}).get(p,
                                                                                      {}).get('equity', None),
                            'operational_DI': self.production_results.get('d_parity',
                                                                          {}).get(p,
                                                                                  {}).get('DI', None),
                            'operational_SPD': self.production_results.get('d_statisticalparity',
                                                                           {}).get(p,
                                                                                   {}).get('SPD', None),
                        }
                    ),

                    **(
                        {} if not self.impacted_results else {
                            'impact_equality': self.impacted_results.get('da_fairness',
                                                                         {}).get(p,
                                                                                 {}).get('equality', None),
                            'impact_equity': self.impacted_results.get('da_fairness',
                                                                       {}).get(p,
                                                                               {}).get('equity', None),
                            'impact_DI': self.impacted_results.get('d_parity',
                                                                   {}).get(p,
                                                                           {}).get('DI', None),
                            'impact_SPD': self.impacted_results.get('d_statisticalparity',
                                                                    {}).get(p,
                                                                            {}).get('SPD', None),
                        }
                    ),
                }

                audit_result[p]['performance'] = {
                    'ref': 80,
                    **(
                        {} if not self.labeled_results else {
                            'poor_performance': self.labeled_results.get('poor_performance',
                                                                         {}).get(p,
                                                                                 {}).get('normalized_risk', None),
                            'recall': self.labeled_results.get('poor_performance',
                                                               {}).get(p,
                                                                       {}).get('recall', None),
                            'f1_score': self.labeled_results.get('poor_performance',
                                                                 {}).get(p,
                                                                         {}).get('f1', None),
                            'accuracy': self.labeled_results.get('poor_performance',
                                                                 {}).get(p,
                                                                         {}).get('accuracy', None),
                            'precision': self.labeled_results.get('poor_performance',
                                                                  {}).get(p,
                                                                          {}).get('precision', None),
                            'TP': self.labeled_results.get('poor_performance',
                                                            {}).get(p,
                                                                    {}).get('TP', None),
                            'FP': self.labeled_results.get('poor_performance',
                                                           {}).get(p,
                                                                   {}).get('FP', None),
                            'TN': self.labeled_results.get('poor_performance',
                                                           {}).get(p,
                                                                   {}).get('TN', None),
                            'FN': self.labeled_results.get('poor_performance',
                                                           {}).get(p,
                                                                   {}).get('FN', None),
                        }
                    ),
                }
        return audit_result

    def json_results_norm(self):
        """
        Aggregate audit results into json
        Returns
        _______
       :return: json with results.
        """
        protected = [[list(f[k].keys()) for k in f.keys()] for f in [self.labeled_results,
                                                                     self.production_results,
                                                                     self.impacted_results]]
        protected = [p[0] for p in protected if len(p) != 0]
        protected = list(set().union(*protected))
        protected = [p for p in protected if p != 'error']
        audit_result = {}
        for p in protected:
            audit_result[p] = {}
            audit_result[p]['benchmarking'] = {
                **(
                    {} if not self.labeled_results else {
                        'labeled_da_inconsistency': self.labeled_results.get('da_inconsistency',
                                                                             {}).get(p,
                                                                                     {}).get('data', None),
                        'labeled_da_positive': self.labeled_results.get('da_positive',
                                                                        {}).get(p,
                                                                                {}).get('data', None),
                    }
                ),
                **(
                    {} if not self.production_results else {
                        'operational_da_inconsistency': self.production_results.get('da_inconsistency',
                                                                                    {}).get(p,
                                                                                            {}).get('data', None),
                        'operational_da_positive': self.production_results.get('da_positive',
                                                                               {}).get(p,
                                                                                       {}).get('data', None),
                    }
                ),
                **(
                    {} if not self.impacted_results else {
                        'impact_da_inconsistency': self.impacted_results.get('da_inconsistency',
                                                                             {}).get(p,
                                                                                     {}).get('data', None),
                        'impact_da_positive': self.impacted_results.get('da_positive',
                                                                        {}).get(p,
                                                                                {}).get('data', None),
                    }
                ),
            }
            audit_result[p]['distribution'] = {
                'ref': 80,
                **(
                    {} if not self.labeled_results else {
                        'labeled_dxa_inconsistency': self.labeled_results.get('dxa_inconsistency',
                                                                              {}).get(p,
                                                                                      {}).get('normalized_risk',
                                                                                              None),
                        'labeled_da_informative': self.labeled_results.get('da_informative',
                                                                           {}).get(p,
                                                                                   {}).get('normalized_risk',
                                                                                           None),
                    }
                ),
                **(
                    {} if not self.production_results else {
                        'operational_dxa_inconsistency': self.production_results.get('dxa_inconsistency',
                                                                                     {}).get(p,
                                                                                             {}).get('normalized_risk',
                                                                                                     None),
                        'operational_da_informative': self.production_results.get('da_informative',
                                                                                  {}).get(p,
                                                                                          {}).get('normalized_risk',
                                                                                                  None),
                    }
                ),
                **(
                    {} if not self.impacted_results else {
                        'impact_dxa_inconsistency': self.impacted_results.get('dxa_inconsistency',
                                                                              {}).get(p,
                                                                                      {}).get('normalized_risk', None),
                        'impact_da_informative': self.impacted_results.get('da_informative',
                                                                           {}).get(p,
                                                                                   {}).get('normalized_risk', None),
                    }
                ),
            }
            audit_result[p]['drift'] = {
                'ref': 80,
                **(
                    {} if not self.drift_results else {
                        'drift': self.drift_results.get('tdx_inconsistency',
                                                        {}).get(p,
                                                                {}).get('normalized_risk', None),

                    }
                ),
            }
            audit_result[p]['fairness'] = {
                'ref': 80,
                **(
                    {} if not self.labeled_results else {
                        'labeled_equality': self.labeled_results.get('da_fairness',
                                                                     {}).get(p,
                                                                             {}).get('equality_norm', None),
                        'labeled_equity': self.labeled_results.get('da_fairness',
                                                                   {}).get(p,
                                                                           {}).get('equity_norm', None),
                        'labeled_DI': self.labeled_results.get('d_parity',
                                                               {}).get(p,
                                                                       {}).get('normalized_risk', None),
                        'labeled_SPD': self.labeled_results.get('d_statisticalparity',
                                                                {}).get(p,
                                                                        {}).get('normalized_risk', None),
                        'labeled_TPR': self.labeled_results.get('d_equalodds',
                                                                {}).get(p,
                                                                        {}).get('true_positive_rate',
                                                                                {}).get('normalized_risk', None),
                        'labeled_FPR': self.labeled_results.get('d_equalodds',
                                                                {}).get(p,
                                                                        {}).get('false_positive_rate',
                                                                                {}).get('normalized_risk', None),
                        'labeled_PPV': self.labeled_results.get('d_calibrated',
                                                                {}).get(p,
                                                                        {}).get('true_calibrated',
                                                                                {}).get('normalized_risk', None),
                        'labeled_PNV': self.labeled_results.get('d_calibrated',
                                                                {}).get(p,
                                                                        {}).get('false_calibrated',
                                                                                {}).get('normalized_risk', None),
                    }
                ),
                **(
                    {} if not self.production_results else {
                        'operational_equality': self.production_results.get('da_fairness',
                                                                            {}).get(p,
                                                                                    {}).get('equality_norm', None),
                        'operational_equity': self.production_results.get('da_fairness',
                                                                          {}).get(p,
                                                                                  {}).get('equity_norm', None),
                        'operational_DI': self.production_results.get('d_parity',
                                                                      {}).get(p,
                                                                              {}).get('normalized_risk', None),
                        'operational_SPD': self.production_results.get('d_statisticalparity',
                                                                       {}).get(p,
                                                                               {}).get('normalized_risk', None),
                    }
                ),

                **(
                    {} if not self.impacted_results else {
                        'impact_equality': self.impacted_results.get('da_fairness',
                                                                     {}).get(p,
                                                                             {}).get('equality_norm', None),
                        'impact_equity': self.impacted_results.get('da_fairness',
                                                                   {}).get(p,
                                                                           {}).get('equity_norm', None),
                        'impact_DI': self.impacted_results.get('d_parity',
                                                               {}).get(p,
                                                                       {}).get('normalized_risk', None),
                        'impact_SPD': self.impacted_results.get('d_statisticalparity',
                                                                {}).get(p,
                                                                        {}).get('normalized_risk', None),
                    }
                ),
            }
            audit_result[p]['performance'] = {
                'ref': 80,
                **(
                    {} if not self.labeled_results else {
                        'poor_performance': self.labeled_results.get('poor_performance',
                                                                     {}).get(p,
                                                                             {}).get('normalized_risk', None),
                        'recall': self.labeled_results.get('poor_performance',
                                                           {}).get(p,
                                                                   {}).get('recall', None),
                        'f1_score': self.labeled_results.get('poor_performance',
                                                             {}).get(p,
                                                                     {}).get('f1', None),
                        'accuracy': self.labeled_results.get('poor_performance',
                                                             {}).get(p,
                                                                     {}).get('accuracy', None),
                        'precision': self.labeled_results.get('poor_performance',
                                                              {}).get(p,
                                                                      {}).get('precision', None),
                        'TP': self.labeled_results.get('poor_performance',
                                                        {}).get(p,
                                                                {}).get('TP', None),
                        'FP': self.labeled_results.get('poor_performance',
                                                       {}).get(p,
                                                               {}).get('FP', None),
                        'TN': self.labeled_results.get('poor_performance',
                                                       {}).get(p,
                                                               {}).get('TN', None),
                        'FN': self.labeled_results.get('poor_performance',
                                                       {}).get(p,
                                                               {}).get('FN', None),
                    }
                ),
            }
        return audit_result

    def df_results(self, norm_values=True):
        """
        Aggregate audit results into df

        Returns
        _______
       :return: dataframe with results.
        """
        if norm_values:
            return self.df_results_norm()
        else:
            return self.df_results_metric()

    def df_results_norm(self):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=pd.errors.PerformanceWarning)
            df = pd.DataFrame(
                columns=['group', 'metric', 'attribute', 'stage', 'value'])
            df = df.set_index(['group', 'metric', 'attribute', 'stage'])
            df.sort_index(inplace=True)
            protected = [[list(f[k].keys()) for k in f.keys()] for f in [
                self.labeled_results, self.production_results, self.impacted_results]]
            protected = [p[0] for p in protected if len(p) != 0]
            protected = list(set().union(*protected))
            protected = [p for p in protected if p != 'error']
            for p in protected:
                if self.labeled_results:
                    d_equality = self.labeled_results.get('da_fairness',
                                                          {}).get(p,
                                                                  {}).get('equality_norm', None)
                    d_equity = self.labeled_results.get('da_fairness',
                                                        {}).get(p,
                                                                {}).get('equity_norm', None)
                    da_inconsistency = self.labeled_results.get('da_inconsistency',
                                                                {}).get(p,
                                                                        {}).get('data', None)
                    da_positive = self.labeled_results.get('da_positive',
                                                           {}).get(p,
                                                                   {}).get('data', None)
                    dxa_inconsistency = self.labeled_results.get('dxa_inconsistency',
                                                                 {}).get(p,
                                                                         {}).get('normalized_risk', None)
                    da_informative = self.labeled_results.get('da_informative',
                                                              {}).get(p,
                                                                      {}).get('normalized_risk', None)
                    d_parity = self.labeled_results.get('d_parity',
                                                        {}).get(p,
                                                                {}).get('normalized_risk', None)
                    d_statisticalparity = self.labeled_results.get('d_statisticalparity',
                                                                   {}).get(p,
                                                                           {}).get('normalized_risk', None)
                    d_equalodds_true = self.labeled_results.get('d_equalodds',
                                                                {}).get(p,
                                                                        {}).get('true_positive_rate',
                                                                                {}).get('normalized_risk', None)
                    d_equalodds_false = self.labeled_results.get('d_equalodds',
                                                                 {}).get(p,
                                                                         {}).get('false_positive_rate',
                                                                                 {}).get('normalized_risk', None)
                    d_calibrated_true = self.labeled_results.get('d_calibrated',
                                                                 {}).get(p,
                                                                         {}).get('true_calibrated',
                                                                                 {}).get('normalized_risk', None)
                    d_calibrated_false = self.labeled_results.get('d_calibrated',
                                                                  {}).get(p,
                                                                          {}).get('false_calibrated',
                                                                                  {}).get('normalized_risk', None)
                    poor_performance = self.labeled_results.get('poor_performance',
                                                                {}).get(p,
                                                                        {}).get('normalized_risk', None)
                    recall = self.labeled_results.get('poor_performance',
                                                      {}).get(p,
                                                              {}).get('recall', None)
                    f1 = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('f1', None)
                    accuracy = self.labeled_results.get('poor_performance',
                                                        {}).get(p,
                                                                {}).get('accuracy', None)
                    precision = self.labeled_results.get('poor_performance',
                                                         {}).get(p,
                                                                 {}).get('precision', None)
                    TP = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('TP', None)
                    FP = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('FP', None)
                    TN = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('TN', None)
                    FN = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('FN', None)
                    df.loc[('benchmarking', 'da_inconsistency', p,
                            '01-labeled'), 'value'] = da_inconsistency
                    df.loc[('benchmarking', 'da_positive', p,
                            '01-labeled'), 'value'] = da_positive
                    df.loc[('distribution', 'dxa_inconsistency', p,
                            '01-labeled'), 'value'] = dxa_inconsistency
                    df.loc[('distribution', 'da_informative', p,
                            '01-labeled'), 'value'] = da_informative
                    df.loc[('fairness', 'd_equality', p, '01-labeled'),
                           'value'] = d_equality
                    df.loc[('fairness', 'd_equity', p, '01-labeled'),
                           'value'] = d_equity
                    df.loc[('fairness', 'd_parity', p, '01-labeled'),
                           'value'] = d_parity
                    df.loc[('fairness', 'd_statisticalparity', p, '01-labeled'),
                           'value'] = d_statisticalparity
                    df.loc[('fairness_label', 'd_equalodds_true', p, '01-labeled'),
                           'value'] = d_equalodds_true
                    df.loc[('fairness_label', 'd_equalodds_false', p, '01-labeled'),
                           'value'] = d_equalodds_false
                    df.loc[('fairness_label', 'd_calibrated_true', p, '01-labeled'),
                           'value'] = d_calibrated_true
                    df.loc[('fairness_label', 'd_calibrated_false', p, '01-labeled'),
                           'value'] = d_calibrated_false
                    df.loc[('performance', 'poor_performance', p,
                            '01-labeled'), 'value'] = poor_performance
                    df.loc[('performance', 'recall', p,
                            '01-labeled'), 'value'] = recall
                    df.loc[('performance', 'f1', p, '01-labeled'), 'value'] = f1
                    df.loc[('performance', 'accuracy', p,
                            '01-labeled'), 'value'] = accuracy
                    df.loc[('performance', 'precision', p,
                            '01-labeled'), 'value'] = precision
                    df.loc[('performance', 'TP', p, '01-labeled'), 'value'] = TP
                    df.loc[('performance', 'FP', p, '01-labeled'), 'value'] = FP
                    df.loc[('performance', 'TN', p, '01-labeled'), 'value'] = TN
                    df.loc[('performance', 'FN', p, '01-labeled'), 'value'] = FN
                if self.production_results:
                    d_equality = self.production_results.get('da_fairness',
                                                             {}).get(p,
                                                                     {}).get('equality_norm', None)
                    d_equity = self.production_results.get('da_fairness',
                                                           {}).get(p,
                                                                   {}).get('equity_norm', None)
                    da_inconsistency = self.production_results.get('da_inconsistency',
                                                                   {}).get(p,
                                                                           {}).get('data', None)
                    da_positive = self.production_results.get('da_positive',
                                                              {}).get(p,
                                                                      {}).get('data', None)
                    dxa_inconsistency = self.production_results.get('dxa_inconsistency',
                                                                    {}).get(p,
                                                                            {}).get('normalized_risk', None)
                    da_informative = self.production_results.get('da_informative',
                                                                 {}).get(p,
                                                                         {}).get('normalized_risk', None)
                    d_parity = self.production_results.get('d_parity',
                                                           {}).get(p,
                                                                   {}).get('normalized_risk', None)
                    d_statisticalparity = self.production_results.get('d_statisticalparity',
                                                                      {}).get(p,
                                                                              {}).get('normalized_risk', None)
                    df.loc[('benchmarking', 'da_inconsistency', p,
                            '02-production'), 'value'] = da_inconsistency
                    df.loc[('benchmarking', 'da_positive', p,
                            '02-production'), 'value'] = da_positive
                    df.loc[('distribution', 'dxa_inconsistency', p,
                            '02-production'), 'value'] = dxa_inconsistency
                    df.loc[('distribution', 'da_informative', p,
                            '02-production'), 'value'] = da_informative
                    df.loc[('fairness', 'd_equality', p, '02-production'),
                           'value'] = d_equality
                    df.loc[('fairness', 'd_equity', p, '02-production'),
                           'value'] = d_equity
                    df.loc[('fairness', 'd_parity', p, '02-production'),
                           'value'] = d_parity
                    df.loc[('fairness', 'd_statisticalparity', p,
                            '02-production'), 'value'] = d_statisticalparity
                if self.impacted_results:
                    d_equality = self.impacted_results.get('da_fairness',
                                                           {}).get(p,
                                                                   {}).get('equality_norm', None)
                    d_equity = self.impacted_results.get('da_fairness',
                                                         {}).get(p,
                                                                 {}).get('equity_norm', None)
                    da_inconsistency = self.impacted_results.get('da_inconsistency',
                                                                 {}).get(p,
                                                                         {}).get('data', None)
                    da_positive = self.impacted_results.get('da_positive',
                                                            {}).get(p,
                                                                    {}).get('data', None)
                    dxa_inconsistency = self.impacted_results.get('dxa_inconsistency',
                                                                  {}).get(p,
                                                                          {}).get('normalized_risk', None)
                    da_informative = self.impacted_results.get('da_informative',
                                                               {}).get(p,
                                                                       {}).get('normalized_risk', None)
                    d_parity = self.impacted_results.get('d_parity',
                                                         {}).get(p,
                                                                 {}).get('normalized_risk', None)
                    d_statisticalparity = self.impacted_results.get('d_statisticalparity',
                                                                    {}).get(p,
                                                                            {}).get('normalized_risk', None)

                    df.loc[('benchmarking', 'da_inconsistency', p,
                            '03-impact'), 'value'] = da_inconsistency
                    df.loc[('benchmarking', 'da_positive', p,
                            '03-impact'), 'value'] = da_positive
                    df.loc[('distribution', 'dxa_inconsistency', p,
                            '03-impact'), 'value'] = dxa_inconsistency
                    df.loc[('distribution', 'da_informative', p,
                            '03-impact'), 'value'] = da_informative
                    df.loc[('fairness', 'd_equality', p, '03-impact'),
                           'value'] = d_equality
                    df.loc[('fairness', 'd_equity', p, '03-impact'),
                           'value'] = d_equity
                    df.loc[('fairness', 'd_parity', p, '03-impact'),
                           'value'] = d_parity
                    df.loc[('fairness', 'd_statisticalparity', p,
                            '03-impact'), 'value'] = d_statisticalparity
                if self.drift_results:
                    drift = self.drift_results.get('tdx_inconsistency',
                                                   {}).get(p,
                                                           {}).get('normalized_risk', None)
                    df.loc[('drift', 'drift', p, '02-production'), 'value'] = drift

        if self.drift_results:
            drift = self.drift_results.get('tdx_inconsistency',
                                           {}).get('overall',
                                                   {}).get('normalized_risk', None)
            df = df.sort_index()
            df.loc[('drift', 'drift', 'overall', '02-production'), 'value'] = drift

        df.sort_index(inplace=True)

        return df

    def df_results_metric(self):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",  category=pd.errors.PerformanceWarning)
            df = pd.DataFrame(
                columns=['group', 'metric', 'attribute', 'stage', 'value'])
            df = df.set_index(['group', 'metric', 'attribute', 'stage'])
            df.sort_index(inplace=True)
            protected = [[list(f[k].keys()) for k in f.keys()] for f in [
                self.labeled_results, self.production_results, self.impacted_results]]
            protected = [p[0] for p in protected if len(p) != 0]
            protected = list(set().union(*protected))
            for p in protected:
                if self.labeled_results:
                    d_equality = self.labeled_results.get('da_fairness',
                                                          {}).get(p,
                                                                  {}).get('equality', None)
                    d_equity = self.labeled_results.get('da_fairness',
                                                        {}).get(p,
                                                                {}).get('equity', None)
                    da_inconsistency = self.labeled_results.get('da_inconsistency',
                                                                {}).get(p,
                                                                        {}).get('data', None)
                    da_positive = self.labeled_results.get('da_positive',
                                                           {}).get(p,
                                                                   {}).get('data', None)
                    dxa_inconsistency = self.labeled_results.get('dxa_inconsistency',
                                                                 {}).get(p,
                                                                         {}).get('rate', None)
                    da_informative = self.labeled_results.get('da_informative',
                                                              {}).get(p,
                                                                      {}).get('accuracy', None)
                    d_parity = self.labeled_results.get('d_parity',
                                                        {}).get(p,
                                                                {}).get('DI', None)
                    d_statisticalparity = self.labeled_results.get('d_statisticalparity',
                                                                   {}).get(p,
                                                                           {}).get('SPD', None)
                    d_equalodds_true = self.labeled_results.get('d_equalodds',
                                                                {}).get(p,
                                                                        {}).get('true_positive_rate',
                                                                                {}).get('ratio_true', None)
                    d_equalodds_false = self.labeled_results.get('d_equalodds',
                                                                 {}).get(p,
                                                                         {}).get('false_positive_rate',
                                                                                 {}).get('ratio_false', None)
                    d_calibrated_true = self.labeled_results.get('d_calibrated',
                                                                 {}).get(p,
                                                                         {}).get('true_calibrated',
                                                                                 {}).get('ratio_true', None)
                    d_calibrated_false = self.labeled_results.get('d_calibrated',
                                                                  {}).get(p,
                                                                          {}).get('false_calibrated',
                                                                                  {}).get('ratio_false', None)
                    poor_performance = self.labeled_results.get('poor_performance',
                                                                {}).get(p,
                                                                        {}).get('normalized_risk', None)
                    recall = self.labeled_results.get('poor_performance',
                                                      {}).get(p,
                                                              {}).get('recall', None)
                    f1 = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('f1', None)
                    accuracy = self.labeled_results.get('poor_performance',
                                                        {}).get(p,
                                                                {}).get('accuracy', None)
                    precision = self.labeled_results.get('poor_performance',
                                                         {}).get(p,
                                                                 {}).get('precision', None)
                    TP = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('TP', None)
                    FP = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('FP', None)
                    TN = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('TN', None)
                    FN = self.labeled_results.get('poor_performance',
                                                  {}).get(p,
                                                          {}).get('FN', None)
                    df.loc[('benchmarking', 'da_inconsistency', p,
                            '01-labeled'), 'value'] = da_inconsistency
                    df.loc[('benchmarking', 'da_positive', p,
                            '01-labeled'), 'value'] = da_positive
                    df.loc[('distribution', 'dxa_inconsistency', p,
                            '01-labeled'), 'value'] = dxa_inconsistency
                    df.loc[('distribution', 'da_informative', p,
                            '01-labeled'), 'value'] = da_informative
                    df.loc[('fairness', 'd_equality', p, '01-labeled'),
                           'value'] = d_equality
                    df.loc[('fairness', 'd_equity', p, '01-labeled'),
                           'value'] = d_equity
                    df.loc[('fairness', 'd_parity', p, '01-labeled'),
                           'value'] = d_parity
                    df.loc[('fairness', 'd_statisticalparity', p,
                            '01-labeled'), 'value'] = d_statisticalparity
                    df.loc[('fairness', 'd_equalodds_true', p,
                            '01-labeled'), 'value'] = d_equalodds_true
                    df.loc[('fairness', 'd_equalodds_false', p,
                            '01-labeled'), 'value'] = d_equalodds_false
                    df.loc[('fairness', 'd_calibrated_true', p,
                            '01-labeled'), 'value'] = d_calibrated_true
                    df.loc[('fairness', 'd_calibrated_false', p,
                            '01-labeled'), 'value'] = d_calibrated_false
                    df.loc[('performance', 'poor_performance', p,
                            '01-labeled'), 'value'] = poor_performance
                    df.loc[('performance', 'recall', p,
                            '01-labeled'), 'value'] = recall
                    df.loc[('performance', 'f1', p, '01-labeled'), 'value'] = f1
                    df.loc[('performance', 'accuracy', p,
                            '01-labeled'), 'value'] = accuracy
                    df.loc[('performance', 'precision', p,
                            '01-labeled'), 'value'] = precision
                    df.loc[('performance', 'TP', p, '01-labeled'), 'value'] = TP
                    df.loc[('performance', 'FP', p, '01-labeled'), 'value'] = FP
                    df.loc[('performance', 'TN', p, '01-labeled'), 'value'] = TN
                    df.loc[('performance', 'FN', p, '01-labeled'), 'value'] = FN
                if self.production_results:
                    d_equality = self.production_results.get('da_fairness',
                                                             {}).get(p,
                                                                     {}).get('equality', None)
                    d_equity = self.production_results.get('da_fairness',
                                                           {}).get(p,
                                                                   {}).get('equity', None)
                    da_inconsistency = self.production_results.get(
                        'da_inconsistency', {}).get(p, {}).get('data', None)
                    da_positive = self.production_results.get(
                        'da_positive', {}).get(p, {}).get('data', None)
                    dxa_inconsistency = self.production_results.get(
                        'dxa_inconsistency', {}).get(p, {}).get('rate', None)
                    da_informative = self.production_results.get(
                        'da_informative', {}).get(p, {}).get('accuracy', None)
                    d_parity = self.production_results.get(
                        'd_parity', {}).get(p, {}).get('DI', None)
                    d_statisticalparity = self.production_results.get(
                        'd_statisticalparity', {}).get(p, {}).get('SPD', None)
                    df.loc[('benchmarking', 'da_inconsistency', p,
                            '02-production'), 'value'] = da_inconsistency
                    df.loc[('benchmarking', 'da_positive', p,
                            '02-production'), 'value'] = da_positive
                    df.loc[('distribution', 'dxa_inconsistency', p,
                            '02-production'), 'value'] = dxa_inconsistency
                    df.loc[('distribution', 'da_informative', p,
                            '02-production'), 'value'] = da_informative
                    df.loc[('fairness', 'd_equality', p, '02-production'),
                           'value'] = d_equality
                    df.loc[('fairness', 'd_equity', p, '02-production'),
                           'value'] = d_equity
                    df.loc[('fairness', 'd_parity', p, '02-production'),
                           'value'] = d_parity
                    df.loc[('fairness', 'd_statisticalparity', p,
                            '02-production'), 'value'] = d_statisticalparity
                if self.impacted_results:
                    d_equality = self.impacted_results.get('da_fairness',
                                                           {}).get(p,
                                                                   {}).get('equality', None)
                    d_equity = self.impacted_results.get('da_fairness',
                                                         {}).get(p,
                                                                 {}).get('equity', None)
                    da_inconsistency = self.impacted_results.get('da_inconsistency',
                                                                 {}).get(p,
                                                                         {}).get('data', None)
                    da_positive = self.impacted_results.get('da_positive',
                                                            {}).get(p,
                                                                    {}).get('data', None)
                    dxa_inconsistency = self.impacted_results.get('dxa_inconsistency',
                                                                  {}).get(p,
                                                                          {}).get('rate', None)
                    da_informative = self.impacted_results.get('da_informative',
                                                               {}).get(p,
                                                                       {}).get('accuracy', None)
                    d_parity = self.impacted_results.get('d_parity',
                                                         {}).get(p,
                                                                 {}).get('DI', None)
                    d_statisticalparity = self.impacted_results.get('d_statisticalparity',
                                                                    {}).get(p,
                                                                            {}).get('SPD', None)
                    df.loc[('benchmarking', 'da_inconsistency', p,
                            '03-impact'), 'value'] = da_inconsistency
                    df.loc[('benchmarking', 'da_positive', p,
                            '03-impact'), 'value'] = da_positive
                    df.loc[('distribution', 'dxa_inconsistency', p,
                            '03-impact'), 'value'] = dxa_inconsistency
                    df.loc[('distribution', 'da_informative', p,
                            '03-impact'), 'value'] = da_informative
                    df.loc[('fairness', 'd_equality', p, '03-impact'),
                           'value'] = d_equality
                    df.loc[('fairness', 'd_equity', p, '03-impact'),
                           'value'] = d_equity
                    df.loc[('fairness', 'd_parity', p, '03-impact'),
                           'value'] = d_parity
                    df.loc[('fairness', 'd_statisticalparity', p,
                            '03-impact'), 'value'] = d_statisticalparity
                if self.drift_results:
                    df.loc[('drift', 'drift', p, '02-production'), 'value'] = self.drift_results.get(
                        'tdx_inconsistency', {}).get(p, {}).get('accuracy', None)

        if self.drift_results:
            df = df.sort_index()
            df.loc[('drift', 'drift', 'overall', '02-production'), 'value'] = self.drift_results.get(
                'tdx_inconsistency', {}).get('overall', {}).get('accuracy', None)
        df.sort_index(inplace=True)

        return df
