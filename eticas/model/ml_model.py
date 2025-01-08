"""
ml_model.py
===========

Provides a concrete implementation of the BaseModel class
that focuses on running audits (training, production, impacted).
"""

from .base_model import BaseModel
from eticas.audit.labeled_audit import LabeledAudit
from eticas.audit.unlabeled_audit import UnlabeledAudit
from eticas.audit.drift_audit import DriftAudit
import logging
import warnings
import pandas as pd



logger = logging.getLogger(__name__)

class MLModel(BaseModel):
    """
    A generic ML model that extends BaseModel
    Methods to run different audits (training, production, impacted).
    """

    def run_training_audit(self, dataset_path : str, 
                           label_column : str, output_column : str, positive_output : list):
        """
        Runs a training audit using the model's metadata.

        Parameters
        ----------
        :param dataset_path : path to training dataset.
        :param label_column: Name of the column containing the target.
        :param output_column: Name of the column containing the prediction / classification.
        :param positive_output: Values of the column_output consider as positive.


        Returns
        -------
        :return: dict. The result of the training audit.
        """
        logger.info(f"Running training audit for model: {self.model_name}")
        audit = LabeledAudit(self)
        self.training_results =  audit.run_audit(dataset_path,label_column,output_column,positive_output)
        logger.info(f"training audit finished for model: {self.model_name}")

    def run_production_audit(self, dataset_path : str, 
                            output_column : str, positive_output : list):
        """
        Runs a production audit using the model's metadata.

        Parameters
        ----------
        :param dataset_path : path to production dataset.
        :param output_column: Name of the column containing the prediction / classification.
        :param positive_output: Values of the column_output consider as positive.


        Returns
        -------
        :return: dict. The result of the production audit.
        """
        logger.info(f"Running production audit for model: {self.model_name}")
        audit = UnlabeledAudit(self)
        self.production_results =  audit.run_audit(dataset_path,output_column,positive_output)
        logger.info(f"Production audit finished for model: {self.model_name}")

    def run_impacted_audit(self, dataset_path : str, 
                            output_column : str, positive_output : list):
        """
        Runs a impacted / recorded audit using the model's metadata.

        Parameters
        ----------
        :param dataset_path : path to impact dataset.
        :param output_column: Name of the column containing the prediction / classification.
        :param positive_output: Values of the column_output consider as positive.


        Returns
        -------
        :return: dict. The result of the impacted audit.
        """
        logger.info(f"Running Impacted audit for model: {self.model_name}")
        audit = UnlabeledAudit(self)
        self.impacted_results =  audit.run_audit(dataset_path,output_column,positive_output)
        logger.info(f"Impacted audit finished for model: {self.model_name}")

    def run_drift_audit(self, 
                    dataset_path_dev : str, output_column_dev : str, positive_output_dev : list,
                    dataset_path_prod : str, output_column_prod : str, positive_output_prod : list):
        """
        Runs a drift detector between two datasets.

        Parameters
        ----------
        :param dataset_path : path to  dataset.
        :param output_column: Name of the column containing the prediction / classification.
        :param positive_output: Values of the column_output consider as positive.

        Returns
        -------
        :return: dict. The result of the drift audit.
        """
        logger.info(f"Running Drift audit for model: {self.model_name}")
        audit = DriftAudit(self)
        self.drift_results =  audit.run_audit(dataset_path_dev,output_column_dev, positive_output_dev,
                                              dataset_path_prod, output_column_prod, positive_output_prod)
        logger.info(f"Drift audit finished for model: {self.model_name}")

    ### to do report API KEY ITACA


    def json_results(self,norm_values = True):
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
        protected = [[list(f[k].keys()) for k in f.keys()] for f in [self.training_results,self.production_results,self.impacted_results]]
        protected = [p[0] for p in protected if len(p) != 0]
        protected = list(set().union(*protected))
        audit_result = {}
        for p in protected:
            if p != 'error':

                audit_result[p] = {}
                audit_result[p]['benchmarking'] = {

                    **(
                        {} if not self.training_results else {
                            'training_da_inconsistency': self.training_results.get('da_inconsistency', {}).get(p, {}).get('data', None),
                            'training_da_positive': self.training_results.get('da_positive', {}).get(p, {}).get('data', None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {
                            'operational_da_inconsistency' : self.production_results.get('da_inconsistency', {}).get(p, {}).get('data',None),
                            'operational_da_positive' : self.production_results.get('da_positive', {}).get(p, {}).get('data',None),
                        }
                    ),
                    **(
                        {} if not self.impacted_results else {
                            'impact_da_inconsistency' : self.impacted_results.get('da_inconsistency', {}).get(p, {}).get('data',None),
                            'impact_da_positive' : self.impacted_results.get('da_positive', {}).get(p, {}).get('data',None),
                        }
                    ),
                    }

                audit_result[p]['distribution'] = {
                    'ref': 80,
                    **(
                        {} if not self.training_results else {
                            'training_dxa_inconsistency' : self.training_results.get('dxa_inconsistency', {}).get(p, {}).get('rate',None),
                            'training_da_informative' : self.training_results.get('da_informative', {}).get(p, {}).get('accuracy',None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {    
                            'operational_dxa_inconsistency' : self.production_results.get('dxa_inconsistency', {}).get(p, {}).get('rate',None),
                            'operational_da_informative' : self.production_results.get('da_informative', {}).get(p, {}).get('accuracy',None),
                        }
                    ),
                    **(
                        {} if not self.impacted_results else {   
                            'impact_dxa_inconsistency' : self.impacted_results.get('dxa_inconsistency', {}).get(p, {}).get('rate',None),
                            'impact_da_informative' : self.impacted_results.get('da_informative', {}).get(p, {}).get('accuracy',None),
                        }
                    ),
                    **(
                        {} if not self.drift_results else {   
                            'drift' : self.drift_results.get('tdx_inconsistency', {}).get(p, {}).get('accuracy',None),
                        }
                    ),
                    }
                
                audit_result[p]['fairness'] = {
                    'ref': 80,
                    **(
                        {} if not self.training_results else {
                            'training_DI' : self.training_results.get('d_parity', {}).get(p, {}).get('DI',None),
                            'training_SPD' : self.training_results.get('d_statisticalparity', {}).get(p, {}).get('SPD',None),
                            'training_TPR' : self.training_results.get('d_equalodds', {}).get(p, {}).get('true_positive_rate', {}).get('ratio_true',None),
                            'training_FPR' : self.training_results.get('d_equalodds', {}).get(p, {}).get('false_positive_rate', {}).get('ratio_false',None),
                            'training_PPV' : self.training_results.get('d_calibrated', {}).get(p, {}).get('true_calibrated', {}).get('ratio_true',None),
                            'training_PNV' : self.training_results.get('d_calibrated', {}).get(p, {}).get('false_calibrated', {}).get('ratio_false',None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {
                            'operational_DI' : self.production_results.get('d_parity', {}).get(p, {}).get('DI',None),
                            'operational_SPD' : self.production_results.get('d_statisticalparity', {}).get(p, {}).get('SPD',None),
                        }
                    ),

                    **(
                        {} if not self.impacted_results else {
                            'impact_DI' : self.impacted_results.get('d_parity', {}).get(p, {}).get('DI',None),
                            'impact_SPD' : self.impacted_results.get('d_statisticalparity', {}).get(p, {}).get('SPD',None),
                        }
                    ),
                    
                    
                    
                    }

                audit_result[p]['performance'] = {
                    'ref': 80,
                    **(
                        {} if not self.training_results else {
                            'poor_performance' : self.training_results.get('poor_performance', {}).get(p, {}).get('normalized_risk',None),
                            'recall' : self.training_results.get('poor_performance', {}).get(p, {}).get('recall',None),
                            'f1_score' : self.training_results.get('poor_performance', {}).get(p, {}).get('f1',None),
                            'accuracy' : self.training_results.get('poor_performance', {}).get(p, {}).get('accuracy',None),
                            'precision' : self.training_results.get('poor_performance', {}).get(p, {}).get('precision',None),
                            'TP' : self.training_results.get('poor_performance', {}).get(p, {}).get('TP',None),
                            'FP' : self.training_results.get('poor_performance', {}).get(p, {}).get('FP',None),
                            'TN' : self.training_results.get('poor_performance', {}).get(p, {}).get('TN',None),
                            'FN' : self.training_results.get('poor_performance', {}).get(p, {}).get('FN',None),
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
        protected = [[list(f[k].keys()) for k in f.keys()] for f in [self.training_results,self.production_results,self.impacted_results]]
        protected = [p[0] for p in protected if len(p) != 0]
        protected = list(set().union(*protected))
        audit_result = {}
        for p in protected:
            if p != 'error':

                audit_result[p] = {}
                audit_result[p]['benchmarking'] = {

                    **(
                        {} if not self.training_results else {
                            'training_da_inconsistency': self.training_results.get('da_inconsistency', {}).get(p, {}).get('data', None),
                            'training_da_positive': self.training_results.get('da_positive', {}).get(p, {}).get('data', None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {
                            'operational_da_inconsistency' : self.production_results.get('da_inconsistency', {}).get(p, {}).get('data',None),
                            'operational_da_positive' : self.production_results.get('da_positive', {}).get(p, {}).get('data',None),
                        }
                    ),
                    **(
                        {} if not self.impacted_results else {
                            'impact_da_inconsistency' : self.impacted_results.get('da_inconsistency', {}).get(p, {}).get('data',None),
                            'impact_da_positive' : self.impacted_results.get('da_positive', {}).get(p, {}).get('data',None),
                        }
                    ),
                    }

                audit_result[p]['distribution'] = {
                    'ref': 80,
                    **(
                        {} if not self.training_results else {
                            'training_dxa_inconsistency' : self.training_results.get('dxa_inconsistency', {}).get(p, {}).get('normalized_risk',None),
                            'training_da_informative' : self.training_results.get('da_informative', {}).get(p, {}).get('normalized_risk',None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {    
                            'operational_dxa_inconsistency' : self.production_results.get('dxa_inconsistency', {}).get(p, {}).get('normalized_risk',None),
                            'operational_da_informative' : self.production_results.get('da_informative', {}).get(p, {}).get('normalized_risk',None),
                        }
                    ),
                    **(
                        {} if not self.impacted_results else {   
                            'impact_dxa_inconsistency' : self.impacted_results.get('dxa_inconsistency', {}).get(p, {}).get('normalized_risk',None),
                            'impact_da_informative' : self.impacted_results.get('da_informative', {}).get(p, {}).get('normalized_risk',None),
                        }
                    ),
                    **(
                        {} if not self.drift_results else {   
                            'drift' : self.drift_results.get('tdx_inconsistency', {}).get(p, {}).get('normalized_risk',None),
                        }
                    ),
                    }
                
                audit_result[p]['fairness'] = {
                    'ref': 80,
                    **(
                        {} if not self.training_results else {
                            'training_DI' : self.training_results.get('d_parity', {}).get(p, {}).get('normalized_risk',None),
                            'training_SPD' : self.training_results.get('d_statisticalparity', {}).get(p, {}).get('normalized_risk',None),
                            'training_TPR' : self.training_results.get('d_equalodds', {}).get(p, {}).get('true_positive_rate', {}).get('normalized_risk',None),
                            'training_FPR' : self.training_results.get('d_equalodds', {}).get(p, {}).get('false_positive_rate', {}).get('normalized_risk',None),
                            'training_PPV' : self.training_results.get('d_calibrated', {}).get(p, {}).get('true_calibrated', {}).get('normalized_risk',None),
                            'training_PNV' : self.training_results.get('d_calibrated', {}).get(p, {}).get('false_calibrated', {}).get('normalized_risk',None),
                        }
                    ),
                    **(
                        {} if not self.production_results else {
                            'operational_DI' : self.production_results.get('d_parity', {}).get(p, {}).get('normalized_risk',None),
                            'operational_SPD' : self.production_results.get('d_statisticalparity', {}).get(p, {}).get('normalized_risk',None),
                        }
                    ),

                    **(
                        {} if not self.impacted_results else {
                            'impact_DI' : self.impacted_results.get('d_parity', {}).get(p, {}).get('normalized_risk',None),
                            'impact_SPD' : self.impacted_results.get('d_statisticalparity', {}).get(p, {}).get('normalized_risk',None),
                        }
                    ),
                    
                    
                    
                    }

                audit_result[p]['performance'] = {
                    'ref': 80,
                    **(
                        {} if not self.training_results else {
                            'poor_performance' : self.training_results.get('poor_performance', {}).get(p, {}).get('normalized_risk',None),
                            'recall' : self.training_results.get('poor_performance', {}).get(p, {}).get('recall',None),
                            'f1_score' : self.training_results.get('poor_performance', {}).get(p, {}).get('f1',None),
                            'accuracy' : self.training_results.get('poor_performance', {}).get(p, {}).get('accuracy',None),
                            'precision' : self.training_results.get('poor_performance', {}).get(p, {}).get('precision',None),
                            'TP' : self.training_results.get('poor_performance', {}).get(p, {}).get('TP',None),
                            'FP' : self.training_results.get('poor_performance', {}).get(p, {}).get('FP',None),
                            'TN' : self.training_results.get('poor_performance', {}).get(p, {}).get('TN',None),
                            'FN' : self.training_results.get('poor_performance', {}).get(p, {}).get('FN',None),
                        }
                    ),
                    
                    }
        return audit_result

            
    def df_results(self, norm_values = True):

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
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            df = pd.DataFrame(columns=['group','metric','attribute','stage','value'])
            df = df.set_index(['group','metric','attribute','stage'])
            df.sort_index(inplace=True)  
            protected = [[list(f[k].keys()) for k in f.keys()] for f in [self.training_results,self.production_results,self.impacted_results]]
            protected = [p[0] for p in protected if len(p) != 0]
            protected = list(set().union(*protected))
            audit_result = {}
            for p in protected:
                if p != 'error':
                    if self.training_results:
                        df.loc[('benchmarking','da_inconsistency',p,'training'),'value'] = self.training_results.get('da_inconsistency', {}).get(p, {}).get('data', None)
                        df.loc[('benchmarking','da_positive',p,'training'),'value'] = self.training_results.get('da_positive', {}).get(p, {}).get('data', None)
                        
                        df.loc[('distribution','dxa_inconsistency',p,'training'),'value'] = self.training_results.get('dxa_inconsistency', {}).get(p, {}).get('normalized_risk', None)
                        df.loc[('distribution','da_informative',p,'training'),'value'] = self.training_results.get('da_informative', {}).get(p, {}).get('normalized_risk', None)

                        df.loc[('fairness','d_parity',p,'training'),'value'] = self.training_results.get('d_parity', {}).get(p, {}).get('normalized_risk',None)
                        df.loc[('fairness','d_statisticalparity',p,'training'),'value'] = self.training_results.get('d_statisticalparity', {}).get(p, {}).get('normalized_risk',None)
                        df.loc[('fairness','d_equalodds_true',p,'training'),'value'] = self.training_results.get('d_equalodds', {}).get(p, {}).get('true_positive_rate', {}).get('normalized_risk',None)
                        df.loc[('fairness','d_equalodds_false',p,'training'),'value'] = self.training_results.get('d_equalodds', {}).get(p, {}).get('false_positive_rate', {}).get('normalized_risk',None)
                        df.loc[('fairness','d_calibrated_tru',p,'training'),'value'] = self.training_results.get('d_calibrated', {}).get(p, {}).get('true_calibrated', {}).get('normalized_risk',None)
                        df.loc[('fairness','d_calibrated_false',p,'training'),'value'] = self.training_results.get('d_calibrated', {}).get(p, {}).get('false_calibrated', {}).get('normalized_risk',None)
                    
                        df.loc[('performance','poor_performance',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('normalized_risk',None)
                        df.loc[('performance','recall',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('recall',None)
                        df.loc[('performance','f1',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('f1',None)
                        df.loc[('performance','accuracy',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('accuracy',None)
                        df.loc[('performance','precision',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('precision',None)
                        df.loc[('performance','TP',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('TP',None)
                        df.loc[('performance','FP',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('FP',None)
                        df.loc[('performance','TN',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('TN',None)
                        df.loc[('performance','FN',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('FN',None)

                    if self.production_results:
                        df.loc[('benchmarking','da_inconsistency',p,'production'),'value'] = self.production_results.get('da_inconsistency', {}).get(p, {}).get('data', None)
                        df.loc[('benchmarking','da_positive',p,'production'),'value'] = self.production_results.get('da_positive', {}).get(p, {}).get('data', None)
                        
                        df.loc[('distribution','dxa_inconsistency',p,'production'),'value'] = self.production_results.get('dxa_inconsistency', {}).get(p, {}).get('normalized_risk', None)
                        df.loc[('distribution','da_informative',p,'production'),'value'] = self.production_results.get('da_informative', {}).get(p, {}).get('normalized_risk', None)
                        
                        df.loc[('fairness','d_parity',p,'production'),'value'] = self.production_results.get('d_parity', {}).get(p, {}).get('normalized_risk',None)
                        df.loc[('fairness','d_statisticalparity',p,'production'),'value'] = self.production_results.get('d_statisticalparity', {}).get(p, {}).get('normalized_risk',None)
                    
                    if self.impacted_results:
                        df.loc[('benchmarking','da_inconsistency',p,'impact'),'value'] = self.impacted_results.get('da_inconsistency', {}).get(p, {}).get('data', None)
                        df.loc[('benchmarking','da_positive',p,'impact'),'value'] = self.impacted_results.get('da_positive', {}).get(p, {}).get('data', None)
                        
                        df.loc[('distribution','dxa_inconsistency',p,'impact'),'value'] = self.impacted_results.get('dxa_inconsistency', {}).get(p, {}).get('normalized_risk', None)
                        df.loc[('distribution','da_informative',p,'impact'),'value'] = self.impacted_results.get('da_informative', {}).get(p, {}).get('normalized_risk', None)

                        df.loc[('fairness','d_parity',p,'impact'),'value'] = self.impacted_results.get('d_parity', {}).get(p, {}).get('normalized_risk',None)
                        df.loc[('fairness','d_statisticalparity',p,'impact'),'value'] = self.impacted_results.get('d_statisticalparity', {}).get(p, {}).get('normalized_risk',None)
                    
                    if self.drift_results:
                        df.loc[('distribution','drift',p,'-'),'value'] = self.impacted_results.get('tdx_inconsistency', {}).get(p, {}).get('normalized_risk', None)

                    
        df.sort_index(inplace=True)           
                    
        return df
    
    def df_results_metric(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            df = pd.DataFrame(columns=['group','metric','attribute','stage','value'])
            df = df.set_index(['group','metric','attribute','stage'])
            df.sort_index(inplace=True)  
            protected = [[list(f[k].keys()) for k in f.keys()] for f in [self.training_results,self.production_results,self.impacted_results]]
            protected = [p[0] for p in protected if len(p) != 0]
            protected = list(set().union(*protected))
            audit_result = {}
            for p in protected:
                if p != 'error':
                    if self.training_results:
                        df.loc[('benchmarking','da_inconsistency',p,'training'),'value'] = self.training_results.get('da_inconsistency', {}).get(p, {}).get('data', None)
                        df.loc[('benchmarking','da_positive',p,'training'),'value'] = self.training_results.get('da_positive', {}).get(p, {}).get('data', None)
                        
                        df.loc[('distribution','dxa_inconsistency',p,'training'),'value'] = self.training_results.get('dxa_inconsistency', {}).get(p, {}).get('rate', None)
                        df.loc[('distribution','da_informative',p,'training'),'value'] = self.training_results.get('da_informative', {}).get(p, {}).get('accuracy', None)

                        df.loc[('fairness','d_parity',p,'training'),'value'] = self.training_results.get('d_parity', {}).get(p, {}).get('DI',None)
                        df.loc[('fairness','d_statisticalparity',p,'training'),'value'] = self.training_results.get('d_statisticalparity', {}).get(p, {}).get('SPD',None)
                        df.loc[('fairness','d_equalodds_true',p,'training'),'value'] = self.training_results.get('d_equalodds', {}).get(p, {}).get('true_positive_rate', {}).get('ratio_true',None)
                        df.loc[('fairness','d_equalodds_false',p,'training'),'value'] = self.training_results.get('d_equalodds', {}).get(p, {}).get('false_positive_rate', {}).get('ratio_false',None)
                        df.loc[('fairness','d_calibrated_tru',p,'training'),'value'] = self.training_results.get('d_calibrated', {}).get(p, {}).get('true_calibrated', {}).get('ratio_true',None)
                        df.loc[('fairness','d_calibrated_false',p,'training'),'value'] = self.training_results.get('d_calibrated', {}).get(p, {}).get('false_calibrated', {}).get('ratio_false',None)
                    
                        df.loc[('performance','poor_performance',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('normalized_risk',None)
                        df.loc[('performance','recall',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('recall',None)
                        df.loc[('performance','f1',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('f1',None)
                        df.loc[('performance','accuracy',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('accuracy',None)
                        df.loc[('performance','precision',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('precision',None)
                        df.loc[('performance','TP',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('TP',None)
                        df.loc[('performance','FP',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('FP',None)
                        df.loc[('performance','TN',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('TN',None)
                        df.loc[('performance','FN',p,'training'),'value'] =  self.training_results.get('poor_performance', {}).get(p, {}).get('FN',None)

                    if self.production_results:
                        df.loc[('benchmarking','da_inconsistency',p,'production'),'value'] = self.production_results.get('da_inconsistency', {}).get(p, {}).get('data', None)
                        df.loc[('benchmarking','da_positive',p,'production'),'value'] = self.production_results.get('da_positive', {}).get(p, {}).get('data', None)
                        
                        df.loc[('distribution','dxa_inconsistency',p,'production'),'value'] = self.production_results.get('dxa_inconsistency', {}).get(p, {}).get('rate', None)
                        df.loc[('distribution','da_informative',p,'production'),'value'] = self.production_results.get('da_informative', {}).get(p, {}).get('accuracy', None)
                        
                        df.loc[('fairness','d_parity',p,'production'),'value'] = self.production_results.get('d_parity', {}).get(p, {}).get('DI',None)
                        df.loc[('fairness','d_statisticalparity',p,'production'),'value'] = self.production_results.get('S^D', {}).get(p, {}).get('normalized_risk',None)
                    
                    if self.impacted_results:
                        df.loc[('benchmarking','da_inconsistency',p,'impact'),'value'] = self.impacted_results.get('da_inconsistency', {}).get(p, {}).get('data', None)
                        df.loc[('benchmarking','da_positive',p,'impact'),'value'] = self.impacted_results.get('da_positive', {}).get(p, {}).get('data', None)
                        
                        df.loc[('distribution','dxa_inconsistency',p,'impact'),'value'] = self.impacted_results.get('dxa_inconsistency', {}).get(p, {}).get('rate', None)
                        df.loc[('distribution','da_informative',p,'impact'),'value'] = self.impacted_results.get('da_informative', {}).get(p, {}).get('accuracy', None)

                        df.loc[('fairness','d_parity',p,'impact'),'value'] = self.impacted_results.get('d_parity', {}).get(p, {}).get('DI',None)
                        df.loc[('fairness','d_statisticalparity',p,'impact'),'value'] = self.impacted_results.get('d_statisticalparity', {}).get(p, {}).get('SPD',None)
                    
                    if self.drift_results:
                        df.loc[('distribution','drift',p,'-'),'value'] = self.impacted_results.get('tdx_inconsistency', {}).get(p, {}).get('accuracy', None)

                    
        df.sort_index(inplace=True)          
                    
        return df


        