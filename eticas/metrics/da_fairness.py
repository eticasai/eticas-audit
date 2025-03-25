import numpy as np
from .base_metric import BaseFairnessMetric
import math
import logging
import pandas as pd
from ..utils.helpers import get_mask, binarize_predictions
logger = logging.getLogger(__name__)


class Da_fairness(BaseFairnessMetric):
    """
    Metric D(A)-fairness
    Calculate two metrics : Equality and Equity.
        Equality: Means that all groups are treated in same manner.
        Equity: Means that some group needs additional resources to get success.
    """

    def compute(self,
                input_data,
                sensitive_attrs=None,
                input_features=None,
                label_column=None,
                positive_output: list = [1]):
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
            input_features=input_features,
            label_column=label_column,
            positive_output=positive_output,
        )

        input_data = input_data.dropna()

        if not input_data[label_column].isin([0, 1]).all():
            # Convert to binary
            input_data[label_column] = binarize_predictions(input_data[label_column].values)

        train_columns = [c for c in input_data.columns if c in input_features]

        if len(train_columns) == 0:
            logger.error("Input features are not in dataset.")
            return {'error': "Input features are not in dataset."}

        json_groups = sensitive_attrs

        n_ranges = 5
        for f in input_features:
            if input_data[f].unique().shape[0] > 10:
                input_data[f] = pd.qcut(input_data[f], q=n_ranges, labels=range(5))

        result_list = {}
        for item in json_groups.items():
            data = input_data.copy()
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
                    mask_privileged, mask_underprivileged = get_mask(input_data, filters)
                    data['complex'] = mask_underprivileged.astype(int)
                    s_columns.append('complex')

                column_filter = s_columns

                df_aux = data.groupby(
                    by=column_filter + input_features, observed=False).agg({label_column: ['count', 'sum']})
                df_aux_ideal = data.groupby(
                    by=input_features, observed=False).agg({label_column: ['count', 'sum']})
                df_aux.columns = df_aux.columns.droplevel(0)
                df_aux = df_aux.reset_index()
                combinations_s = df_aux[column_filter].value_counts().index.values
                df_aux = df_aux.set_index(column_filter + input_features)

                df_aux_ideal.columns = df_aux_ideal.columns.droplevel(0)
                df_aux_ideal = df_aux_ideal.reset_index()
                # combinations_f = df_aux_ideal[input_features].value_counts().index.values
                df_aux_ideal['px'] = df_aux_ideal['sum'] / df_aux_ideal['count']
                df_aux_ideal = df_aux_ideal.sort_values(by=['px']+input_features)
                df_aux_ideal = df_aux_ideal.set_index(input_features)
                df_aux_ideal['dx'] = [0] + (df_aux_ideal['count'].cumsum() / df_aux_ideal['count'].sum()).tolist()[:-1]

                df_aux['px'] = df_aux['sum'] / df_aux['count']

                n_group = combinations_s.shape[0]
                # groups = [str(column_filter) + str(s) for s in combinations_s]
                # combinations = [[s + f for s in combinations_s] for f in combinations_f]
                df_aux = df_aux.reset_index().set_index(input_features)

                for cs, n in zip(combinations_s, range(n_group)):
                    condition = True
                    for feature, value in zip(column_filter, cs):
                        condition &= (df_aux[feature] == value)
                    filtered_aux = df_aux[condition]
                    df_aux_ideal = df_aux_ideal.merge(
                        filtered_aux, how='outer', left_index=True,
                        right_index=True, suffixes=["", "_"+str(n)]).fillna(0)
                    df_aux_ideal = df_aux_ideal.reset_index().sort_values(by=['px']+input_features)
                    df_aux_ideal = df_aux_ideal.set_index(input_features)
                    df_aux_ideal['dx_' + str(n)] = (
                        [0] +
                        (
                            df_aux_ideal['count_' + str(n)].cumsum() /
                            df_aux_ideal['count_' + str(n)].sum()
                        ).tolist()[:-1]
                    )
                n_p = -1
                p_max = 0
                d_max = 0
                for n in range(n_group):
                    p_aux = df_aux_ideal['px_'+str(n)].max()
                    d_aux = df_aux_ideal['dx_'+str(n)].max()
                    if p_aux > p_max:
                        p_max = p_aux
                        d_max = d_aux
                        n_p = n
                    elif p_aux == p_max:
                        if d_aux < d_max:
                            p_max = p_aux
                            d_max = d_aux
                            n_p = n

                # df_f = pd.DataFrame(columns=['group', 'reference', 'EQI', 'EQA', 'F'])
                for n in range(n_group):
                    if n != n_p:
                        eqi = (df_aux_ideal['dx_'+str(n_p)] - df_aux_ideal['dx_'+str(n)]).values
                        eqa = (df_aux_ideal['px_'+str(n_p)] - df_aux_ideal['px_'+str(n)]).values

                        # EQI = np.round(eqi.mean(),2)
                        EQI = abs((eqi * (df_aux_ideal['count'] / df_aux_ideal['count'].sum()).values).sum())
                        EQI = np.round(EQI, 2)
                        # EQA = np.round(eqa.mean(),2)
                        EQA = abs((eqa * (df_aux_ideal['count'] / df_aux_ideal['count'].sum()).values).sum())
                        EQA = np.round(EQA, 2)
                        F = np.round(math.sqrt(EQA**2 + EQI**2), 2)
                        EQA_norm = np.round(100*(1-EQA), 2)
                        EQI_norm = np.round(100*(1-EQI), 2)
                        normalized_risk = np.round(100*(1-F), 2)
                        # df_f.loc[n] = [groups[n], groups[n_p], EQI, EQA, F]
                result_list.update({group: {
                                            'equality': EQA.item(),
                                            'equity': EQI.item(),
                                            'fairness': F.item(),
                                            'equality_norm': EQA_norm.item(),
                                            'equity_norm': EQI_norm.item(),
                                            'normalized_risk': normalized_risk.item(),
                                            'bias_level': self.get_bias_level(normalized_risk)}})
            except KeyError:
                # Captura cualquier ValueError lanzado por get_benchmarking
                logger.error(f"Group no present in data: '{group}'")
                result_list.update({group: {
                                            'equality': 0,
                                            'equity': 0,
                                            'fairness': 0,
                                            'equality_norm': 0,
                                            'equity_norm': 0,
                                            'normalized_risk': None,
                                            'bias_level': self.get_bias_level(0),
                                            'error': 'group no present in data.'}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list
