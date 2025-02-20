from .base_metric import BaseFairnessMetric
from ..utils.helpers import binarize_predictions
import logging
logger = logging.getLogger(__name__)


class DisparateImpact(BaseFairnessMetric):
    """
    Calculates the impact ratio.
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

        input_data = input_data.dropna()

        if not input_data[label_column].isin([0, 1]).all():
            # Convert to binary
            input_data[label_column] = binarize_predictions(input_data[label_column].values)
        json_groups = sensitive_attrs
        result_list = {}

        for item in json_groups.items():
            group = item[0]

            column = []
            if item[1]['type'] == 'simple':
                column.append(item[1]['columns'][0]['name'])
            if item[1]['type'] == 'complex':
                for g in item[1]['groups']:
                    column.append(json_groups[g]['columns'][0]['name'])
            if all([c in input_data.columns for c in column]):
                df_aux = (
                    input_data.groupby(column)
                    .agg(
                        **{
                            '# of Applicants': (label_column, 'count'),  # Conteo
                            'Selection Rate': (label_column, 'mean')     # Promedio
                        }
                    )
                    .reset_index()
                )
                df_aux['Impact Ratio'] = df_aux['Selection Rate'] / df_aux['Selection Rate'].max()
                df_aux['Selection Rate'] = (df_aux['Selection Rate'] * 100).map(lambda x: f"{x:.1f}%")

                n_pass = df_aux[df_aux['Impact Ratio'] >= 0.8].shape[0]
                df_aux['Impact Ratio'] = (df_aux['Impact Ratio']).map(lambda x: f"{x:.3f}")

                # df_merge = pd.DataFrame(input_data[column].drop_duplicates().values.tolist(), columns=[column])
                # df_aux = df_aux.merge(df_merge, how='right', left_on=column, right_on=column).fillna(0)

                mask = df_aux['# of Applicants'] <= 0
                df_aux.loc[mask, 'Selection Rate'] = 0
                df_aux.loc[mask, 'Impact Ratio'] = 0

                aux_unknown = input_data.shape[0] - df_aux['# of Applicants'].sum()

                result_list.update({group: {'df_result': df_aux.to_json(orient='split'),
                                            'unknown': aux_unknown.item(),
                                            'pass': n_pass,
                                            'total_filter': df_aux.shape[0] - mask.sum().item(),
                                            'total': df_aux.shape[0]}})
            else:
                result_list.update({group: {'df_result': None,
                                            'unknown': None,
                                            'pass': None,
                                            'total_filter': None,
                                            'total': None}})
        logger.info(f"Completed: '{self.__str__()}'")
        return result_list
