# run test python -m unittest discover tests

import logging
import unittest
from eticas.model.ml_model import MLModel
from eticas.audit.labeled_audit import LabeledAudit
from eticas.audit.unlabeled_audit import UnlabeledAudit
from eticas.audit.drift_audit import DriftAudit
import pandas as pd
import pandas.testing as pdt
import json
logger = logging.getLogger(__name__)

sensitive_attributes = {'sex': {'columns': [
                                            {
                                                "name": "sex",
                                                "underprivileged": [2]
                                            }
                                        ],
                                'type': 'simple'},
                        'ethnicity': {'columns': [
                                                 {
                                                     "name": "ethnicity",
                                                     "privileged": [1]
                                                 }
                                        ],
                                      'type': 'simple'},
                        'age': {'columns': [
                                            {
                                                 "name": "age",
                                                 "privileged": [3, 4]
                                            }
                                        ],
                                'type': 'simple'},
                        'error': {'columns': [
                            {
                                "name": "error",
                                  "privileged": [3, 4]
                                  }
                                  ],
                                  'type': 'simple'},
                        'sex_ethnicity': {'groups': [
                                                    "sex", "ethnicity"
                                                    ],
                                          'type': 'complex'}}
logging.basicConfig(
    level=logging.ERROR,
    format='[%(levelname)s] %(name)s - %(message)s'
)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.model = MLModel(
            model_name="ML Testing Regression",
            description="A logistic regression model to illustrate audits",
            country="USA",
            state="CA",
            sensitive_attributes=sensitive_attributes,
            features=["feature_0", "feature_1", "feature_2"]
        )
        self.model.run_labeled_audit(dataset_path='files/example_training_binary_2.csv',
                                     label_column='outcome', output_column='predicted_outcome', positive_output=[1])
        self.model.run_production_audit(dataset_path='files/example_operational_binary_2.csv',
                                        output_column='predicted_outcome', positive_output=[1])
        self.model.run_impacted_audit(dataset_path='files/example_impact_binary_2.csv',
                                      output_column='recorded_outcome', positive_output=[1])
        self.model.run_drift_audit(dataset_path_dev='files/example_training_binary_2.csv',
                                   output_column_dev='outcome', positive_output_dev=[1],
                                   dataset_path_prod='files/example_operational_binary_2.csv',
                                   output_column_prod='predicted_outcome', positive_output_prod=[1])

    def test_df_norm(self):
        expected = pd.read_pickle('tests/ml_files/result_df_norm.pickle')
        result = self.model.df_results(norm_values=True).reset_index()
        pdt.assert_frame_equal(result, expected)

    def test_df(self):
        expected = pd.read_pickle('tests/ml_files/result_df.pickle')
        result = self.model.df_results(norm_values=False).reset_index()
        pdt.assert_frame_equal(result, expected)

    def test_json_norm(self):
        with open('tests/ml_files/result_json_norm.json', 'r', encoding='utf-8') as f:
            expected = json.load(f)
        result = self.model.json_results(norm_values=True)
        self.assertEqual(result, expected)

    def test_json(self):
        with open('tests/ml_files/result_json.json', 'r', encoding='utf-8') as f:
            expected = json.load(f)
        result = self.model.json_results(norm_values=False)
        self.assertEqual(result, expected)

    def test_json_labeled(self):
        with open('tests/ml_files/result_json_labeled.json', 'r', encoding='utf-8') as f:
            expected = json.load(f)
        result = self.model.labeled_results
        self.assertEqual(result, expected)

    def test_json_production(self):
        with open('tests/ml_files/result_json_production.json', 'r', encoding='utf-8') as f:
            expected = json.load(f)
        result = self.model.production_results
        self.assertEqual(result, expected)

    def test_json_impacted(self):
        with open('tests/ml_files/result_json_impacted.json', 'r', encoding='utf-8') as f:
            expected = json.load(f)
        result = self.model.impacted_results
        self.assertEqual(result, expected)

    def test_test_drift_dev_shape(self):
        with self.assertRaises(ValueError) as context:
            DriftAudit(self.model).run_audit('tests/ml_files/zero_shape.csv', None, None, None, None, None)
        self.assertIn("DEV data shape is 0.", str(context.exception))

    def test_test_drift_prod_shape(self):
        with self.assertRaises(ValueError) as context:
            DriftAudit(self.model).run_audit('files/example_training_binary_2.csv', None, None,
                                             'tests/ml_files/zero_shape.csv', None, None)
        self.assertIn("PROD data shape is 0.", str(context.exception))

    def test_test_labeled_shape(self):
        with self.assertRaises(ValueError) as context:
            LabeledAudit(self.model).run_audit('tests/ml_files/zero_shape.csv', None, None)
        self.assertIn("labeled dataset shape is 0.", str(context.exception))

    def test_test_unlabeled_shape(self):
        with self.assertRaises(ValueError) as context:
            UnlabeledAudit(self.model).run_audit('tests/ml_files/zero_shape.csv', None, None)
        self.assertIn("production dataset shape is 0.", str(context.exception))

    def test_str_method(self):
        # Test that the __str__ method returns the class name.
        self.assertEqual(str(self.model), "MLModel(ML Testing Regression)",
                         "The __str__ method should return the class name.")


if __name__ == '__main__':
    unittest.main()
