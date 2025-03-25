# run test python -m unittest discover tests

import logging
from eticas.metrics.d_calibrated import D_calibrated
from eticas.metrics.d_equalodds import D_equalodds
from eticas.metrics.d_parity import D_parity
from eticas.metrics.d_statisticalparity import D_statisticalparity
from eticas.metrics.da_fairness import Da_fairness
from eticas.metrics.da_inconsistency import Da_inconsistency
from eticas.metrics.da_informative import Da_informative
from eticas.metrics.da_positive import Da_positive
from eticas.metrics.dxa_inconsistency import Dxa_inconsistency
from eticas.metrics.performance import Performance
from eticas.metrics.tdx_inconsistency import Tdx_inconsistency
from eticas.data.loaders import load_dataset
from eticas.metrics.disparate_impact import DisparateImpact
import unittest

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
    'sex_ethnicity': {'groups': ["sex", "ethnicity"],
                      'type': 'complex'}}

sensitive_attributes_error = {'sex': {'columns': [
                                            {
                                                "name": "sex",
                                                "underprivileged": [2]
                                            }
                                        ], 'type': 'simple'},
                              'ethnicity': {'columns': [
                                                 {
                                                     "name": "ethnicity",
                                                     "privileged": [1]
                                                 }
                                        ], 'type': 'simple'},
                              'age': {'columns': [
                                            {
                                                 "name": "age",
                                                 "privileged": [3, 4]
                                            }
                                        ], 'type': 'simple'},
                              'error': {'columns': [
                                  {
                                      "name": "error",
                                      "privileged": [3, 4]
                                  }
                                  ],
                                  'type': 'simple'},
                              'sex_ethnicity': {'groups': [
                                                    "sex", "ethnicity"
                                                    ], 'type': 'complex'}}

features = ["feature_0", "feature_1", "feature_2"]

label_column = 'outcome'
output_column = 'predicted_outcome'
positive_output = [1]

logger = logging.getLogger(__name__)


class TestMetrics(unittest.TestCase):

    def test_da_inconsistency(self):

        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Da_inconsistency().compute(input_data, sensitive_attributes)

        expected_result = {
            'age': {
                'data': 44.8},
            'ethnicity': {
                'data': 40.0},
            'sex': {
                'data': 60.0},
            'sex_ethnicity': {
                'data': 25.0}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["data"],
            expected_result["age"]["data"],
            places=7,
            msg="da_inconsistency for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["data"],
            expected_result["sex"]["data"],
            places=7,
            msg="da_inconsistency for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["data"],
            expected_result["ethnicity"]["data"],
            places=7,
            msg="da_inconsistency for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["data"],
            expected_result["sex_ethnicity"]["data"],
            places=7,
            msg="da_positive for sex_ethnicity does not match expected value"
        )

    def test_da_positive(self):

        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Da_positive().compute(input_data, sensitive_attributes,
                                       label_column, positive_output)
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Da_positive().compute(input_data, sensitive_attributes,
                                       label_column, positive_output)

        expected_result = {
            'age': {
                'data': 45.2},
            'ethnicity': {
                'data': 39.8},
            'sex': {
                'data': 59.8},
            'sex_ethnicity': {
                'data': 24.7}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["data"],
            expected_result["age"]["data"],
            places=7,
            msg="da_positive for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["data"],
            expected_result["sex"]["data"],
            places=7,
            msg="da_positive for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["data"],
            expected_result["ethnicity"]["data"],
            places=7,
            msg="da_positive for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["data"],
            expected_result["sex_ethnicity"]["data"],
            places=7,
            msg="da_positive for sex_ethnicity does not match expected value"
        )

    def test_dxa_inconsistency(self):
        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Dxa_inconsistency().compute(input_data, sensitive_attributes, features)

        expected_result = {
            'age': {
                'normalized_risk': 99.28},
            'ethnicity': {
                'normalized_risk': 99.64},
            'sex': {
                'normalized_risk': 99.128},
            'sex_ethnicity': {
                'normalized_risk': 98.808}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["normalized_risk"],
            expected_result["age"]["normalized_risk"],
            places=7,
            msg="dxa_inconsistency for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["normalized_risk"],
            expected_result["sex"]["normalized_risk"],
            places=7,
            msg="dxa_inconsistency for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["normalized_risk"],
            expected_result["ethnicity"]["normalized_risk"],
            places=7,
            msg="dxa_inconsistency for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["normalized_risk"],
            expected_result["sex_ethnicity"]["normalized_risk"],
            places=7,
            msg="dxa_inconsistency for sex_ethnicity does not match expected value"
        )

    def test_da_informative(self):
        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Da_informative().compute(
            input_data, sensitive_attributes, label_column, features)

        expected_result = {
            'age': {
                'normalized_risk': 100.0},
            'ethnicity': {
                'normalized_risk': 100},
            'sex': {
                'normalized_risk': 99.3294},
            'sex_ethnicity': {
                'normalized_risk': 100.0}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["normalized_risk"],
            expected_result["age"]["normalized_risk"],
            places=7,
            msg="da_informative for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["normalized_risk"],
            expected_result["sex"]["normalized_risk"],
            places=7,
            msg="da_informative for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["normalized_risk"],
            expected_result["ethnicity"]["normalized_risk"],
            places=7,
            msg="da_informative for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["normalized_risk"],
            expected_result["sex_ethnicity"]["normalized_risk"],
            places=7,
            msg="da_informative for sex_ethnicity does not match expected value"
        )

        self.assertAlmostEqual(Da_informative().normalize_value(1.0), 100)
        self.assertAlmostEqual(Da_informative().normalize_value(1.09), 81.99999999999999)
        self.assertAlmostEqual(Da_informative().normalize_value(1.19), 62.000000000000014)
        self.assertAlmostEqual(Da_informative().normalize_value(1.21), 59.25)

    def test_d_statisticalparity(self):

        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = D_statisticalparity().compute(input_data, sensitive_attributes,
                                               label_column, positive_output)
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = D_statisticalparity().compute(input_data, sensitive_attributes,
                                               label_column, positive_output)

        expected_result = {
            'age': {
                'normalized_risk': 98.0},
            'ethnicity': {
                'normalized_risk': 100},
            'sex': {
                'normalized_risk': 100},
            'sex_ethnicity': {
                'normalized_risk': 98.0}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["normalized_risk"],
            expected_result["age"]["normalized_risk"],
            places=7,
            msg="d_statisticalparity for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["normalized_risk"],
            expected_result["sex"]["normalized_risk"],
            places=7,
            msg="d_statisticalparity for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["normalized_risk"],
            expected_result["ethnicity"]["normalized_risk"],
            places=7,
            msg="d_statisticalparity for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["normalized_risk"],
            expected_result["sex_ethnicity"]["normalized_risk"],
            places=7,
            msg="d_statisticalparity for sex_ethnicity does not match expected value"
        )

    def test_d_parity(self):
        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = D_parity().compute(input_data, sensitive_attributes,
                                    label_column, positive_output)
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = D_parity().compute(input_data, sensitive_attributes,
                                    label_column, positive_output)

        expected_result = {
            'age': {
                'normalized_risk': 98.0},
            'ethnicity': {
                'normalized_risk': 99.0},
            'sex': {
                'normalized_risk': 99.0},
            'sex_ethnicity': {
                'normalized_risk': 99.0}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["normalized_risk"],
            expected_result["age"]["normalized_risk"],
            places=7,
            msg="d_parity for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["normalized_risk"],
            expected_result["sex"]["normalized_risk"],
            places=7,
            msg="d_parity for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["normalized_risk"],
            expected_result["ethnicity"]["normalized_risk"],
            places=7,
            msg="d_parity for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["normalized_risk"],
            expected_result["sex_ethnicity"]["normalized_risk"],
            places=7,
            msg="d_parity for sex_ethnicity does not match expected value"
        )

        self.assertEqual(D_parity().normalize_value(0.8), 80)
        self.assertEqual(D_parity().normalize_value(1), 100)
        self.assertEqual(D_parity().normalize_value(1.2), 80)
        self.assertAlmostEqual(D_parity().normalize_value(0.9), 90)
        self.assertAlmostEqual(D_parity().normalize_value(1.1), 90)
        self.assertAlmostEqual(D_parity().normalize_value(1.5), 72.5)
        self.assertAlmostEqual(D_parity().normalize_value(1.8), 65)
        self.assertEqual(D_parity().normalize_value(2), 0)
        self.assertEqual(D_parity().normalize_value(3), 0)
        self.assertAlmostEqual(D_parity().normalize_value(0.65), 70)
        self.assertAlmostEqual(D_parity().normalize_value(0.70), 73.33333333333333)
        self.assertAlmostEqual(D_parity().normalize_value(0.5), 60)
        self.assertAlmostEqual(D_parity().normalize_value(0.3), 36)
        self.assertEqual(D_parity().normalize_value(0), 0)
        self.assertEqual(D_parity().normalize_value(-1), -120)

    def test_performance(self):
        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Performance().compute(input_data, sensitive_attributes,
                                       label_column, positive_output, output_column)
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Performance().compute(input_data, sensitive_attributes,
                                       label_column, positive_output, output_column)

        expected_result = {
            'age': {
                'normalized_risk': 57.588},
            'ethnicity': {
                'normalized_risk': 59.58},
            'sex': {
                'normalized_risk': 59.052},
            'sex_ethnicity': {
                'normalized_risk': 58.572}
        }

        # --- Check keys --
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["normalized_risk"],
            expected_result["age"]["normalized_risk"],
            places=7,
            msg="performance for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["normalized_risk"],
            expected_result["sex"]["normalized_risk"],
            places=7,
            msg="performance for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["normalized_risk"],
            expected_result["ethnicity"]["normalized_risk"],
            places=7,
            msg="performance for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["normalized_risk"],
            expected_result["sex_ethnicity"]["normalized_risk"],
            places=7,
            msg="performance for sex_ethnicity does not match expected value"
        )

        self.assertEqual(Performance().normalize_value(0.9), 54.0)
        self.assertEqual(Performance().normalize_value(1.2), 73.33333333333333)
        self.assertEqual(Performance().normalize_value(1.5), 85.71428571428571)
        self.assertAlmostEqual(Performance().normalize_value(4), 100)

    def test_da_fairness(self):
        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Da_fairness().compute(input_data, sensitive_attributes,
                                       features, label_column, positive_output)

        expected_result = {
            'age': {
                'equity': 0.0,
                'equality': 0.01},
            'ethnicity': {
                'equity': 0.0,
                'equality': 0.0},
            'sex': {
                'equity': 0.01,
                'equality': 0.0},
            'sex_ethnicity': {
                'equity': 0.0,
                'equality': 0.01}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["equity"],
            expected_result["age"]["equity"],
            places=7,
            msg="da_fairness equity for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["equity"],
            expected_result["sex"]["equity"],
            places=7,
            msg="da_fairness equity for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["equity"],
            expected_result["ethnicity"]["equity"],
            places=7,
            msg="da_fairness equity for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["equity"],
            expected_result["sex_ethnicity"]["equity"],
            places=7,
            msg="da_fairness equity for sex_ethnicity does not match expected value"
        )

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["equality"],
            expected_result["age"]["equality"],
            places=7,
            msg="da_fairness equality for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["equality"],
            expected_result["sex"]["equality"],
            places=7,
            msg="da_fairness equality for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["equality"],
            expected_result["ethnicity"]["equality"],
            places=7,
            msg="da_fairness equality for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["equality"],
            expected_result["sex_ethnicity"]["equality"],
            places=7,
            msg="da_fairness equality for sex_ethnicity does not match expected value"
        )

    def test_calibrated(self):

        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")
        result = D_calibrated().compute(input_data, sensitive_attributes,
                                        label_column, positive_output, output_column)
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")
        result = D_calibrated().compute(input_data, sensitive_attributes,
                                        label_column, positive_output, output_column)
        expected_result = {
            'age': {
                'true_calibrated': 96.82,
                'false_calibrated': 99.76},
            'ethnicity': {
                'true_calibrated': 98.6,
                'false_calibrated': 97.86},
            'sex': {
                'true_calibrated': 96.36,
                'false_calibrated': 98.69},
            'sex_ethnicity': {
                'true_calibrated': 95.01,
                'false_calibrated': 99.18}
        }

        # --- Check keys --
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["true_calibrated"]["normalized_risk"],
            expected_result["age"]["true_calibrated"],
            places=7,
            msg="d_calibrated true for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["true_calibrated"]["normalized_risk"],
            expected_result["sex"]['true_calibrated'],
            places=7,
            msg="d_calibrated true for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["true_calibrated"]["normalized_risk"],
            expected_result["ethnicity"]['true_calibrated'],
            places=7,
            msg="d_calibrated true for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["true_calibrated"]["normalized_risk"],
            expected_result["sex_ethnicity"]['true_calibrated'],
            places=7,
            msg="d_calibrated true for sex_ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["age"]["false_calibrated"]["normalized_risk"],
            expected_result["age"]['false_calibrated'],
            places=7,
            msg="d_calibrated false for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["false_calibrated"]["normalized_risk"],
            expected_result["sex"]['false_calibrated'],
            places=7,
            msg="d_calibrated false for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["false_calibrated"]["normalized_risk"],
            expected_result["ethnicity"]['false_calibrated'],
            places=7,
            msg="d_calibrated false for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["false_calibrated"]["normalized_risk"],
            expected_result["sex_ethnicity"]['false_calibrated'],
            places=7,
            msg="d_calibrated false for sex_ethnicity does not match expected value"
        )

        self.assertEqual(D_calibrated().normalize_value(0.8), 80)
        self.assertEqual(D_calibrated().normalize_value(1), 100)
        self.assertEqual(D_calibrated().normalize_value(1.2), 80)
        self.assertAlmostEqual(D_calibrated().normalize_value(0.9), 90)
        self.assertAlmostEqual(D_calibrated().normalize_value(1.1), 90)
        self.assertAlmostEqual(D_calibrated().normalize_value(1.5), 72.5)
        self.assertAlmostEqual(D_calibrated().normalize_value(1.8), 65)
        self.assertEqual(D_calibrated().normalize_value(2), 0)
        self.assertEqual(D_calibrated().normalize_value(3), 0)
        self.assertAlmostEqual(D_calibrated().normalize_value(0.65), 70)
        self.assertAlmostEqual(D_calibrated().normalize_value(0.70), 73.33333333333333)
        self.assertAlmostEqual(D_calibrated().normalize_value(0.5), 60)
        self.assertAlmostEqual(D_calibrated().normalize_value(0.3), 36)
        self.assertEqual(D_calibrated().normalize_value(0), 0)
        self.assertEqual(D_calibrated().normalize_value(-1), -120)

    def test_equalodd(self):

        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = D_equalodds().compute(input_data, sensitive_attributes,
                                       label_column, positive_output, output_column)
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = D_equalodds().compute(input_data, sensitive_attributes,
                                       label_column, positive_output, output_column)

        expected_result = {
            'age': {
                'true_positive_rate': 96.82,
                'false_positive_rate': 99.76},
            'ethnicity': {
                'true_positive_rate': 98.6,
                'false_positive_rate': 97.86},
            'sex': {
                'true_positive_rate': 96.36,
                'false_positive_rate': 98.69},
            'sex_ethnicity': {
                'true_positive_rate': 95.01,
                'false_positive_rate': 99.18}
        }

        # --- Check keys --
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["true_positive_rate"]["normalized_risk"],
            expected_result["age"]["true_positive_rate"],
            places=7,
            msg="d_equalodd true for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["true_positive_rate"]["normalized_risk"],
            expected_result["sex"]['true_positive_rate'],
            places=7,
            msg="d_equalodd true for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["true_positive_rate"]["normalized_risk"],
            expected_result["ethnicity"]['true_positive_rate'],
            places=7,
            msg="d_equalodd true for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["true_positive_rate"]["normalized_risk"],
            expected_result["sex_ethnicity"]['true_positive_rate'],
            places=7,
            msg="d_equalodd true for sex_ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["age"]["false_positive_rate"]["normalized_risk"],
            expected_result["age"]['false_positive_rate'],
            places=7,
            msg="d_equalodd false for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["false_positive_rate"]["normalized_risk"],
            expected_result["sex"]['false_positive_rate'],
            places=7,
            msg="d_equalodd false for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["false_positive_rate"]["normalized_risk"],
            expected_result["ethnicity"]['false_positive_rate'],
            places=7,
            msg="d_equalodd false for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["false_positive_rate"]["normalized_risk"],
            expected_result["sex_ethnicity"]['false_positive_rate'],
            places=7,
            msg="d_equalodd false for sex_ethnicity does not match expected value"
        )

        self.assertEqual(D_equalodds().normalize_value(0.8), 80)
        self.assertEqual(D_equalodds().normalize_value(1), 100)
        self.assertEqual(D_equalodds().normalize_value(1.2), 80)
        self.assertAlmostEqual(D_equalodds().normalize_value(0.9), 90)
        self.assertAlmostEqual(D_equalodds().normalize_value(1.1), 90)
        self.assertAlmostEqual(D_equalodds().normalize_value(1.5), 72.5)
        self.assertAlmostEqual(D_equalodds().normalize_value(1.8), 65)
        self.assertEqual(D_equalodds().normalize_value(2), 0)
        self.assertEqual(D_equalodds().normalize_value(3), 0)
        self.assertAlmostEqual(D_equalodds().normalize_value(0.65), 70)
        self.assertAlmostEqual(D_equalodds().normalize_value(0.70), 73.33333333333333)
        self.assertAlmostEqual(D_equalodds().normalize_value(0.5), 60)
        self.assertAlmostEqual(D_equalodds().normalize_value(0.3), 36)
        self.assertEqual(D_equalodds().normalize_value(0), 0)
        self.assertEqual(D_equalodds().normalize_value(-1), -120)

    def test_tdx_inconsistency(self):

        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Tdx_inconsistency().compute(sensitive_attributes, features,
                                             input_data, output_column, output_column,
                                             input_data, output_column, output_column)
        expected_result = {
            'age': {
                'normalized_risk': 100.0},
            'ethnicity': {
                'normalized_risk': 100.0},
            'sex': {
                'normalized_risk': 100.0},
            'sex_ethnicity': {
                'normalized_risk': 100.0}
        }

        # --- Check keys ---
        self.assertIn("age", result)
        self.assertIn("ethnicity", result)
        self.assertIn("sex", result)
        self.assertIn("sex_ethnicity", result)

        # --- Check numeric values (using assertAlmostEqual for floats) ---
        self.assertAlmostEqual(
            result["age"]["normalized_risk"],
            expected_result["age"]["normalized_risk"],
            places=7,
            msg="Tdx_inconsistency for age does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex"]["normalized_risk"],
            expected_result["sex"]["normalized_risk"],
            places=7,
            msg="Tdx_inconsistency for sex does not match expected value"
        )
        self.assertAlmostEqual(
            result["ethnicity"]["normalized_risk"],
            expected_result["ethnicity"]["normalized_risk"],
            places=7,
            msg="Tdx_inconsistency for ethnicity does not match expected value"
        )
        self.assertAlmostEqual(
            result["sex_ethnicity"]["normalized_risk"],
            expected_result["sex_ethnicity"]["normalized_risk"],
            places=7,
            msg="Tdx_inconsistency for sex_ethnicity does not match expected value"
        )

        self.assertAlmostEqual(Tdx_inconsistency().normalize_value(1.0), 100)
        self.assertAlmostEqual(Tdx_inconsistency().normalize_value(1.09), 81.99999999999999)
        self.assertAlmostEqual(Tdx_inconsistency().normalize_value(1.19), 62.000000000000014)
        self.assertAlmostEqual(Tdx_inconsistency().normalize_value(1.21), 59.25)

    def test_disparate_impact(self):

        input_data = load_dataset('files/example_training_scoring.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        DisparateImpact().compute(input_data, sensitive_attributes,
                                  label_column, positive_output)
        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        DisparateImpact().compute(input_data, sensitive_attributes,
                                  label_column, positive_output)


if __name__ == '__main__':
    unittest.main()
