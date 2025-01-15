# run test python -m unittest discover tests

import logging
from eticas.metrics.performance import Performance
from eticas.metrics.d_parity import D_parity
from eticas.metrics.d_statisticalparity import D_statisticalparity
from eticas.metrics.da_informative import Da_informative
from eticas.metrics.dxa_inconsistency import Dxa_inconsistency
from eticas.metrics.da_positive import Da_positive
from eticas.metrics.da_inconsistency import Da_inconsistency
from eticas.data.loaders import load_dataset
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
                'normalized_risk': 99.656}
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

        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Da_informative().compute(
            input_data, sensitive_attributes, label_column, features)

        expected_result = {
            'age': {
                'normalized_risk': 98.0},
            'ethnicity': {
                'normalized_risk': 99.72},
            'sex': {
                'normalized_risk': 99.08},
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

    def test_d_statisticalparity(self):

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

    def test_performance(self):

        input_data = load_dataset('files/example_training_binary_2.csv')
        input_data = input_data.dropna()
        logger.info(f"Training data loaded '{input_data.shape}'")
        if input_data.shape[0] == 0:
            raise ValueError("Training dataset shape is 0.")

        result = Performance().compute(input_data, sensitive_attributes,
                                       label_column, output_column)

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


if __name__ == '__main__':
    unittest.main()
