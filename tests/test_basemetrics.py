# run test python -m unittest discover tests

from eticas.metrics.base_metric import BaseFairnessMetric
import pandas as pd
import unittest


class TestFairnessMetric(BaseFairnessMetric):

    def compute(self, input_data):
        return super().compute(input_data)


class TestBaseFairnessMetric(unittest.TestCase):

    def setUp(self):
        # Create an instance of the dummy metric
        self.metric = TestFairnessMetric()
        # Create a simple DataFrame for testing

    def test_compute(self):

        # Attempting to call compute should raise NotImplementedError.
        with self.assertRaises(NotImplementedError) as context:
            self.metric.compute(pd.DataFrame())
        # Optional: Assert that the exception message is as expected.
        self.assertEqual(str(context.exception), "Not implemented.")

    def test_normalize_value(self):
        # Test normalization function.
        with self.assertRaises(NotImplementedError) as context:
            self.metric.compute(self.metric.normalize_value(50))
        # Optional: Assert that the exception message is as expected.
        self.assertEqual(str(context.exception), "Not implemented.")

    def test_get_bias_level(self):
        # Test bias level determination based on normalized value.
        self.assertEqual(self.metric.get_bias_level(85),
                         'Low/no bias', "Value above 80 should result in 'Low/no bias'.")
        self.assertEqual(self.metric.get_bias_level(75),
                         'Medium bias', "Value between 61 and 80 should result in 'Medium bias'.")
        self.assertEqual(self.metric.get_bias_level(60),
                         'High bias', "Value of 60 or less should result in 'High bias'.")
        self.assertEqual(self.metric.get_bias_level(30),
                         'High bias', "Low value should result in 'High bias'.")

    def test_validate_parameters_success(self):
        # Should not raise an error when valid parameters are provided.
        try:
            self.metric.validate_parameters(input_data=1, sensitive_attrs="test", input_features=['age'])
        except ValueError:
            self.fail("validate_parameters raised ValueError unexpectedly with valid parameters.")

    def test_validate_parameters_failure(self):
        # Should raise a ValueError when a parameter is None.
        with self.assertRaises(ValueError) as context:
            self.metric.validate_parameters(input_data=None)
        self.assertIn("You must provide a value for input_data", str(context.exception))

    def test_str_method(self):
        # Test that the __str__ method returns the class name.
        self.assertEqual(str(self.metric), "TestFairnessMetric", "The __str__ method should return the class name.")


if __name__ == '__main__':
    unittest.main()
