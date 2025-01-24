from eticas.audit.base_audit import BaseAudit
import unittest


class TestBaseAudit(BaseAudit):

    def run_audit(self, dataset_path, label_column, output_column, positive_output):
        return super().run_audit(dataset_path, label_column, output_column, positive_output)


class TestBaseFairnessMetric(unittest.TestCase):

    def setUp(self):
        # Create an instance of the dummy metric
        self.audit = TestBaseAudit(None)
        # Create a simple DataFrame for testing

    def test_audit(self):

        # Attempting to call compute should raise NotImplementedError.
        with self.assertRaises(NotImplementedError) as context:
            self.audit.run_audit(None, None, None, None)
        # Optional: Assert that the exception message is as expected.
        self.assertEqual(str(context.exception), "Not implemented.")

    def test_str_method(self):
        # Test that the __str__ method returns the class name.
        self.assertEqual(str(self.audit), "TestBaseAudit", "The __str__ method should return the class name.")


if __name__ == '__main__':
    unittest.main()
