import os
import unittest
from unittest.mock import patch, MagicMock
import requests
import pandas as pd

# Importa las funciones a testear desde tu módulo
# coverage run -m unittest discover
# coverage html
from eticas.utils.api import get_audit, get_departments, get_models, get_audits, upload_audit
from eticas.utils.api import scoring_evolution, bias_direction


class TestAPIMethods(unittest.TestCase):

    def setUp(self):
        os.environ["ITACA_API_TOKEN"] = "dummy_token"

    @patch("eticas.utils.api.requests.get")
    def test_get_audit_success(self, mock_get):
        expected_response = {"id": 2289, "audit": "dummy audit"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_get.return_value = mock_response

        result = get_audit(audit_id=2289)
        self.assertEqual(result, expected_response)
        mock_get.assert_called_once()

    @patch("eticas.utils.api.requests.get")
    def test_get_audit_failure(self, mock_get):
        error_text = "Not Found"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = error_text
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            get_audit(audit_id=999)

    @patch("eticas.utils.api.requests.get")
    def test_get_departments_success(self, mock_get):
        expected_data = [{"id": 1, "name": "HR"}, {"id": 2, "name": "IT"}]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_data
        mock_get.return_value = mock_response

        df = get_departments()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(expected_data))

    @patch("eticas.utils.api.requests.get")
    def test_get_model_success(self, mock_get):
        expected_data = [{"id": 10, "name": "Model A"}]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_data
        mock_get.return_value = mock_response

        df = get_models(department=216)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(expected_data))

    @patch("eticas.utils.api.requests.get")
    def test_get_audits_success(self, mock_get):
        expected_results = {"results": [{"id": 100, "name": "Audit 1"}, {"id": 101, "name": "Audit 2"}]}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_results
        mock_get.return_value = mock_response

        df = get_audits(model=263)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(expected_results["results"]))

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        with self.assertRaises(ValueError) as context:
            get_audit(audit_id=2289)
        self.assertEqual(str(context.exception), "❌ 'ITACA_API_TOKEN' NO DEFINED.")

        with self.assertRaises(ValueError) as context:
            get_departments()
        self.assertEqual(str(context.exception), "❌ 'ITACA_API_TOKEN' NO DEFINED.")

        with self.assertRaises(ValueError) as context:
            get_models(department=216)
        self.assertEqual(str(context.exception), "❌ 'ITACA_API_TOKEN' NO DEFINED.")

        with self.assertRaises(ValueError) as context:
            get_audits(model=263)
        self.assertEqual(str(context.exception), "❌ 'ITACA_API_TOKEN' NO DEFINED.")

    @patch("eticas.utils.api.requests.get")
    def test_api_error_responses(self, mock_get):
        error_text = "Internal Server Error"
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = error_text
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError) as context:
            get_audit(audit_id=2289)
        self.assertIn("500", str(context.exception))

        with self.assertRaises(requests.HTTPError) as context:
            get_departments()
        self.assertIn("500", str(context.exception))

        with self.assertRaises(requests.HTTPError) as context:
            get_models(department=216)
        self.assertIn("500", str(context.exception))

        with self.assertRaises(requests.HTTPError) as context:
            get_audits(model=263)
        self.assertIn("500", str(context.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_upload_audit_missing_api_key(self):
        with self.assertRaises(ValueError) as context:
            upload_audit(department_id=1, model_id=100, model={"dummy": "data"})
        self.assertEqual(str(context.exception), "❌ 'ITACA_API_TOKEN' NO DEFINED.")

    @patch("eticas.utils.api.get_departments")
    def test_upload_audit_invalid_department_empty(self, mock_get_departments):
        mock_get_departments.return_value = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            upload_audit(department_id=1, model_id=100, model={"dummy": "data"})
        self.assertEqual(str(context.exception), "Deparment ID does not exist..")

    @patch("eticas.utils.api.get_departments")
    def test_upload_audit_invalid_department_not_in_list(self, mock_get_departments):
        df_dept = pd.DataFrame({"id": [10, 20, 30]})
        mock_get_departments.return_value = df_dept
        with self.assertRaises(ValueError) as context:
            upload_audit(department_id=1, model_id=100, model={"dummy": "data"})
        self.assertEqual(str(context.exception), "Deparment ID does not exist..")

    @patch("eticas.utils.api.get_models")
    @patch("eticas.utils.api.get_departments")
    def test_upload_audit_invalid_model_empty(self, mock_get_departments, mock_get_models):
        df_dept = pd.DataFrame({"id": [1, 2, 3]})
        mock_get_departments.return_value = df_dept
        mock_get_models.return_value = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            upload_audit(department_id=1, model_id=100, model={"dummy": "data"})
        self.assertEqual(str(context.exception), "Model ID does not exist..")

    @patch("eticas.utils.api.get_models")
    @patch("eticas.utils.api.get_departments")
    def test_upload_audit_invalid_model_not_in_list(self, mock_get_departments, mock_get_models):
        df_dept = pd.DataFrame({"id": [1, 2, 3]})
        mock_get_departments.return_value = df_dept
        df_models = pd.DataFrame({"id": [10, 20, 30]})
        mock_get_models.return_value = df_models
        with self.assertRaises(ValueError) as context:
            upload_audit(department_id=1, model_id=100, model={"dummy": "data"})
        self.assertEqual(str(context.exception), "Model ID does not exist..")

    @patch("eticas.utils.api.requests.post")
    @patch("eticas.utils.api.upload_json_audit")
    @patch("eticas.utils.api.get_models")
    @patch("eticas.utils.api.get_departments")
    def test_upload_audit_success(self, mock_get_departments, mock_get_models, mock_upload_json_audit, mock_post):
        df_dept = pd.DataFrame({"id": [1, 2, 3]})
        mock_get_departments.return_value = df_dept
        df_models = pd.DataFrame({"id": [100, 200, 300]})
        mock_get_models.return_value = df_models
        audit_metrics = {"metric": "value"}
        mock_upload_json_audit.return_value = audit_metrics
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        department_id = 1
        model_id = 100
        model_input = {"dummy": "data"}
        status_code = upload_audit(department_id=department_id, model_id=model_id, model=model_input)

        self.assertEqual(status_code, 200)

    def test_scoring_evolution_last_share_gt_ref_and_last_positive_gt_ref(self):
        result = scoring_evolution(first_share=10, last_share=20,
                                   first_positive=5, last_positive=16, ref_share=15)
        self.assertEqual(result["score_first_last"], 100)
        self.assertEqual(result["score_positives_first_last"], 100)

    def test_scoring_evolution_last_share_lt_first_share_and_last_positive_lt_first_positive(self):
        # Caso 2: last_share < first_share y last_positive < first_positive
        result = scoring_evolution(first_share=20, last_share=10,
                                   first_positive=15, last_positive=10, ref_share=15)
        self.assertEqual(result["score_first_last"], 0)
        self.assertEqual(result["score_positives_first_last"], 0)

    def test_scoring_evolution_else_branch_for_both(self):
        result = scoring_evolution(first_share=10, last_share=12,
                                   first_positive=10, last_positive=12, ref_share=15)
        self.assertAlmostEqual(result["score_first_last"], 80.0, places=4)
        self.assertAlmostEqual(result["score_positives_first_last"], 80.0, places=4)

    def test_scoring_evolution_bug_branch_for_positive(self):
        with self.assertRaises(UnboundLocalError):
            scoring_evolution(first_share=10, last_share=12,
                              first_positive=15, last_positive=15, ref_share=15)

    def test_bias_direction(self):
        self.assertEqual(bias_direction(1), "Correct-representation")
        self.assertEqual(bias_direction(0.5), "Under-representation")
        self.assertEqual(bias_direction(2), "Over-representation")


if __name__ == "__main__":
    unittest.main()
