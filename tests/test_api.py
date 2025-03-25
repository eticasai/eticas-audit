import os
import unittest
from unittest.mock import patch, MagicMock
import requests
import pandas as pd

# Importa las funciones a testear desde tu módulo
# coverage run -m unittest discover
# coverage html
from eticas.utils.api import get_audit, get_departments, get_models, get_audits, upload_audit
from eticas.utils.api import scoring_evolution, bias_direction, overview


class TestAPIMethods(unittest.TestCase):

    def setUp(self):
        os.environ["ITACA_API_TOKEN"] = "dummy_token"
        os.environ["ITACA_BASE_URL"] = "https://itaca.eticas.ai/api/v1/"

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

    @patch.dict(os.environ, {'ITACA_BASE_URL': "https://itaca.eticas.ai/api/v1/"}, clear=True)
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

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_base_url(self):
        with self.assertRaises(ValueError) as context:
            get_audit(audit_id=2289)
        self.assertEqual(str(context.exception), "❌ 'ITACA_BASE_URL' NO DEFINED.")

        with self.assertRaises(ValueError) as context:
            get_departments()
        self.assertEqual(str(context.exception), "❌ 'ITACA_BASE_URL' NO DEFINED.")

        with self.assertRaises(ValueError) as context:
            get_models(department=216)
        self.assertEqual(str(context.exception), "❌ 'ITACA_BASE_URL' NO DEFINED.")

        with self.assertRaises(ValueError) as context:
            get_audits(model=263)
        self.assertEqual(str(context.exception), "❌ 'ITACA_BASE_URL' NO DEFINED.")

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

    @patch.dict(os.environ, {'ITACA_BASE_URL': "https://itaca.eticas.ai/api/v1/"}, clear=True)
    def test_upload_audit_missing_api_key(self):
        with self.assertRaises(ValueError) as context:
            upload_audit(department_id=1, model_id=100, model={"dummy": "data"})
        self.assertEqual(str(context.exception), "❌ 'ITACA_API_TOKEN' NO DEFINED.")

    @patch.dict(os.environ, {'ITACA_API_TOKEN': "DUMMY"}, clear=True)
    def test_upload_audit_missing_base_url(self):
        with self.assertRaises(ValueError) as context:
            upload_audit(department_id=1, model_id=100, model={"dummy": "data"})
        self.assertEqual(str(context.exception), "❌ 'ITACA_BASE_URL' NO DEFINED.")

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

    def test_bias_direction(self):
        self.assertEqual(bias_direction(1), "Correct-representation")
        self.assertEqual(bias_direction(0.5), "Under-representation")
        self.assertEqual(bias_direction(2), "Over-representation")

    def test_overview_single_element_equal(self):
        key0 = "share"
        key1 = "delta"
        no_none_share = [(10, "groupA")]
        ref_evol = 10
        expected = {
            "share_groupA_difference": 0,
            "share_groupA_evol": "equal to"
        }
        result = overview(key0, key1, no_none_share, ref_evol)
        self.assertEqual(result, expected)

    def test_overview_single_element_above(self):
        key0 = "share"
        key1 = "delta"
        no_none_share = [(15, "groupA")]
        ref_evol = 10
        expected = {
            "share_groupA_difference": 5,
            "share_groupA_evol": "above"
        }
        result = overview(key0, key1, no_none_share, ref_evol)
        self.assertEqual(result, expected)

    def test_overview_single_element_below(self):
        key0 = "share"
        key1 = "delta"
        no_none_share = [(8, "groupA")]
        ref_evol = 10
        expected = {
            "share_groupA_difference": 2,
            "share_groupA_evol": "below"
        }
        result = overview(key0, key1, no_none_share, ref_evol)
        self.assertEqual(result, expected)

    def test_overview_two_elements(self):
        key0 = "share"
        key1 = "delta"
        no_none_share = [(8, "A"), (12, "B")]
        ref_evol = 10
        expected = {
            "share_A_difference": 2,
            "share_A_evol": "below",
            "share_B_difference": 2,
            "share_B_evol": "above",
            "delta_A_B": "positive"
        }
        result = overview(key0, key1, no_none_share, ref_evol)
        self.assertEqual(result, expected)

    def test_overview_three_elements(self):
        key0 = "k0"
        key1 = "k1"
        no_none_share = [(10, "X"), (10, "Y"), (5, "Z")]
        ref_evol = 10
        expected = {
            "k0_X_difference": 0,
            "k0_X_evol": "equal to",
            "k0_Y_difference": 0,
            "k0_Y_evol": "equal to",
            "k0_Z_difference": 5,
            "k0_Z_evol": "below",
            "k1_X_Y": "neutral",
            "k1_Y_Z": "negative"
        }
        result = overview(key0, key1, no_none_share, ref_evol)
        self.assertEqual(result, expected)

    def test_scoring_evolution_above_ref(self):
        first_share = 5
        last_share = 15
        ref_share = 10
        result = scoring_evolution(first_share, last_share, ref_share)
        self.assertEqual(result, 100)

    def test_scoring_evolution_below_first(self):
        first_share = 10
        last_share = 5
        ref_share = 8
        result = scoring_evolution(first_share, last_share, ref_share)
        self.assertEqual(result, 0)

    def test_scoring_evolution_normalized(self):
        first_share = 10
        last_share = 12
        ref_share = 20
        result = scoring_evolution(first_share, last_share, ref_share)
        self.assertAlmostEqual(result, 12.0, places=4)


if __name__ == "__main__":
    unittest.main()
