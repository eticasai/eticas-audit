# run test python -m unittest discover tests

import logging
import unittest
from eticas.data.loaders import load_dataset
from unittest.mock import patch
logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.ERROR,
    format='[%(levelname)s] %(name)s - %(message)s'
)


class TestDataLoader(unittest.TestCase):

    def test_df_load_csv(self):
        expected = load_dataset('tests/loader_files/test_load.csv')
        self.assertTrue(expected.shape[0] > 0)

    def test_df_load_pickle(self):
        expected = load_dataset('tests/loader_files/test_load.pkl')
        self.assertTrue(expected.shape[0] > 0)

    def test_df_load_pickle_2(self):
        expected = load_dataset('tests/loader_files/test_load.pickle')
        self.assertTrue(expected.shape[0] > 0)

    def test_df_load_parquet(self):
        expected = load_dataset('tests/loader_files/test_load.parquet')
        self.assertTrue(expected.shape[0] > 0)

    def test_file_not_found(self):
        file_path = "non_existent_file.csv"
        with self.assertRaises(FileNotFoundError) as context:
            load_dataset(file_path)
        self.assertIn(f"The file '{file_path}' does not exist.", str(context.exception))

    def test_unsupported_extension(self):
        file_path = "file.unsupported"
        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError) as context:
                load_dataset(file_path)

        self.assertIn("Extension '.unsupported' is not supported.", str(context.exception))


if __name__ == '__main__':
    unittest.main()
