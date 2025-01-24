import os
import logging
import pandas as pd

# Basic logger configuration (customize it to suit your needs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a .pkl, .parquet, or .csv file.

    Parameters
    ----------
    file_path: str
        Full path to the file to be read.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the file data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    Exception
        If an error occurs while loading the file.
    """
    # 1. Check if the file exists
    if not os.path.exists(file_path):
        error_msg = f"The file '{file_path}' does not exist."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 2. Identify the file extension
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    # 3. Load the data based on the extension
    if (extension == '.pkl') | (extension == '.pickle'):
        df = pd.read_pickle(file_path)
        logger.info(f"Dataset successfully loaded from {file_path} (Pickle).")
    elif extension == '.parquet':
        df = pd.read_parquet(file_path)
        logger.info(f"Dataset successfully loaded from {file_path} (Parquet).")
    elif extension == '.csv':
        df = pd.read_csv(file_path)
        logger.info(f"Dataset successfully loaded from {file_path} (CSV).")
    else:
        error_msg = (f"Extension '{extension}' is not supported. "
                     f"Only .pkl, .parquet, or .csv files are allowed.")
        logger.error(error_msg)
        raise ValueError(error_msg)

    return df
