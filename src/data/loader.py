"""Data loading utilities for SmartCare Analytics."""

import os
import zipfile
import io
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

UCI_DIABETES_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
RAW_DATA_DIR = "data/raw"
CSV_FILENAME = "diabetic_data.csv"
IDS_FILENAME = "IDS_mapping.csv"


def download_dataset(target_dir: str = RAW_DATA_DIR, force_download: bool = False) -> str:
    """Download and extract the UCI Diabetes 130-US Hospitals dataset if not present.

    Args:
        target_dir: Directory where raw dataset files should be stored.
        force_download: Whether to force re-download even if files exist.

    Returns:
        Path to the raw diabetic_data.csv file.
    """
    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, CSV_FILENAME)

    if os.path.exists(csv_path) and not force_download:
        logger.info(f"Dataset already exists at {csv_path}")
        return csv_path

    logger.info(f"Downloading dataset from {UCI_DIABETES_URL}...")
    try:
        response = requests.get(UCI_DIABETES_URL, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(target_dir)
        logger.info(f"Successfully downloaded and extracted dataset to {target_dir}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        if not os.path.exists(csv_path):
            raise RuntimeError(
                f"Dataset missing and automatic download failed ({e}). "
                f"Please manually download the zip from {UCI_DIABETES_URL} "
                f"and extract '{CSV_FILENAME}' into '{target_dir}'."
            ) from e

    return csv_path


def load_raw_data(data_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """Load the raw dataset into a Pandas DataFrame. Downloads if necessary.

    Args:
        data_dir: Directory containing raw data.

    Returns:
        Raw DataFrame.
    """
    csv_path = download_dataset(target_dir=data_dir)
    df = pd.read_csv(csv_path)
    return df
