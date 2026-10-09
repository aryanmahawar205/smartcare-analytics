"""Unit tests for dataset loading functions."""

import os
import pandas as pd
from src.data.loader import download_dataset, load_raw_data, RAW_DATA_DIR, CSV_FILENAME


def test_download_dataset():
    csv_path = download_dataset(target_dir=RAW_DATA_DIR)
    assert os.path.exists(csv_path)
    assert csv_path.endswith(CSV_FILENAME)


def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "encounter_id" in df.columns
    assert "patient_nbr" in df.columns
    assert "readmitted" in df.columns
