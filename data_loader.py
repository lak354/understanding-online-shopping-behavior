import os
from pathlib import Path
import pandas as pd


def load_and_validate(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Loads dataset from csv_path and returns dataframe along with missingness and shape summary.
    Raises FileNotFoundError if file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at '{csv_path}'. Please ensure the file exists.")

    df = pd.read_csv(path)

    missing_counts = df.isna().sum().to_dict()
    missing_percents = (df.isna().mean() * 100).round(2).to_dict()

    summary = {
        'shape': df.shape,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': missing_counts,
        'missing_percent': missing_percents,
        'total_missing': int(df.isna().sum().sum()),
        'column_names': list(df.columns)
    }

    return df, summary
