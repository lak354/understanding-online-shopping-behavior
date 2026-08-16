import os
import logging
from pathlib import Path
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PurchaseLens.Persistence")


def save_artifacts(artifacts_dict: dict, artifact_dir: str | Path = "artifacts") -> None:
    """
    Saves pipeline models, metadata, and lookup dataframes into artifact_dir using joblib.
    """
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, content in artifacts_dict.items():
        file_path = out_dir / f"{name}.joblib"
        joblib.dump(content, file_path)
        logger.info(f"Artifact '{name}' successfully saved to {file_path}")


def load_artifacts(artifact_dir: str | Path = "artifacts") -> dict:
    """
    Loads saved pipeline artifacts with error checking for failure modes.
    Returns a dictionary of artifacts. Raises FileNotFoundError or Exception with detailed context.
    """
    out_dir = Path(artifact_dir)
    if not out_dir.exists():
        msg = f"Artifact directory '{artifact_dir}' does not exist. Please run 'python run_pipeline.py' first."
        logger.error(msg)
        raise FileNotFoundError(msg)

    required_artifacts = [
        "kmeans_model",
        "scaler",
        "clv_model",
        "feature_columns",
        "metrics_dict",
        "cluster_profiles",
        "customer_lookup",
        "top_categories",
        "top_states",
        "dataset_summary",
        "test_results",
        "coefficients_df",
        "eda_df"
    ]

    loaded_artifacts = {}
    missing_files = []

    for name in required_artifacts:
        file_path = out_dir / f"{name}.joblib"
        if not file_path.exists():
            missing_files.append(str(file_path))
            continue

        try:
            artifact_data = joblib.load(file_path)
            loaded_artifacts[name] = artifact_data
        except Exception as e:
            msg = f"Corrupt pickle or failed load for '{name}.joblib': {str(e)}"
            logger.error(msg)
            raise ValueError(msg) from e

    if missing_files:
        msg = f"Missing required artifact files: {', '.join(missing_files)}. Please execute 'python run_pipeline.py'."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Basic Schema Validation
    if not isinstance(loaded_artifacts.get("feature_columns"), list):
        raise TypeError("Schema Mismatch: 'feature_columns' artifact must be a list of strings.")

    if not isinstance(loaded_artifacts.get("metrics_dict"), dict):
        raise TypeError("Schema Mismatch: 'metrics_dict' artifact must be a dictionary.")

    logger.info("All pipeline artifacts successfully reloaded from cache.")
    return loaded_artifacts
