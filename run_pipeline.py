import sys
import time
from pathlib import Path
import pandas as pd

from data_loader import load_and_validate
from preprocessing import clean_and_engineer
from segmentation import compute_rfm, fit_kmeans
from modeling import build_customer_features, train_clv_model
from persistence import save_artifacts


def main():
    print("=" * 70)
    print("      PURCHASELENS: ONE-TIME DATA & MODEL TRAINING PIPELINE      ")
    print("=" * 70)

    start_time = time.time()

    csv_path = Path("dataverse_files/amazon-purchases.csv")
    print(f"\n[1/5] Loading and validating dataset from '{csv_path}'...")
    raw_df, loader_summary = load_and_validate(csv_path)
    print(f"      Loaded {loader_summary['rows']:,} rows and {loader_summary['columns']} columns.")
    print(f"      Missing values count: {loader_summary['total_missing']:,}")

    print("\n[2/5] Cleaning dataset and engineering features...")
    cleaned_df = clean_and_engineer(raw_df)
    print(f"      Cleaned dataset shape: {cleaned_df.shape}")

    date_min = str(cleaned_df['Order Date'].min().strftime('%Y-%m-%d'))
    date_max = str(cleaned_df['Order Date'].max().strftime('%Y-%m-%d'))
    print(f"      Transaction date range: {date_min} to {date_max}")

    print("\n[3/5] Computing RFM metrics & K-Means customer segmentation (k=4)...")
    rfm_df = compute_rfm(cleaned_df)
    kmeans_model, scaler, labeled_rfm, cluster_profiles = fit_kmeans(rfm_df, k=4, random_state=42)
    print("      Customer Segmentation Profiles:")
    print(cluster_profiles.to_string(index=True))

    print("\n[4/5] Building customer features & training CLV Linear Regression model...")
    customer_df, top_categories, top_states = build_customer_features(cleaned_df)
    model, X_test, y_test, y_pred, metrics_dict, coefficients_df, feature_columns = train_clv_model(customer_df)
    print(f"      Model Performance Metrics:")
    print(f"        - MAE:  ${metrics_dict['MAE']:,.2f}")
    print(f"        - RMSE: ${metrics_dict['RMSE']:,.2f}")
    print(f"        - R²:   {metrics_dict['R2']:.4f}")

    print("\n[5/5] Extracting lightweight artifacts for web deployment...")
    # Prepare compact eda_df for fast interactive filtering in Streamlit
    eda_df = cleaned_df[
        ['Order Date', 'Category', 'Shipping Address State', 'OrderYear', 'OrderMonth', 'OrderValue']
    ].copy()

    dataset_summary = {
        'rows': loader_summary['rows'],
        'columns': loader_summary['columns'],
        'date_min': date_min,
        'date_max': date_max,
        'missing_summary': loader_summary['missing_values'],
        'total_customers': int(cleaned_df['Survey ResponseID'].nunique()),
        'total_orders': int(len(cleaned_df)),
        'total_revenue': float(cleaned_df['OrderValue'].sum())
    }

    artifacts_payload = {
        'kmeans_model': kmeans_model,
        'scaler': scaler,
        'clv_model': model,
        'feature_columns': feature_columns,
        'metrics_dict': metrics_dict,
        'cluster_profiles': cluster_profiles,
        'customer_lookup': labeled_rfm,
        'top_categories': top_categories,
        'top_states': top_states,
        'dataset_summary': dataset_summary,
        'test_results': {'y_test': y_test.values, 'y_pred': y_pred},
        'coefficients_df': coefficients_df,
        'eda_df': eda_df
    }

    save_artifacts(artifacts_payload, artifact_dir="artifacts")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"SUCCESS: Pipeline completed in {elapsed:.2f} seconds. All artifacts saved to 'artifacts/'.")
    print("Next step: Run 'streamlit run app.py' to launch the PurchaseLens dashboard.")
    print("=" * 70)


if __name__ == '__main__':
    main()
