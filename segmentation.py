import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Recency, Frequency, and Monetary metrics per customer (Survey ResponseID).
    """
    customer_orders = df.copy()
    reference_date = customer_orders['Order Date'].max() + pd.Timedelta(days=1)

    rfm = (
        customer_orders.groupby('Survey ResponseID')
        .agg(
            Recency=('Order Date', lambda x: (reference_date - x.max()).days),
            Frequency=('Order Date', 'count'),
            Monetary=('OrderValue', 'sum')
        )
        .reset_index()
    )

    rfm = rfm[(rfm['Monetary'] > 0) & (rfm['Recency'] >= 0)].copy()
    return rfm


def fit_kmeans(
    rfm_df: pd.DataFrame,
    k: int = 4,
    random_state: int = 42
) -> tuple[KMeans, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Fits StandardScaler and KMeans clustering on RFM features, assigns cluster names based on profiles.
    Returns (kmeans_model, scaler, labeled_rfm_df, cluster_profile_df).
    """
    rfm = rfm_df.copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Compute cluster aggregate profiles
    cluster_profile = (
        rfm.groupby('Cluster')
        .agg(
            customers=('Survey ResponseID', 'count'),
            avg_recency=('Recency', 'mean'),
            avg_frequency=('Frequency', 'mean'),
            avg_monetary=('Monetary', 'mean')
        )
        .round(2)
    )

    # Profile-driven cluster naming logic
    cluster_names = {}

    # Highest monetary spend -> Champions
    top_monetary = cluster_profile['avg_monetary'].idxmax()
    cluster_names[top_monetary] = "Champions"

    # Highest recency (days inactive) -> At-Risk Customers
    top_recency = cluster_profile['avg_recency'].idxmax()
    if top_recency not in cluster_names:
        cluster_names[top_recency] = "At-Risk Customers"

    # Highest frequency among remaining -> High-Value Customers / Loyal Customers
    remaining_by_freq = cluster_profile.drop(index=list(cluster_names.keys()))['avg_frequency'].idxmax()
    if remaining_by_freq not in cluster_names:
        cluster_names[remaining_by_freq] = "High-Value Customers"

    # Remaining cluster -> Regular Customers
    for c_id in cluster_profile.index:
        if c_id not in cluster_names:
            cluster_names[c_id] = "Regular Customers"

    cluster_profile['segment_name'] = cluster_profile.index.map(cluster_names)
    rfm['segment_name'] = rfm['Cluster'].map(cluster_names)

    return kmeans, scaler, rfm, cluster_profile
