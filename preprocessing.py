import pandas as pd
import numpy as np


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset and performs transaction-level feature engineering matching the notebook pipeline.
    """
    cleaned_df = df.copy()

    # Drop duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # Convert Order Date to datetime and drop rows missing essential keys
    cleaned_df['Order Date'] = pd.to_datetime(cleaned_df['Order Date'], errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['Order Date', 'Survey ResponseID'])

    # Fill categorical missing values with 'unknown'
    categorical_cols = ['Category', 'Shipping Address State', 'Title', 'ASIN/ISBN (Product Code)']
    for col in categorical_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna('unknown')

    # Convert numeric fields and enforce lower bound of 0
    cleaned_df['Purchase Price Per Unit'] = pd.to_numeric(cleaned_df['Purchase Price Per Unit'], errors='coerce').fillna(0)
    cleaned_df['Quantity'] = pd.to_numeric(cleaned_df['Quantity'], errors='coerce').fillna(1)
    cleaned_df['Purchase Price Per Unit'] = cleaned_df['Purchase Price Per Unit'].clip(lower=0)
    cleaned_df['Quantity'] = cleaned_df['Quantity'].clip(lower=0)

    # Feature Engineering
    cleaned_df['OrderValue'] = cleaned_df['Purchase Price Per Unit'] * cleaned_df['Quantity']
    cleaned_df['OrderMonth'] = cleaned_df['Order Date'].dt.to_period('M').astype(str)
    cleaned_df['OrderYear'] = cleaned_df['Order Date'].dt.year
    cleaned_df['OrderDayOfWeek'] = cleaned_df['Order Date'].dt.day_name()

    # Customer aggregates mapped to transaction rows
    customer_order_counts = cleaned_df.groupby('Survey ResponseID').size()
    cleaned_df['RepeatPurchaseFlag'] = cleaned_df['Survey ResponseID'].map((customer_order_counts > 1).astype(int))

    customer_clv = cleaned_df.groupby('Survey ResponseID')['OrderValue'].sum()
    cleaned_df['CustomerLifetimeValue'] = cleaned_df['Survey ResponseID'].map(customer_clv)

    return cleaned_df
