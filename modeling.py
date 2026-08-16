import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_customer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Aggregates transaction data into customer-level features for CLV modeling.
    Applies top-12 category/state thresholding with an 'other' bucket.
    """
    customer_level = (
        df.groupby('Survey ResponseID')
        .agg(
            CustomerLifetimeValue=('OrderValue', 'sum'),
            NumberOfOrders=('Order Date', 'count'),
            AvgPrice=('Purchase Price Per Unit', 'mean'),
            AvgQuantity=('Quantity', 'mean'),
            RepeatPurchaseFlag=('RepeatPurchaseFlag', 'max'),
            MostPurchasedCategory=('Category', lambda x: x.mode().iat[0] if not x.mode().empty else 'unknown'),
            MostCommonState=('Shipping Address State', lambda x: x.mode().iat[0] if not x.mode().empty else 'unknown')
        )
        .reset_index()
    )

    top_categories = list(customer_level['MostPurchasedCategory'].value_counts().head(12).index)
    top_states = list(customer_level['MostCommonState'].value_counts().head(12).index)

    customer_level['MostPurchasedCategory'] = np.where(
        customer_level['MostPurchasedCategory'].isin(top_categories),
        customer_level['MostPurchasedCategory'],
        'other'
    )
    customer_level['MostCommonState'] = np.where(
        customer_level['MostCommonState'].isin(top_states),
        customer_level['MostCommonState'],
        'other'
    )

    return customer_level, top_categories, top_states


def train_clv_model(
    customer_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[LinearRegression, pd.DataFrame, pd.Series, np.ndarray, dict, pd.DataFrame, list[str]]:
    """
    Encodes customer features, splits 80/20, trains Linear Regression model, and computes performance metrics.
    """
    model_data = pd.get_dummies(
        customer_df.drop(columns=['Survey ResponseID']),
        columns=['MostPurchasedCategory', 'MostCommonState'],
        drop_first=True
    )

    X = model_data.drop(columns=['CustomerLifetimeValue'])
    y = model_data['CustomerLifetimeValue']
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics_dict = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

    coefficients_df = (
        pd.DataFrame({'feature': feature_columns, 'coefficient': model.coef_})
        .assign(abs_coefficient=lambda df_: df_['coefficient'].abs())
        .sort_values('abs_coefficient', ascending=False)
        .reset_index(drop=True)
    )

    return model, X_test, y_test, y_pred, metrics_dict, coefficients_df, feature_columns
