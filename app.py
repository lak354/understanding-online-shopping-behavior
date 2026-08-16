import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from persistence import load_artifacts

# Page Configuration
st.set_page_config(
    page_title="PurchaseLens - Amazon E-Commerce Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def fetch_artifacts():
    """Loads pre-built pipeline artifacts safely with error trapping."""
    try:
        artifacts = load_artifacts("artifacts")
        return artifacts, None
    except Exception as e:
        return None, str(e)


# Load Pipeline Artifacts
artifacts, load_error = fetch_artifacts()

# Sidebar Navigation & Summary Panel
st.sidebar.title("PurchaseLens")
st.sidebar.markdown("**Amazon Behavioral Analytics**")

if load_error:
    st.error("### Pipeline Artifacts Not Found")
    st.error(load_error)
    st.info("Resolution Steps: Run 'python run_pipeline.py' in your workspace terminal to train models and generate required binaries.")
    st.stop()

# Sidebar Summary Statistics
summary = artifacts['dataset_summary']
metrics = artifacts['metrics_dict']
cluster_profiles = artifacts['cluster_profiles']

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.metric("Total Customers", f"{summary['total_customers']:,}")
st.sidebar.metric("Customer Segments", f"{len(cluster_profiles)}")
st.sidebar.metric("CLV Model R²", f"{metrics['R2']:.4f}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["1. Overview", "2. Explore (EDA)", "3. Segments", "4. Predict CLV", "5. Model Performance"]
)

# ------------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ------------------------------------------------------------------------------
if page == "1. Overview":
    st.title("Dataset & Executive Overview")
    st.markdown("""
    Welcome to **PurchaseLens**, an e-commerce data analytics interface analyzing five years of crowdsourced U.S. Amazon purchase histories 
    *(Harvard Dataverse Open E-Commerce 1.0 dataset)*.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Purchases", f"{summary['total_orders']:,}")
    col2.metric("Total Revenue", f"${summary['total_revenue']:,.2f}")
    col3.metric("Tracked Customers", f"{summary['total_customers']:,}")
    col4.metric("Date Range", f"{summary['date_min']} to {summary['date_max']}")

    st.markdown("---")
    st.subheader("Dataset Schema & Missingness Summary")

    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown(f"**Dataset Dimensions**: `{summary['rows']:,}` rows $\\times$ `{summary['columns']}` columns")
        st.markdown(f"**Observation Window**: `{summary['date_min']}` to `{summary['date_max']}`")

    missing_df = pd.DataFrame({
        "Column Name": list(summary['missing_summary'].keys()),
        "Missing Value Count": list(summary['missing_summary'].values())
    })
    missing_df["Missing Percentage (%)"] = (missing_df["Missing Value Count"] / summary['rows'] * 100).round(2)

    with summary_cols[1]:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# PAGE 2: EXPLORE (EDA)
# ------------------------------------------------------------------------------
elif page == "2. Explore (EDA)":
    st.title("Exploratory Data Analysis")
    st.markdown("Interactively filter purchase volumes and product categories across geographic regions and time ranges.")

    eda_df = artifacts['eda_df']
    top_categories = artifacts['top_categories']
    top_states = artifacts['top_states']

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([2, 2, 3])

    with f_col1:
        selected_cats = st.multiselect(
            "Filter Categories",
            options=sorted(eda_df['Category'].unique()),
            default=top_categories[:5]
        )

    with f_col2:
        selected_states = st.multiselect(
            "Filter Shipping States",
            options=sorted(eda_df['Shipping Address State'].unique()),
            default=top_states[:5]
        )

    min_year = int(eda_df['OrderYear'].min())
    max_year = int(eda_df['OrderYear'].max())

    with f_col3:
        selected_years = st.slider(
            "Filter Order Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

    # Filter Application
    filtered = eda_df.copy()
    if selected_cats:
        filtered = filtered[filtered['Category'].isin(selected_cats)]
    if selected_states:
        filtered = filtered[filtered['Shipping Address State'].isin(selected_states)]

    filtered = filtered[
        (filtered['OrderYear'] >= selected_years[0]) & (filtered['OrderYear'] <= selected_years[1])
    ]

    st.markdown("---")

    if filtered.empty:
        st.warning("No transactions match the selected filter criteria. Please broaden your selection.")
    else:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Top 10 Revenue Categories (Filtered)")
            cat_rev = (
                filtered.groupby('Category')['OrderValue']
                .sum()
                .reset_index()
                .sort_values('OrderValue', ascending=False)
                .head(10)
            )
            fig_bar = px.bar(
                cat_rev,
                x='OrderValue',
                y='Category',
                orientation='h',
                color='OrderValue',
                color_continuous_scale='Viridis',
                labels={'OrderValue': 'Total Revenue ($)', 'Category': 'Category'},
                title="Revenue by Category ($)"
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.subheader("Monthly Purchase Volume Trend")
            monthly_vol = (
                filtered.groupby('OrderMonth')
                .size()
                .reset_index(name='OrderCount')
                .sort_values('OrderMonth')
            )
            fig_line = px.line(
                monthly_vol,
                x='OrderMonth',
                y='OrderCount',
                markers=True,
                line_shape='linear',
                title="Monthly Transaction Volume",
                labels={'OrderMonth': 'Month', 'OrderCount': 'Purchases'}
            )
            fig_line.update_traces(line_color='#2a9d8f', line_width=2.5)
            fig_line.update_layout(xaxis_tickangle=-45, height=420)
            st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------------------------------------------------------
# PAGE 3: SEGMENTS
# ------------------------------------------------------------------------------
elif page == "3. Segments":
    st.title("Customer Segmentation (RFM + K-Means)")
    st.markdown("Customers grouped into behavioral segments based on **Recency** (days since purchase), **Frequency** (order count), and **Monetary** (total spend).")

    customer_lookup = artifacts['customer_lookup']
    cluster_profiles = artifacts['cluster_profiles']

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.subheader("Recency vs. Monetary Spend Scatter Plot")
        fig_scatter = px.scatter(
            customer_lookup,
            x='Recency',
            y='Monetary',
            color='segment_name',
            hover_data=['Survey ResponseID', 'Frequency'],
            opacity=0.75,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                'Recency': 'Recency (Days Inactive)',
                'Monetary': 'Monetary Spend ($)',
                'segment_name': 'Segment'
            },
            title="Customer Clusters: Recency vs Monetary"
        )
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_table:
        st.subheader("Cluster Profile Summary")
        st.dataframe(cluster_profiles, use_container_width=True)

    st.markdown("---")
    st.subheader("Customer Segment Lookup Tool")

    search_id = st.text_input("Enter Survey ResponseID (e.g. customer GUID):", placeholder="Paste Survey ResponseID here...")

    if search_id:
        search_id = search_id.strip()
        matched = customer_lookup[customer_lookup['Survey ResponseID'] == search_id]

        if matched.empty:
            st.warning(f"Customer ID '{search_id}' was not found in the segmentation registry.")
        else:
            cust = matched.iloc[0]
            st.success(f"Found Customer: **{search_id}**")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Segment", str(cust['segment_name']))
            res_col2.metric("Recency", f"{cust['Recency']} days")
            res_col3.metric("Frequency", f"{cust['Frequency']} orders")
            res_col4.metric("Monetary Spend", f"${cust['Monetary']:,.2f}")

            # Plain English Explanation
            seg_name = cust['segment_name']
            if seg_name == "Champions":
                explanation = f"Classified as **Champions** because this customer demonstrates exceptional cumulative spend (${cust['Monetary']:,.2f}) and recent interaction."
            elif seg_name == "High-Value Customers" or seg_name == "Loyal Customers":
                explanation = f"Classified as **{seg_name}** due to their high purchase frequency ({cust['Frequency']} transactions) and strong engagement."
            elif seg_name == "At-Risk Customers":
                explanation = f"Classified as **At-Risk** because {cust['Recency']} days have elapsed since their last recorded order."
            else:
                explanation = f"Classified as **Regular Customers** reflecting moderate spending (${cust['Monetary']:,.2f}) and typical purchase intervals."

            st.info(f"Segment Rationale: {explanation}")

# ------------------------------------------------------------------------------
# PAGE 4: PREDICT CLV
# ------------------------------------------------------------------------------
elif page == "4. Predict CLV":
    st.title("Customer Lifetime Value (CLV) Predictor")
    st.markdown("Estimate expected Customer Lifetime Value based on transactional attributes.")

    clv_model = artifacts['clv_model']
    feature_columns = artifacts['feature_columns']
    top_categories = artifacts['top_categories']
    top_states = artifacts['top_states']

    # Dynamic Caveat Box
    st.warning(
        f"Model Integrity Caveat: This model's R² of **{metrics['R2']:.3f}** is partly driven by feature leakage — "
        f"several input attributes (e.g., number of orders and average price) are mathematically related to the target (`OrderValue`). "
        f"Treat this prediction as an explanatory baseline estimate, not an unconstrained causal forecast."
    )

    with st.form("clv_prediction_form"):
        st.subheader("Customer Transaction Profile Inputs")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            num_orders = st.number_input("Number of Orders", min_value=1, value=5, step=1)
            avg_price = st.number_input("Average Price Per Unit ($)", min_value=0.0, value=25.50, step=1.0)
            avg_qty = st.number_input("Average Quantity Per Order", min_value=1.0, value=1.5, step=0.5)

        with p_col2:
            repeat_flag = st.checkbox("Is Repeat Customer? (More than 1 purchase)", value=True)
            sel_cat = st.selectbox("Most Purchased Category", options=top_categories + ['other'])
            sel_state = st.selectbox("Most Common Shipping State", options=top_states + ['other'])

        submitted = st.form_submit_button("Predict Customer Lifetime Value")

    if submitted:
        # Input Validation
        if num_orders <= 0 or avg_price < 0 or avg_qty <= 0:
            st.error("Invalid Input: Orders must be > 0, Price >= 0, and Quantity > 0.")
        else:
            # Construct Feature Vector dynamically matching feature_columns exactly
            input_dict = {col: 0.0 for col in feature_columns}

            # Numerical assignments
            if 'NumberOfOrders' in input_dict:
                input_dict['NumberOfOrders'] = float(num_orders)
            if 'AvgPrice' in input_dict:
                input_dict['AvgPrice'] = float(avg_price)
            if 'AvgQuantity' in input_dict:
                input_dict['AvgQuantity'] = float(avg_qty)
            if 'RepeatPurchaseFlag' in input_dict:
                input_dict['RepeatPurchaseFlag'] = 1.0 if repeat_flag else 0.0

            # One-hot dummy assignments
            cat_col = f"MostPurchasedCategory_{sel_cat}"
            if cat_col in input_dict:
                input_dict[cat_col] = 1.0

            state_col = f"MostCommonState_{sel_state}"
            if state_col in input_dict:
                input_dict[state_col] = 1.0

            # Create 1-row DataFrame matching feature matrix
            input_df = pd.DataFrame([input_dict], columns=feature_columns)

            prediction = clv_model.predict(input_df)[0]
            st.success(f"### Predicted Customer Lifetime Value: **${prediction:,.2f} USD**")

# ------------------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE
# ------------------------------------------------------------------------------
elif page == "5. Model Performance":
    st.title("Model Evaluation & Regression Diagnostics")
    st.markdown("Performance evaluation of the Linear Regression Customer Lifetime Value model evaluated on held-out test data (20% split).")

    test_res = artifacts['test_results']
    coef_df = artifacts['coefficients_df']

    # Performance Metrics Table
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:,.2f}")
    m_col2.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:,.2f}")
    m_col3.metric("R² Score", f"{metrics['R2']:.4f}")

    st.markdown("---")
    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.subheader("Actual vs. Predicted CLV (Test Set)")
        y_test = test_res['y_test']
        y_pred = test_res['y_pred']

        fig_pred = px.scatter(
            x=y_test,
            y=y_pred,
            opacity=0.55,
            labels={'x': 'Actual CLV ($)', 'y': 'Predicted CLV ($)'},
            title="Actual vs Predicted CLV"
        )

        min_val = min(float(np.min(y_test)), float(np.min(y_pred)))
        max_val = max(float(np.max(y_test)), float(np.max(y_pred)))

        fig_pred.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Identity Line (y=x)',
                line=dict(color='red', dash='dash', width=2)
            )
        )
        fig_pred.update_layout(height=450)
        st.plotly_chart(fig_pred, use_container_width=True)

    with eval_col2:
        st.subheader("Top 15 Regression Coefficients")
        top_coefs = coef_df.head(15).sort_values('coefficient', ascending=True)

        fig_coef = px.bar(
            top_coefs,
            x='coefficient',
            y='feature',
            orientation='h',
            color='coefficient',
            color_continuous_scale='Coolwarm',
            title="Regression Feature Importance"
        )
        fig_coef.update_layout(height=450)
        st.plotly_chart(fig_coef, use_container_width=True)
