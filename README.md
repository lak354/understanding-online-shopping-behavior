# Understanding Online Shopping Behavior Through Data Analytics

An end-to-end data analytics and machine learning project analyzing five years of crowdsourced U.S. Amazon purchase histories. The repository contains modular Python packages for data loading, preprocessing, customer segmentation (RFM + K-Means), customer lifetime value (CLV) regression modeling, automated artifact persistence, and an interactive multi-page Streamlit web dashboard (**PurchaseLens**).

---

## Dataset Source & Academic Citation

The project analyzes the open academic dataset **"Open E-Commerce 1.0: Five years of crowdsourced U.S. Amazon purchase histories with user demographics"**:

* **Source**: Harvard Dataverse
* **DOI**: [10.7910/DVN/YGLYDY](https://doi.org/10.7910/DVN/YGLYDY)
* **Journal Publication**: Berke et al., *Scientific Data* (Nature, 2024), DOI: [10.1038/s41597-024-03329-6](https://doi.org/10.1038/s41597-024-03329-6)
* **Sample**: 1,850,717 purchase records from 5,027 U.S. users spanning 2018 through 2022.

> **Note on Data Placement**: Due to GitHub file size limits (>100 MB), the raw dataset `amazon-purchases.csv` is excluded via `.gitignore`. Download the file from the Dataverse repository and place it inside the `dataverse_files/` directory before running the pipeline.

---

## Repository Structure

```text
.
├── Understanding_Online_Shopping_Behavior_Through_Data_Analytics.ipynb  # Primary analysis notebook
├── Jupyter Notebook.pdf                                                # Rendered PDF export of notebook
├── data_loader.py                                                      # Data loading and schema validation
├── preprocessing.py                                                    # Cleaning & feature engineering
├── segmentation.py                                                     # RFM aggregate & K-Means clustering
├── modeling.py                                                         # CLV feature matrix & Linear Regression
├── persistence.py                                                      # Joblib model & metadata persistence
├── run_pipeline.py                                                     # One-time pipeline driver script
├── app.py                                                              # PurchaseLens Streamlit web app
├── dataverse_files/                                                    # Raw dataset folder (place CSV here)
├── artifacts/                                                          # Saved model binaries (.joblib)
├── requirements.txt                                                    # Python dependencies
├── .gitignore                                                          # Git ignore rules
└── README.md                                                           # Project documentation
```

---

## Setup & Execution Guide

### 1. Install Dependencies
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Place Dataset File
Download `amazon-purchases.csv` from [Harvard Dataverse](https://doi.org/10.7910/DVN/YGLYDY) and place it in the `dataverse_files/` folder:
```text
dataverse_files/amazon-purchases.csv
```

### 3. Run Pipeline (Model Training & Binary Persistence)
Execute the pipeline script once to clean data, compute RFM metrics, fit K-Means ($K=4$), train the Linear Regression CLV model, and save binaries into `artifacts/`:
```bash
python run_pipeline.py
```

### 4. Launch the Streamlit Web Dashboard
Start the multi-page web application. The GUI reloads pre-compiled binaries from `artifacts/` on startup and never retrains models during user interaction:
```bash
streamlit run app.py
```

---

## Analytical & Machine Learning Workflow

1. **Data Preprocessing**: Deduplication, date parsing, non-negative numerical bounds clipping, missing category/state imputation (`unknown`), and calculation of `OrderValue` and `RepeatPurchaseFlag`.
2. **Customer Segmentation**: Computes Recency, Frequency, and Monetary (RFM) aggregates per customer (`Survey ResponseID`) and fits K-Means ($K=4$, `random_state=42`) with profile-driven segment labels (*Champions*, *High-Value Customers*, *At-Risk Customers*, *Regular Customers*).
3. **CLV Predictive Regression**: Trains a Linear Regression model on customer behavioral metrics and top-12 category/state dummy encodings. Evaluated on a 20% test split, the model achieves:
   * **MAE**: \$1,873.43
   * **RMSE**: \$3,674.90
   * **R²**: 0.8471
4. **Error & Residual Analysis**: Section 7.1 of the notebook contains residual distribution plots, homoscedasticity scatter plots, and absolute error analysis across spend ranges.

---

## Software Environment & Version Citation

| Dependency | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.12+` | Core programming runtime |
| **Streamlit** | `1.30.0+` | Web application dashboard (`app.py`) |
| **Pandas** | `2.0.0+` | Dataframe manipulation and aggregations |
| **Scikit-Learn** | `1.3.0+` | KMeans clustering, LinearRegression, StandardScaler |
| **Plotly** | `5.18.0+` | Interactive web visualizations |
| **Joblib** | `1.3.0+` | Binary artifact serialization |
