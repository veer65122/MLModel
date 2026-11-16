from pathlib import Path
import pandas as pd
import streamlit as st

from main1 import (
    analyze_top_products,
    evaluate_prophet_model,
    preprocess_data,
    sales_performance_analysis,
)
# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Retail Sales Forecasting Dashboard")
st.write(
    "Prophet is trained on weekly sales observations. Its weekly predictions "
    "are then combined into calendar-month sales estimates for reporting."
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CANDIDATES = [
    BASE_DIR / "data" / "sales_data.csv",
    BASE_DIR / "data" / "weekly_sales.csv",
    BASE_DIR / "data" / "sales_data.xlsx",
    BASE_DIR / "data" / "weekly_sales.xlsx",
]

# --------------------------------------------------
# 2. DATA LOADING HELPER
# --------------------------------------------------
def read_sales_file(file_or_path):
    """Read an uploaded or local CSV/Excel sales file."""
    name = getattr(file_or_path, "name", str(file_or_path)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_or_path)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path)
    raise ValueError("Please upload a CSV or Excel (.xlsx/.xls) file.")


# --------------------------------------------------
# 3. SIDEBAR: UPLOAD AND FORECAST SETTINGS
# --------------------------------------------------
st.sidebar.header("Data and Forecast Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your sales dataset",
    type=["csv", "xlsx", "xls"],
)

forecast_months = st.sidebar.slider(
    "Monthly Forecast Horizon",
    min_value=1,
    max_value=24,
    value=6,
    step=1,
)

run_evaluation = st.sidebar.checkbox(
    "Evaluate weekly model performance",
    value=True,
)

evaluation_weeks = st.sidebar.slider(
    "Weekly Holdout Period",
    min_value=8,
    max_value=52,
    value=26,
    step=1,
    disabled=not run_evaluation,
)

# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------
try:
    if uploaded_file is not None:
        raw_data = read_sales_file(uploaded_file)
        data_source = uploaded_file.name
    else:
        default_path = next(
            (path for path in DEFAULT_DATA_CANDIDATES if path.exists()),
            None,
        )
        if default_path is None:
            st.info(
                "Upload a CSV/Excel dataset using the sidebar, or place "
                "sales_data.csv / weekly_sales.csv inside the project's data folder."
            )
            st.stop()
        raw_data = read_sales_file(default_path)
        data_source = str(default_path.relative_to(BASE_DIR))
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

st.caption(f"Dataset: {data_source}  |  Columns found: {len(raw_data.columns)}")

# --------------------------------------------------
# 5. COLUMN MAPPING INTERFACE
# --------------------------------------------------
st.subheader("Column Mapping")
st.caption("Map the columns from your uploaded file to the required fields.")

available_columns = list(raw_data.columns)

if not available_columns:
    st.error("No columns found in the uploaded file.")
    st.stop()

file_columns = ["— Not selected —"] + available_columns


def guess_column(candidates, columns):
    """Try to auto-select the best matching column."""
    cols_lower = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return "— Not selected —"

default_date = guess_column(
    ["Month", "Week Ending", "Date", "Week", "Transaction Date", "Order Date", "Bill Date"],
    available_columns,
)
default_sales = guess_column(
    ["Sales Value", "Sales", "Amount", "Revenue", "Net Sales", "Total Sales", "Invoice Value"],
    available_columns,
)
default_units = guess_column(
    ["Revised Sales", "Units", "Quantity", "Qty", "Units Sold", "Volume"],
    available_columns,
)
default_product = guess_column(
    ["Company Product Name", "Product", "Product Name", "Item", "SKU", "Item Name"],
    available_columns,
)

col1, col2 = st.columns(2)

with col1:
    date_col = st.selectbox(
        "Date column (Month / Week Ending) *",
        options=file_columns,
        index=file_columns.index(default_date) if default_date in file_columns else 0,
        key="map_date_col"
    )
    sales_col = st.selectbox(
        "Sales Value column *",
        options=file_columns,
        index=file_columns.index(default_sales) if default_sales in file_columns else 0,
        key="map_sales_col"
    )

with col2:
    units_col = st.selectbox(
        "Revised Sales / Units column (optional)",
        options=file_columns,
        index=file_columns.index(default_units) if default_units in file_columns else 0,
        key="map_units_col"
    )
    product_col = st.selectbox(
        "Product Name column (optional)",
        options=file_columns,
        index=file_columns.index(default_product) if default_product in file_columns else 0,
        key="map_product_col"
    )

# --------------------------------------------------
# Apply column mapping
# --------------------------------------------------
column_mapping = {
    "Month": date_col,
    "Sales Value": sales_col,
}
if units_col != "— Not selected —":
    column_mapping["Revised Sales"] = units_col
if product_col != "— Not selected —":
    column_mapping["Company Product Name"] = product_col

mapped_data = raw_data.rename(columns={v: k for k, v in column_mapping.items()})

# --------------------------------------------------
# GENERATE BUTTON
# --------------------------------------------------
st.markdown("---")
generate = st.button("Generate Forecast & Charts", type="primary", use_container_width=True)

if generate:
    try:
        with st.spinner("Processing data and training model... Please wait."):
            # 1. Preprocess to weekly
            weekly_sales = preprocess_data(mapped_data)

            # 2. Run forecasting
            results = sales_performance_analysis(
                weekly_sales,
                forecast_months=forecast_months,
            )

        st.success("Forecast generated successfully!")

        # ---------- Summary Metrics ----------
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Source Records", f"{len(raw_data):,}")
        with m2:
            st.metric("Weekly Observations", f"{len(weekly_sales):,}")
        with m3:
            total_sales = pd.to_numeric(weekly_sales["Sales Value"], errors="coerce").sum()
            st.metric("Total Sales Value", f"{total_sales:,.2f}")

        # ---------- Top Products ----------
        if "Company Product Name" in mapped_data.columns:
            with st.expander("View Top 10 Products"):
                try:
                    top_products = analyze_top_products(mapped_data, top_n=10)
                    st.dataframe(top_products, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Top-product analysis could not be completed: {exc}")

        # ---------- Charts ----------
        st.subheader("Monthly Sales Forecast")
        st.plotly_chart(results["forecast_fig"], use_container_width=True)

        st.caption(
            "Monthly forecast values are created by allocating each predicted "
            "Monday–Sunday weekly total across calendar months in proportion "
            "to the number of days in each month."
        )

        # ---------- Prediction Table ----------
        st.subheader("Predicted Monthly Sales")
        forecast_df = results["forecast"].copy().rename(
            columns={
                "ds": "Month",
                "yhat": "Predicted Monthly Sales",
                "yhat_lower": "Approx. Lower Limit",
                "yhat_upper": "Approx. Upper Limit",
            }
        )
        st.dataframe(forecast_df, use_container_width=True)

        total_predicted = forecast_df["Predicted Monthly Sales"].sum()
        st.metric(
            f"Total Predicted Sales Across {forecast_months} Months",
            f"{total_predicted:,.2f}",
        )

        # ---------- Components ----------
        st.subheader("Weekly Prophet Components")
        st.pyplot(results["components_fig"], clear_figure=True)

        # ---------- Evaluation ----------
        if run_evaluation:
            st.subheader("Weekly Model Evaluation")
            try:
                evaluation = evaluate_prophet_model(
                    weekly_sales,
                    test_periods=evaluation_weeks,
                )
                metrics = evaluation["metrics"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Weekly MAE", f"{metrics['MAE']:,.2f}")
                c2.metric("Weekly RMSE", f"{metrics['RMSE']:,.2f}")
                c3.metric("Weekly MAPE", f"{metrics['MAPE']:.2f}%")
                if "WAPE" in metrics:
                    c4.metric("Weekly WAPE", f"{metrics['WAPE']:.2f}%")
                elif "Accuracy" in metrics:
                    c4.metric("Approx. Accuracy", f"{metrics['Accuracy']:.2f}%")

                st.caption(
                    f"Evaluation uses the last {evaluation_weeks} weekly observations."
                )
                evaluation_predictions = evaluation["predictions"].rename(
                    columns={
                        "ds": "Week Ending",
                        "y": "Actual Weekly Sales",
                        "yhat": "Predicted Weekly Sales",
                    }
                )
                st.dataframe(evaluation_predictions, use_container_width=True)
            except Exception as exc:
                st.warning(f"Weekly model evaluation was skipped: {exc}")

        # ---------- Download ----------
        csv_data = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Monthly Forecast CSV",
            data=csv_data,
            file_name="retail_monthly_sales_forecast.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")
        st.exception(exc)