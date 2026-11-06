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

st.caption(f"Dataset: {data_source}")


# --------------------------------------------------
# 5. PREPROCESS TO WEEKLY SALES
# --------------------------------------------------
try:
    weekly_sales = preprocess_data(raw_data)
except Exception as exc:
    st.error(f"Weekly data preparation failed: {exc}")
    st.info(
        "The uploaded file must contain a date column named 'Month' or 'Week Ending' and a "
        "numeric 'Sales Value' column. 'Revised Sales' is optional and is "
        "used for units-sold analysis. The date values must be daily transaction "
        "dates or weekly dates, not an already-monthly aggregated series."
    )
    st.stop()


# --------------------------------------------------
# 6. DATASET PREVIEW AND SUMMARY
# --------------------------------------------------
st.subheader("Weekly Training Data")
st.caption(
    "Each source date is assigned to its Monday–Sunday week, labelled by "
    "the Sunday week-ending date. Prophet is trained on these weekly totals."
)
st.dataframe(weekly_sales.head(20), width=True)

metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.metric("Source Records", f"{len(raw_data):,}")
with metric2:
    st.metric("Weekly Observations", f"{len(weekly_sales):,}")
with metric3:
    total_sales = pd.to_numeric(
        weekly_sales["Sales Value"], errors="coerce"
    ).sum()
    st.metric("Total Sales Value", f"{total_sales:,.2f}")


# --------------------------------------------------
# 7. OPTIONAL: TOP PRODUCTS
# --------------------------------------------------
if "Company Product Name" in raw_data.columns:
    with st.expander("View Top 10 Products"):
        try:
            top_products = analyze_top_products(raw_data, top_n=10)
            st.dataframe(top_products, width=True)
        except Exception as exc:
            st.warning(f"Top-product analysis could not be completed: {exc}")


# --------------------------------------------------
# 8. TRAIN WEEKLY MODEL, DISPLAY MONTHLY FORECAST
# --------------------------------------------------
if st.button("Generate Monthly Forecast", type="primary", width=True):
    try:
        with st.spinner("Training Prophet on weekly sales and aggregating predictions by month..."):
            results = sales_performance_analysis(
                weekly_sales,
                forecast_months=forecast_months,
            )

        st.success("Weekly model trained; monthly forecast generated.")

        st.subheader("Historical Weekly Sales")
        st.plotly_chart(results["sales_fig"], width=True)

        st.subheader("Monthly Sales Forecast")
        st.plotly_chart(results["forecast_fig"], width=True)

        st.caption(
            "Monthly forecast values are created by allocating each predicted "
            "Monday–Sunday weekly total across calendar months in proportion "
            "to the number of days in each month. This assumes sales are evenly "
            "distributed within each week. Monthly lower/upper ranges are "
            "approximate aggregates of weekly Prophet bounds."
        )

        st.subheader("Predicted Monthly Sales")
        forecast_df = results["forecast"].copy().rename(
            columns={
                "ds": "Month",
                "yhat": "Predicted Monthly Sales",
                "yhat_lower": "Approx. Lower Limit",
                "yhat_upper": "Approx. Upper Limit",
            }
        )
        st.dataframe(forecast_df, width=True)

        total_predicted = forecast_df["Predicted Monthly Sales"].sum()
        st.metric(
            f"Total Predicted Sales Across {forecast_months} Months",
            f"{total_predicted:,.2f}",
        )

        st.subheader("Weekly Prophet Components")
        st.pyplot(results["components_fig"], clear_figure=True)

        # Weekly holdout evaluation, consistent with the model's training frequency.
        if run_evaluation:
            st.subheader("Weekly Model Evaluation")
            try:
                evaluation = evaluate_prophet_model(
                    weekly_sales,
                    test_periods=evaluation_weeks,
                )
                metrics = evaluation["metrics"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Weekly MAE", f"{metrics['MAE']:,.2f}")
                m2.metric("Weekly RMSE", f"{metrics['RMSE']:,.2f}")
                m3.metric("Weekly MAPE", f"{metrics['MAPE']:.2f}%")
                m4.metric("Weekly WAPE", f"{metrics['WAPE']:.2f}%")

                st.caption(
                    f"Evaluation uses the last {evaluation_weeks} weekly observations. "
                    "These metrics evaluate weekly predictions, while the main "
                    "forecast table displays calendar-month totals."
                )
                evaluation_predictions = evaluation["predictions"].rename(
                    columns={
                        "ds": "Week Ending",
                        "y": "Actual Weekly Sales",
                        "yhat": "Predicted Weekly Sales",
                    }
                )
                st.dataframe(evaluation_predictions, width=True)
            except Exception as exc:
                st.warning(f"Weekly model evaluation was skipped: {exc}")

        # Download monthly forecast
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
