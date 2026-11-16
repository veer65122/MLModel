"""Weekly sales modelling with calendar-month forecast output.

Input data has a date column named `Month` (the source dataset's date field),
plus `Sales Value` and optionally `Revised Sales`.

The input is converted to Monday-Sunday weekly totals. Prophet is trained on
those weekly totals. Future weekly forecasts are then split proportionally by
the number of days that each week overlaps each calendar month, producing
monthly forecast totals for display.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert currency/number strings to numeric values."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[₹$,]", "", regex=True)
        .str.replace(r"\s", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse DD-MM-YYYY source dates and ISO/other standard date formats."""
    # The source notebook used DD-MM-YYYY. Fall back to pandas' normal parser
    # for ISO strings such as YYYY-MM-DD and already-parsed timestamps.
    parsed_dmy = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
    parsed_general = pd.to_datetime(series, errors="coerce")
    return parsed_dmy.fillna(parsed_general)


def _week_ending_sunday(dates: pd.Series) -> pd.Series:
    """Map each date to the Sunday ending its Monday-Sunday week."""
    days_until_sunday = (6 - dates.dt.weekday) % 7
    return (dates + pd.to_timedelta(days_until_sunday, unit="D")).dt.normalize()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction/dated sales records and aggregate to weekly totals.

    Required columns: Month (source date) or Week Ending, plus Sales Value.
    Optional column: Revised Sales (units sold).

    Returns columns: Week Ending, Sales Value, and optionally Revised Sales.
    Each week is Monday through Sunday and is labelled by its Sunday date.
    """
    date_column = "Week Ending" if "Week Ending" in df.columns else "Month"
    required = [date_column, "Sales Value"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Input must contain a date column named 'Month' or 'Week Ending' "
            f"and 'Sales Value'. Missing: {missing}"
        )

    selected = required + (["Revised Sales"] if "Revised Sales" in df.columns else [])
    data = df[selected].copy()

    # The source notebook used DD-MM-YYYY; ISO dates are also supported.
    data[date_column] = _parse_dates(data[date_column])
    data["Sales Value"] = _to_numeric(data["Sales Value"])
    if "Revised Sales" in data.columns:
        data["Revised Sales"] = _to_numeric(data["Revised Sales"])

    data = data.dropna(subset=[date_column, "Sales Value"])
    if data.empty:
        raise ValueError("No valid dated sales records were found.")

    data["Week Ending"] = _week_ending_sunday(data[date_column])

    aggregations = {"Sales Value": "sum"}
    if "Revised Sales" in data.columns:
        aggregations["Revised Sales"] = "sum"

    weekly_sales = (
        data.groupby("Week Ending", as_index=False)
        .agg(aggregations)
        .sort_values("Week Ending")
        .reset_index(drop=True)
    )

    if len(weekly_sales) < 2:
        raise ValueError("At least two valid weekly observations are required.")

    return weekly_sales


def analyze_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return products ranked by total sales value."""
    required = ["Company Product Name", "Sales Value"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for product analysis: {missing}")

    columns = required + (["Revised Sales"] if "Revised Sales" in df.columns else [])
    data = df[columns].copy()
    data["Sales Value"] = _to_numeric(data["Sales Value"])
    if "Revised Sales" in data.columns:
        data["Revised Sales"] = _to_numeric(data["Revised Sales"])

    data = data.dropna(subset=["Company Product Name", "Sales Value"])
    aggregations = {"Sales Value": "sum"}
    if "Revised Sales" in data.columns:
        aggregations["Revised Sales"] = "sum"

    return (
        data.groupby("Company Product Name", as_index=False)
        .agg(aggregations)
        .sort_values("Sales Value", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _prepare_weekly_prophet_data(weekly_sales: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize weekly rows for Prophet."""
    date_column = "Week Ending" if "Week Ending" in weekly_sales.columns else "Month"
    required = [date_column, "Sales Value"]
    missing = [column for column in required if column not in weekly_sales.columns]
    if missing:
        raise ValueError(
            "Weekly forecast input must contain 'Week Ending' (or the source "
            f"date column 'Month') and 'Sales Value'. Missing: {missing}"
        )

    data = weekly_sales[required].copy()
    data["ds"] = _parse_dates(data[date_column])
    data["y"] = _to_numeric(data["Sales Value"])
    data = data.dropna(subset=["ds", "y"])
    if data.empty:
        raise ValueError("No valid weekly dates and sales values were found.")

    data["ds"] = _week_ending_sunday(data["ds"])
    prophet_df = (
        data.groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .reset_index(drop=True)
    )
    if len(prophet_df) < 2:
        raise ValueError("Prophet needs at least two valid weekly observations.")
    return prophet_df


def _create_weekly_model(training_data: pd.DataFrame) -> Prophet:
    """Create Prophet configured for weekly totals."""
    # Avoid fitting a yearly seasonal pattern when training history is too short.
    yearly = len(training_data) >= 104
    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=False,  # Observations are already weekly totals.
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95,
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=0.1
    )
    model.fit(training_data)
    return model


def _weekly_values_to_monthly(
    weekly_values: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """Allocate each Monday-Sunday weekly total across calendar months.

    If a week crosses a month boundary, the weekly total is split according
    to the number of days in that week that fall within each month. This is an
    explicit even-sales-within-the-week approximation.
    """
    monthly_totals: dict[pd.Timestamp, dict[str, float]] = {}

    for row in weekly_values[["ds", *value_columns]].itertuples(index=False, name=None):
        week_end = pd.Timestamp(row[0]).normalize()
        week_start = week_end - pd.Timedelta(days=6)
        values = row[1:]

        month_period = week_start.to_period("M")
        while month_period.start_time <= week_end:
            month_start = month_period.start_time.normalize()
            next_month_start = (month_period + 1).start_time.normalize()
            overlap_start = max(week_start, month_start)
            overlap_end_exclusive = min(
                week_end + pd.Timedelta(days=1), next_month_start
            )
            overlap_days = (overlap_end_exclusive - overlap_start).days

            if overlap_days > 0:
                bucket = monthly_totals.setdefault(
                    month_start, {column: 0.0 for column in value_columns}
                )
                share = overlap_days / 7.0
                for column, value in zip(value_columns, values):
                    if pd.notna(value):
                        bucket[column] += float(value) * share

            month_period += 1

    if not monthly_totals:
        return pd.DataFrame(columns=["ds", *value_columns])

    result = pd.DataFrame(
        [{"ds": month, **values} for month, values in monthly_totals.items()]
    )
    return result.sort_values("ds").reset_index(drop=True)


def sales_performance_analysis(
    weekly_sales: pd.DataFrame,
    forecast_months: int = 6,
) -> dict:
    """Train Prophet on weekly sales and return calendar-month forecasts.

    `forecast_months` is the number of complete calendar months after the
    month containing the latest observed week-ending date.
    """
    if forecast_months < 1:
        raise ValueError("forecast_months must be at least 1.")

    prophet_df = _prepare_weekly_prophet_data(weekly_sales)
    last_week = prophet_df["ds"].max()

    # Create enough weekly steps to cover the end of the requested last month.
    first_forecast_month = last_week.to_period("M") + 1
    last_forecast_month = last_week.to_period("M") + forecast_months
    target_month_end = last_forecast_month.end_time.normalize()
    last_required_sunday = target_month_end + pd.Timedelta(
        days=(6 - target_month_end.weekday()) % 7
    )
    weekly_periods = int((last_required_sunday - last_week).days // 7)
    if weekly_periods < 1:
        raise ValueError("Could not determine future weekly forecast dates.")

    model = _create_weekly_model(prophet_df)

    future = model.make_future_dataframe(
        periods=weekly_periods,
        freq="W-SUN",
        include_history=True,
    )
    full_weekly_forecast = model.predict(future)

    future_weekly = full_weekly_forecast.loc[
        full_weekly_forecast["ds"] > last_week,
        ["ds", "yhat", "yhat_lower", "yhat_upper"],
    ].copy()

    # Convert weekly predictions to calendar-month totals. A week crossing
    # month-end is prorated by the number of days in each month.
    monthly_forecast = _weekly_values_to_monthly(
        future_weekly,
        ["yhat", "yhat_lower", "yhat_upper"],
    )
    month_start = first_forecast_month.start_time.normalize()
    month_end = last_forecast_month.start_time.normalize()
    monthly_forecast = monthly_forecast.loc[
        monthly_forecast["ds"].between(month_start, month_end)
    ].reset_index(drop=True)

    # Historical chart: display the observed weekly data.
    has_units = "Revised Sales" in weekly_sales.columns
    sales_fig = make_subplots(
        rows=2 if has_units else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            ("Weekly Sales Value", "Weekly Units Sold")
            if has_units else ("Weekly Sales Value",)
        ),
    )
    sales_fig.add_trace(
        go.Scatter(
            x=prophet_df["ds"],
            y=prophet_df["y"],
            mode="lines+markers",
            name="Actual Weekly Sales",
        ),
        row=1,
        col=1,
    )

    if has_units:
        unit_date_col = "Week Ending" if "Week Ending" in weekly_sales.columns else "Month"
        units = weekly_sales[[unit_date_col, "Revised Sales"]].copy()
        units["ds"] = _parse_dates(units[unit_date_col])
        units["units"] = _to_numeric(units["Revised Sales"])
        units = units.dropna(subset=["ds", "units"])
        units["ds"] = _week_ending_sunday(units["ds"])
        units = units.groupby("ds", as_index=False)["units"].sum()
        sales_fig.add_trace(
            go.Scatter(
                x=units["ds"],
                y=units["units"],
                mode="lines+markers",
                name="Weekly Units Sold",
            ),
            row=2,
            col=1,
        )

    sales_fig.update_layout(
        height=650 if has_units else 450,
        title="Historical Weekly Retail Sales",
        template="plotly_white",
        hovermode="x unified",
    )
    sales_fig.update_xaxes(title_text="Week Ending", row=2 if has_units else 1, col=1)
    sales_fig.update_yaxes(title_text="Weekly Sales Value", row=1, col=1)
    if has_units:
        sales_fig.update_yaxes(title_text="Units Sold", row=2, col=1)

    # Aggregate historical weekly actuals for a monthly history + monthly
    # forecast chart. Both sides use the same week-to-month allocation rule.
    historical_monthly = _weekly_values_to_monthly(
        prophet_df.rename(columns={"y": "actual"}), ["actual"]
    )

    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=historical_monthly["ds"],
            y=historical_monthly["actual"],
            mode="lines+markers",
            name="Historical Monthly Sales (from weekly data)",
            line=dict(color="#2563eb", width=2),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=monthly_forecast["ds"],
            y=monthly_forecast["yhat"],
            mode="lines+markers",
            name="Predicted Monthly Sales",
            line=dict(color="#dc2626", dash="dash", width=2),
        )
    )
    # These monthly bands are sums of prorated weekly Prophet bounds, so they
    # are indicative ranges rather than a formally calibrated monthly 95% PI.
    forecast_fig.add_trace(
        go.Scatter(
            x=monthly_forecast["ds"],
            y=monthly_forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=monthly_forecast["ds"],
            y=monthly_forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(220, 38, 38, 0.12)",
            line=dict(width=0),
            name="Approximate monthly range",
        )
    )
    forecast_fig.update_layout(
        title=f"Monthly Retail Sales Forecast - Next {forecast_months} Months",
        xaxis_title="Month",
        yaxis_title="Monthly Sales Value",
        template="plotly_white",
        hovermode="x unified",
        height=550,
    )

    components_fig = model.plot_components(full_weekly_forecast)

    return {
        "model": model,
        "weekly_forecast": future_weekly.reset_index(drop=True),
        "forecast": monthly_forecast,
        "historical_monthly": historical_monthly,
        "full_forecast": full_weekly_forecast,
        "sales_fig": sales_fig,
        "forecast_fig": forecast_fig,
        "components_fig": components_fig,
        "weekly_periods_generated": weekly_periods,
    }


def evaluate_prophet_model(
    weekly_sales: pd.DataFrame,
    test_periods: int = 46,
) -> dict:
    """Evaluate weekly forecasts on the final `test_periods` observations."""
    if test_periods < 1:
        raise ValueError("test_periods must be at least 1.")

    data = _prepare_weekly_prophet_data(weekly_sales)
    if len(data) <= test_periods + 1:
        raise ValueError(
            "Not enough weekly observations for this holdout. "
            "Reduce the holdout weeks or provide more history."
        )

    train = data.iloc[:-test_periods].copy()
    test = data.iloc[-test_periods:].copy()

    model = _create_weekly_model(train)
    future = model.make_future_dataframe(
        periods=test_periods,
        freq='W-SUN'
    )
    forecast_eval = model.predict(future)

    predictions = test[['ds', 'y']].merge(
        forecast_eval[['ds', 'yhat']],
        on='ds',
        how='left'
    )

    actual = predictions['y'].to_numpy(dtype=float)
    predicted = predictions['yhat'].to_numpy(dtype=float)
    errors = actual - predicted

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
        if np.any(nonzero) else float("nan")
    )

    accuracy = 100 - mape if not np.isnan(mape) else float("nan")

    return {
        "model": model,
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Accuracy": accuracy,
        },
        "predictions": predictions,
    }