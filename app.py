import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"
SQL_PATH = BASE_DIR / "queries" / "analytical_joins.sql"

TABLES = ["drones", "flights", "telemetry_logs"]
FEATURE_COLUMNS = ["motor_rpm", "package_weight_kg", "avg_wind_speed", "battery_capacity_mah"]
FEATURE_LABELS = ["Motor RPM", "Payload Weight", "Average Wind Speed", "Battery Capacity"]
TARGET_COLUMN = "estimated_flight_time_minutes"
MODEL_PARAMS = {
    "n_estimators": 50,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}


st.set_page_config(
    page_title="Drone Telemetry Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #5f6b7a;
        --line: #d9e0e8;
        --panel: #f7f9fb;
        --accent: #256f8f;
        --accent-2: #8c5a2b;
    }

    header[data-testid="stHeader"] { display: none; }
    .stApp {
        background: #ffffff;
        color: var(--ink);
        font-family: Inter, Segoe UI, sans-serif;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    .page-title {
        border-bottom: 1px solid var(--line);
        margin: -2.5rem 0 1.25rem 0;
        padding: 1.25rem 0 0.85rem 0;
    }

    .page-title h1 {
        font-size: 1.65rem;
        margin: 0;
        font-weight: 700;
    }

    .page-title p {
        color: var(--muted);
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
    }

    .metric-tile {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.95rem 1rem;
        min-height: 96px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .relationship {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 1rem;
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        line-height: 1.8;
    }

    .relationship strong {
        color: var(--accent);
    }

    .note {
        background: #fffaf2;
        border-left: 4px solid var(--accent-2);
        border-radius: 4px;
        color: #3f3429;
        padding: 0.85rem 1rem;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.9rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def stop_if_database_missing() -> None:
    if not DB_PATH.exists():
        st.error(
            "The SQLite warehouse was not found at data/drone_fleet.db. "
            "Run `python generate_data.py` to create the database first."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_sql_text() -> tuple[str, str]:
    full_sql = SQL_PATH.read_text(encoding="utf-8")
    marker = "-- ANALYTICAL JOIN QUERY"
    if marker not in full_sql:
        raise ValueError("Could not find the analytical join query marker in the SQL file.")
    analytical_query = full_sql.split(marker, 1)[1].strip()
    return full_sql, analytical_query


def _schema_frame(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    schema = pd.read_sql_query(f"PRAGMA table_info({table_name});", connection)
    schema = schema.rename(
        columns={
            "name": "column",
            "type": "type",
            "notnull": "required",
            "pk": "primary_key",
        }
    )
    schema["required"] = schema["required"].map({0: "No", 1: "Yes"})
    schema["primary_key"] = schema["primary_key"].map({0: "No", 1: "Yes"})
    return schema[["column", "type", "required", "primary_key"]]


@st.cache_data(show_spinner=False)
def load_warehouse_metadata() -> tuple[dict[str, int], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    stop_if_database_missing()

    with sqlite3.connect(DB_PATH) as connection:
        counts = {
            table_name: pd.read_sql_query(f"SELECT COUNT(*) AS row_count FROM {table_name};", connection)
            .iloc[0]["row_count"]
            .item()
            for table_name in TABLES
        }
        schemas = {table_name: _schema_frame(connection, table_name) for table_name in TABLES}
        previews = {
            table_name: pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 200;", connection)
            for table_name in TABLES
        }

    return counts, schemas, previews


@st.cache_data(show_spinner=False)
def load_joined_dataset() -> pd.DataFrame:
    stop_if_database_missing()
    _, analytical_query = load_sql_text()

    with sqlite3.connect(DB_PATH) as connection:
        joined = pd.read_sql_query(analytical_query, connection)

    return joined


@st.cache_resource(show_spinner=False)
def train_flight_time_model() -> dict[str, object]:
    joined = load_joined_dataset()

    X = joined[FEATURE_COLUMNS]
    y = joined[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    importances = pd.DataFrame(
        {
            "feature": FEATURE_LABELS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "model": model,
        "r2_score": model.score(X_test, y_test),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "importances": importances,
    }


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-title">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, note: str = "") -> None:
    note_html = f"<div class='metric-note'>{note}</div>" if note else ""
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sampled_chart_data(joined: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    if len(joined) <= n:
        return joined
    return joined.sample(n=n, random_state=42)


def show_schema_tabs(schemas: dict[str, pd.DataFrame]) -> None:
    tabs = st.tabs(["drones", "flights", "telemetry_logs"])
    for tab, table_name in zip(tabs, TABLES):
        with tab:
            st.dataframe(schemas[table_name], use_container_width=True, hide_index=True)


def show_table_preview_tabs(previews: dict[str, pd.DataFrame]) -> None:
    tabs = st.tabs(["drones rows", "flights rows", "telemetry_logs rows"])
    for tab, table_name in zip(tabs, TABLES):
        with tab:
            st.dataframe(previews[table_name], use_container_width=True, hide_index=True)


stop_if_database_missing()
counts, schemas, previews = load_warehouse_metadata()

st.markdown("## Drone Telemetry Analytics")

tab_warehouse, tab_sql, tab_analysis, tab_predictor, tab_evaluation = st.tabs(
    [
        "📦 Data Warehouse",
        "🔗 SQL Analytics",
        "📊 Telemetry Analysis",
        "🚁 ML Predictor",
        "📈 Model Evaluation",
    ]
)

with tab_warehouse:
    page_header(
        "Data Warehouse",
        "SQLite tables, row counts, schema, and the real relationship chain used by the project.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile("Drones Table", f"{counts['drones']:,}", "fleet asset records")
    with c2:
        metric_tile("Flights Table", f"{counts['flights']:,}", "one row per simulated flight")
    with c3:
        metric_tile("Telemetry Logs Table", f"{counts['telemetry_logs']:,}", "second-by-second telemetry rows")

    st.subheader("Relationship Diagram")
    st.markdown(
        """
        <div class="relationship">
            <strong>drones</strong> (drone_id primary key)<br>
            &nbsp;&nbsp;down via flights.drone_id<br>
            <strong>flights</strong> (flight_id primary key, drone_id foreign key)<br>
            &nbsp;&nbsp;down via telemetry_logs.flight_id<br>
            <strong>telemetry_logs</strong> (log_id primary key, flight_id foreign key)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Actual Schema")
    show_schema_tabs(schemas)

    st.subheader("Table Samples")
    show_table_preview_tabs(previews)

with tab_sql:
    joined_df = load_joined_dataset()
    full_sql, analytical_query = load_sql_text()

    page_header(
        "SQL Analytics",
        "The dashboard uses the same 3-table INNER JOIN stored in queries/analytical_joins.sql.",
    )

    st.subheader("Project SQL")
    st.code(analytical_query, language="sql")

    j1, j2, j3 = st.columns(3)
    with j1:
        metric_tile("Joined Rows", f"{len(joined_df):,}")
    with j2:
        metric_tile("Joined Columns", f"{joined_df.shape[1]:,}")
    with j3:
        metric_tile("Source Tables", "3", "drones + flights + telemetry_logs")

    st.subheader("Joined Dataset Sample")
    st.dataframe(joined_df.head(200), use_container_width=True, hide_index=True)

    st.subheader("Filtering Examples")
    models = sorted(joined_df["model_name"].unique().tolist())
    selected_models = st.multiselect("Drone models", models, default=models)

    payload_min = float(joined_df["package_weight_kg"].min())
    payload_max = float(joined_df["package_weight_kg"].max())
    wind_min = float(joined_df["avg_wind_speed"].min())
    wind_max = float(joined_df["avg_wind_speed"].max())

    f1, f2 = st.columns(2)
    with f1:
        payload_range = st.slider(
            "Payload range (kg)",
            min_value=payload_min,
            max_value=payload_max,
            value=(payload_min, payload_max),
            step=0.1,
        )
    with f2:
        wind_range = st.slider(
            "Wind speed range (km/h)",
            min_value=wind_min,
            max_value=wind_max,
            value=(wind_min, wind_max),
            step=0.5,
        )

    filtered = joined_df[
        joined_df["model_name"].isin(selected_models)
        & joined_df["package_weight_kg"].between(payload_range[0], payload_range[1])
        & joined_df["avg_wind_speed"].between(wind_range[0], wind_range[1])
    ]

    metric_tile("Filtered Rows", f"{len(filtered):,}", "preview limited to 200 rows")
    st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)

with tab_analysis:
    joined_df = load_joined_dataset()
    page_header(
        "Telemetry Analysis",
        "Exploratory plots using real voltage-drop and flight-time columns from the joined SQLite dataset.",
    )

    chart_df = sampled_chart_data(joined_df)

    st.subheader("Voltage Drop Drivers")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.scatter_chart(
            chart_df,
            x="motor_rpm",
            y="voltage_drop_rate",
            height=280,
            use_container_width=True,
        )
    with p2:
        st.scatter_chart(
            chart_df,
            x="avg_wind_speed",
            y="voltage_drop_rate",
            height=280,
            use_container_width=True,
        )
    with p3:
        st.scatter_chart(
            chart_df,
            x="package_weight_kg",
            y="voltage_drop_rate",
            height=280,
            use_container_width=True,
        )

    st.subheader("Estimated Flight Time Drivers")
    f1, f2, f3, f4 = st.columns(4)
    for column_name, target_column in [
        ("motor_rpm", f1),
        ("package_weight_kg", f2),
        ("avg_wind_speed", f3),
        ("battery_capacity_mah", f4),
    ]:
        with target_column:
            st.scatter_chart(
                chart_df,
                x=column_name,
                y=TARGET_COLUMN,
                height=230,
                use_container_width=True,
            )

    st.subheader("Correlation With Estimated Flight Time")
    corr = (
        joined_df[FEATURE_COLUMNS + [TARGET_COLUMN]]
        .corr(numeric_only=True)[[TARGET_COLUMN]]
        .drop(index=TARGET_COLUMN)
        .rename(columns={TARGET_COLUMN: "correlation_with_flight_time"})
    )
    st.dataframe(corr, use_container_width=True)

    st.subheader("Telemetry Distributions")
    h1, h2, h3, h4 = st.columns(4)
    for column_name, label, target_column in [
        ("motor_rpm", "Motor RPM", h1),
        ("avg_wind_speed", "Wind Speed", h2),
        ("package_weight_kg", "Payload Weight", h3),
        ("battery_capacity_mah", "Battery Capacity", h4),
    ]:
        counts_array, bin_edges = np.histogram(joined_df[column_name], bins=20)
        hist_df = pd.DataFrame({"bin": bin_edges[:-1], "count": counts_array}).set_index("bin")
        with target_column:
            st.caption(label)
            st.bar_chart(hist_df, height=220, use_container_width=True)

with tab_predictor:
    joined_df = load_joined_dataset()
    page_header(
        "ML Predictor",
        "Random Forest inference for estimated remaining flight time.",
    )

    model_info = train_flight_time_model()
    model = model_info["model"]

    d1, d2 = st.columns([1, 1.4], gap="large")

    with d1:
        st.subheader("Inputs")
        rpm_value = st.slider(
            "RPM",
            min_value=int(joined_df["motor_rpm"].min()),
            max_value=int(joined_df["motor_rpm"].max()),
            value=int(joined_df["motor_rpm"].median()),
            step=25,
        )
        payload_value = st.slider(
            "Payload (kg)",
            min_value=float(joined_df["package_weight_kg"].min()),
            max_value=float(joined_df["package_weight_kg"].max()),
            value=float(joined_df["package_weight_kg"].median()),
            step=0.1,
        )
        wind_value = st.slider(
            "Wind Speed (km/h)",
            min_value=float(joined_df["avg_wind_speed"].min()),
            max_value=float(joined_df["avg_wind_speed"].max()),
            value=float(joined_df["avg_wind_speed"].median()),
            step=0.5,
        )
        battery_value = st.select_slider(
            "Battery Capacity (mAh)",
            options=sorted(joined_df["battery_capacity_mah"].unique().tolist()),
            value=int(joined_df["battery_capacity_mah"].median()),
        )

    prediction_row = pd.DataFrame(
        [[rpm_value, payload_value, wind_value, battery_value]],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(prediction_row)[0]

    with d2:
        st.subheader("Prediction")
        st.metric("Predicted Remaining Flight Time", f"{prediction:.1f} minutes")
        st.caption("Estimate assumes a full battery under the selected operating profile.")
        st.dataframe(prediction_row, use_container_width=True, hide_index=True)

with tab_evaluation:
    page_header(
        "Model Evaluation",
        "Training footprint, test split, Random Forest settings, R2 score, and feature importances.",
    )

    model_info = train_flight_time_model()

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.metric("Training Rows", f"{model_info['train_rows']:,}")
    with e2:
        st.metric("Testing Rows", f"{model_info['test_rows']:,}")
    with e3:
        st.metric("Model", "Random Forest")
    with e4:
        st.metric("Trees", str(MODEL_PARAMS["n_estimators"]))
    with e5:
        st.metric("Max Depth", str(MODEL_PARAMS["max_depth"]))

    st.metric("R2 Score", f"{model_info['r2_score']:.4f}")

    st.subheader("Feature Importances")
    importance_df = model_info["importances"]
    st.bar_chart(
        importance_df.set_index("feature"),
        height=280,
        use_container_width=True,
    )
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.subheader("Model Scope")
    st.markdown(
        """
        <div class="note">
            This project uses simulated telemetry. The target column
            <code>estimated_flight_time_minutes</code> is generated from battery drain,
            motor load, payload, wind speed, and battery capacity. A high R2 score is
            expected here; it shows the model learned the synthetic generator relationship.
        </div>
        """,
        unsafe_allow_html=True,
    )
