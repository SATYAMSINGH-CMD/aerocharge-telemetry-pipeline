# Drone Telemetry Analytics - Codebase Export

This file collects the text/code files in the project so the codebase can be shared from one copy-paste friendly document.

Generated on 2026-06-18.

## Project Tree

```text
drone_telemetry_analytics/
|-- README.md
|-- requirements.txt
|-- app.py
|-- dashboard.py
|-- generate_data.py
|-- main.py
|-- queries/
|   `-- analytical_joins.sql
|-- data/
|   `-- drone_fleet.db              [binary SQLite database, not embedded]
|-- rpm_vs_battery_depletion.png    [binary plot image, not embedded]
`-- wind_vs_rpm_analysis.png        [binary plot image, not embedded]
```

## Files

### README.md

````markdown
# Drone Telemetry Analytics

A SQLite-to-Streamlit analytics project for simulated drone telemetry.

The project focuses on the full data pipeline:

```text
Data generation
-> SQLite warehouse
-> SQL join layer
-> exploratory telemetry analysis
-> Random Forest flight-time prediction
-> Streamlit dashboard
```

Live deployment:
[Launch Streamlit Application Dashboard](https://aerocharge-telemetry-pipeline-bhw5fwsh9dtfgbepmmfjvp.streamlit.app/)

## Database

The local SQLite warehouse is stored at `data/drone_fleet.db` and contains:

```text
drones
  -> flights
      -> telemetry_logs
```

Core tables:

- `drones`: drone model, payload capacity, battery capacity
- `flights`: drone assignment, package weight, average wind speed
- `telemetry_logs`: second-by-second motor RPM, voltage drop rate, and estimated flight time

The analytical query is stored in `queries/analytical_joins.sql` and joins all
three tables with `INNER JOIN`.

## Machine Learning Scope

The model predicts one generated project target:

```text
estimated_flight_time_minutes
```

Inputs:

```text
motor_rpm
package_weight_kg
avg_wind_speed
battery_capacity_mah
```

Model:

```text
RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
```

Important caveat: the telemetry is simulated. The generator creates
`estimated_flight_time_minutes` from battery drain, RPM, payload, wind speed,
and battery capacity, so a high R2 score is expected. This project demonstrates
a data engineering and ML workflow; it does not claim to discover additional
measured outputs.

## Dashboard Pages

- Data Warehouse: row counts, schemas, table samples, and relationships
- SQL Analytics: the project join query, joined dataset size, and filters
- Telemetry Analysis: RPM, wind, and payload plotted against voltage drop
- ML Predictor: predicts remaining flight time from RPM, payload, wind, and battery capacity
- Model Evaluation: train/test rows, model settings, R2, and feature importance

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the SQLite database:

```bash
python generate_data.py
```

Run the dashboard:

```bash
streamlit run app.py
```

Run the analysis script:

```bash
python main.py
```

## Shareable Code Export

`CODEBASE_EXPORT.md` contains the text code from the project files in one
copy-paste friendly document. Binary assets such as the SQLite database and PNG
plots are referenced but not embedded.
````

### requirements.txt

````text
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
````

### app.py

````python
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

st.sidebar.title("Drone Telemetry")
page = st.sidebar.radio(
    "Navigation",
    [
        "Data Warehouse",
        "SQL Analytics",
        "Telemetry Analysis",
        "ML Predictor",
        "Model Evaluation",
    ],
)

if page == "Data Warehouse":
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

elif page == "SQL Analytics":
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

elif page == "Telemetry Analysis":
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

elif page == "ML Predictor":
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

elif page == "Model Evaluation":
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
````

### dashboard.py

````python
"""Compatibility entry point for Streamlit Cloud or older run commands.

The main dashboard lives in app.py. Running `streamlit run dashboard.py` imports
that app so the project has one consistent analytics-first interface.
"""

import app  # noqa: F401
````

### generate_data.py

````python
import random
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"
SQL_PATH = BASE_DIR / "queries" / "analytical_joins.sql"

RANDOM_SEED = 42
FLIGHT_COUNT = 1000

DRONES = [
    {
        "model_name": "AeroLift X1",
        "max_payload_kg": 2.5,
        "battery_capacity_mah": 5000,
    },
    {
        "model_name": "CargoHawk Pro",
        "max_payload_kg": 5.0,
        "battery_capacity_mah": 8500,
    },
    {
        "model_name": "SkyMule Heavy",
        "max_payload_kg": 8.0,
        "battery_capacity_mah": 12000,
    },
]


def load_schema_sql() -> str:
    sql_script = SQL_PATH.read_text(encoding="utf-8")
    return sql_script.split("-- ANALYTICAL JOIN QUERY", 1)[0]


def reset_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS telemetry_logs;
        DROP TABLE IF EXISTS flights;
        DROP TABLE IF EXISTS drones;
        """
    )
    connection.executescript(load_schema_sql())


def build_simulated_records() -> tuple[list[tuple], list[tuple]]:
    flight_records = []
    telemetry_records = []
    telemetry_id_counter = 1

    for flight_id in range(1, FLIGHT_COUNT + 1):
        drone_id = random.randint(1, len(DRONES))
        max_payload = DRONES[drone_id - 1]["max_payload_kg"]

        package_weight_kg = round(random.uniform(0.5, max_payload), 2)
        avg_wind_speed = round(random.uniform(0.0, 45.0), 2)

        flight_records.append(
            (
                flight_id,
                drone_id,
                package_weight_kg,
                avg_wind_speed,
            )
        )

        base_rpm = 3000
        weight_stress = package_weight_kg * 250
        wind_stress = avg_wind_speed * 40
        stable_rpm = base_rpm + weight_stress + wind_stress
        battery_capacity_mah = DRONES[drone_id - 1]["battery_capacity_mah"]
        capacity_factor = battery_capacity_mah / 5000

        # This is a transparent synthetic target for a learning project.
        voltage_drop_rate = ((stable_rpm / 3000) ** 2 * 0.05) / capacity_factor
        estimated_flight_time_minutes = 100 / voltage_drop_rate / 60

        seconds_elapsed = 0
        battery_level = 100.0

        while battery_level > 0:
            current_rpm = int(stable_rpm + random.uniform(-50, 50))
            battery_level = max(0.0, battery_level - voltage_drop_rate)

            telemetry_records.append(
                (
                    telemetry_id_counter,
                    flight_id,
                    seconds_elapsed,
                    round(voltage_drop_rate, 4),
                    round(estimated_flight_time_minutes, 2),
                    current_rpm,
                )
            )

            telemetry_id_counter += 1
            seconds_elapsed += 1

    return flight_records, telemetry_records


def main() -> None:
    random.seed(RANDOM_SEED)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Creating fresh SQLite warehouse...")
    with sqlite3.connect(DB_PATH) as connection:
        reset_database(connection)
        cursor = connection.cursor()

        cursor.executemany(
            """
            INSERT INTO drones (model_name, max_payload_kg, battery_capacity_mah)
            VALUES (:model_name, :max_payload_kg, :battery_capacity_mah)
            """,
            DRONES,
        )

        print(f"Simulating {FLIGHT_COUNT:,} flights and second-by-second telemetry logs...")
        flight_records, telemetry_records = build_simulated_records()

        cursor.executemany(
            """
            INSERT INTO flights (flight_id, drone_id, package_weight_kg, avg_wind_speed)
            VALUES (?, ?, ?, ?)
            """,
            flight_records,
        )

        cursor.executemany(
            """
            INSERT INTO telemetry_logs (
                log_id,
                flight_id,
                seconds_elapsed,
                voltage_drop_rate,
                estimated_flight_time_minutes,
                motor_rpm
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            telemetry_records,
        )

        connection.commit()

    print(f"Inserted {len(DRONES):,} drones.")
    print(f"Inserted {len(flight_records):,} flights.")
    print(f"Inserted {len(telemetry_records):,} telemetry logs.")
    print(f"Database saved to {DB_PATH}.")


if __name__ == "__main__":
    main()
````

### main.py

````python
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"
SQL_PATH = BASE_DIR / "queries" / "analytical_joins.sql"
JOIN_MARKER = "-- ANALYTICAL JOIN QUERY"


def load_join_query() -> str:
    sql_script = SQL_PATH.read_text(encoding="utf-8")
    return sql_script.split(JOIN_MARKER, 1)[1].strip()


def load_joined_dataset() -> pd.DataFrame:
    print("Connecting to the SQLite data warehouse...")
    with sqlite3.connect(DB_PATH) as connection:
        print("Running the 3-table INNER JOIN...")
        return pd.read_sql_query(load_join_query(), connection)


def save_eda_plots(df: pd.DataFrame) -> None:
    print("Generating EDA plots...")
    graph_sample = df.sample(min(2000, len(df)), random_state=42)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=graph_sample,
        x="avg_wind_speed",
        y="motor_rpm",
        hue="model_name",
        palette="viridis",
        alpha=0.8,
    )
    plt.title("Impact of Wind Speed on Motor RPM", fontsize=14, fontweight="bold")
    plt.xlabel("Average Wind Speed (km/h)", fontsize=12)
    plt.ylabel("Motor RPM", fontsize=12)
    wind_plot_path = BASE_DIR / "wind_vs_rpm_analysis.png"
    plt.savefig(wind_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {wind_plot_path.name}")

    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=graph_sample,
        x="motor_rpm",
        y="voltage_drop_rate",
        scatter_kws={"alpha": 0.3, "color": "#2a9d8f"},
        line_kws={"color": "#e76f51", "linewidth": 3},
    )
    plt.title("Motor RPM vs Voltage Drop Rate", fontsize=14, fontweight="bold")
    plt.xlabel("Motor RPM", fontsize=12)
    plt.ylabel("Voltage Drop Rate (V/s)", fontsize=12)
    battery_plot_path = BASE_DIR / "rpm_vs_battery_depletion.png"
    plt.savefig(battery_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {battery_plot_path.name}")


def train_flight_time_model(df: pd.DataFrame) -> None:
    print("Training Random Forest model for estimated_flight_time_minutes...")
    features = ["motor_rpm", "package_weight_kg", "avg_wind_speed", "battery_capacity_mah"]
    X = df[features]
    y = df["estimated_flight_time_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    print(f"Training rows: {X_train.shape[0]:,}")
    print(f"Testing rows: {X_test.shape[0]:,}")

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    r2_score = model.score(X_test, y_test)
    print(f"R2 score: {r2_score:.4f}")
    print("Feature importances:")
    for feature, importance in zip(features, model.feature_importances_):
        print(f"  {feature}: {importance:.4f}")

    print("Target: estimated_flight_time_minutes")
    print(
        "Note: this is simulated telemetry. A high R2 score is expected because "
        "flight time is generated from battery drain and these same input variables."
    )


def main() -> None:
    df = load_joined_dataset()
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
    print("First 5 rows:")
    print(df.head())

    save_eda_plots(df)
    train_flight_time_model(df)


if __name__ == "__main__":
    main()
````

### queries/analytical_joins.sql

````sql
-- ========================================================
-- PHASE 1: WAREHOUSE STRUCTURAL BLUEPRINTS (Leave these here!)
-- ========================================================

CREATE TABLE IF NOT EXISTS drones (
    drone_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    max_payload_kg REAL NOT NULL,
    battery_capacity_mah INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS flights (
    flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    drone_id INTEGER,
    package_weight_kg REAL NOT NULL,
    avg_wind_speed REAL NOT NULL,
    FOREIGN KEY (drone_id) REFERENCES drones (drone_id)
);

CREATE TABLE IF NOT EXISTS telemetry_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_id INTEGER,
    seconds_elapsed INTEGER NOT NULL,
    voltage_drop_rate REAL NOT NULL,
    estimated_flight_time_minutes REAL NOT NULL,
    motor_rpm INTEGER NOT NULL,
    FOREIGN KEY (flight_id) REFERENCES flights (flight_id)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_flight ON telemetry_logs(flight_id);

-- ========================================================
-- ANALYTICAL JOIN QUERY
-- ========================================================

SELECT 
    telemetry_logs.log_id,
    telemetry_logs.flight_id,
    drones.model_name,
    drones.battery_capacity_mah,
    flights.package_weight_kg,   
    telemetry_logs.motor_rpm,
    flights.avg_wind_speed,
    telemetry_logs.voltage_drop_rate,
    telemetry_logs.estimated_flight_time_minutes
FROM telemetry_logs
INNER JOIN flights ON telemetry_logs.flight_id = flights.flight_id
INNER JOIN drones ON flights.drone_id = drones.drone_id;
````
