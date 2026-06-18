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
