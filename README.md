# Drone Telemetry Analytics

A SQLite-to-Streamlit analytics project for simulated drone telemetry.

The project focuses on the full data pipeline:

```text
Data generation
-> SQLite warehouse
-> SQL join layer
-> exploratory telemetry analysis
-> Random Forest voltage-drop prediction
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
- `telemetry_logs`: second-by-second motor RPM and voltage drop rate

The analytical query is stored in `queries/analytical_joins.sql` and joins all
three tables with `INNER JOIN`.

## Machine Learning Scope

The model predicts one real project target:

```text
voltage_drop_rate
```

Inputs:

```text
motor_rpm
package_weight_kg
avg_wind_speed
```

Model:

```text
RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
```

Important caveat: the telemetry is simulated. The generator creates
`voltage_drop_rate` from RPM, payload, and wind speed, so a high R2 score is
expected. This project demonstrates a data engineering and ML workflow; it does
not claim to discover additional measured outputs.

## Dashboard Pages

- Data Warehouse: row counts, schemas, table samples, and relationships
- SQL Analytics: the project join query, joined dataset size, and filters
- Telemetry Analysis: RPM, wind, and payload plotted against voltage drop
- ML Predictor: predicts `voltage_drop_rate` from RPM, payload, and wind
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
