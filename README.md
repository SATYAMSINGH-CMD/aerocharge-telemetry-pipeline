# AeroCharge: Predictive Drone Telemetry Pipeline & ML Core

An end-to-end data platform and full-stack software architecture simulating drone flight physics, archiving core metrics inside a relational database warehouse, and serving real-time machine learning degradation forecasts via a FastAPI microservice and Streamlit dashboard.

## 📊 Engineering Insights & Visualizations
* include your `wind_vs_rpm_analysis.png` image here *
* include your `rpm_vs_battery_depletion.png` image here *

## 🏗️ System Architecture Layout
1. **Simulation Layer (`generate_data.py`):** Multi-loop flight telemetry engine generating near-million-row operational records utilizing kinematic equations.
2. **Warehouse Layer (`data/drone_fleet.db`):** Relational SQLite database maintaining structural normalization across indexed tables.
3. **Transformation Layer (`queries/analytical_joins.sql`):** 3-table analytical `INNER JOIN` matrix optimizations.
4. **Intelligence Layer (`app.py`):** Parallel multi-core trained Random Forest Regressor yielding **99.91% R² accuracy**, exposed via a high-performance FastAPI endpoint.
5. **UI Dashboard Layer (`dashboard.py`):** Interactive pilot mission control frontend built with Streamlit.

## 🚀 How to Run the Ecosystem
1. Install dependencies: `pip install -r requirements.txt`
2. Spin up the backend API: `python app.py`
3. Launch the pilot dashboard in a parallel terminal: `streamlit run dashboard.py`