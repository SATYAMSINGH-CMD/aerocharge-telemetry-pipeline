# Drone Telemetry Analyzer

An interactive, multi-subsystem flight telemetry analytics dashboard built to predict drone airframe performance and battery degradation curves using Machine Learning.

🔗 **Live Deployment URL:** [Launch Streamlit Application Dashboard](https://aerocharge-telemetry-pipeline-bhw5fwsh9dtfgbepmmfjvp.streamlit.app/)

## ⚙️ Core Architecture & Subsystems
Instead of running standard heuristic formulas, this analyzer trains **four independent Random Forest Regressor models** simultaneously in the background to handle real-time inference across separate mechanical layers:
1. **Power Plant Degradation:** Predicts real-time battery voltage drop rate ($V/s$).
2. **Thermal Dynamics:** Estimates brushless motor core temperature buildup (°C).
3. **Structural Vibration:** Forecasts airframe high-frequency oscillation metrics ($g$-force).
4. **Aerodynamic Performance:** Simulates physical drag resistance vectors (Newtons).

## 🛠️ Data Pipeline & Features
The intelligence core ingests relational telemetry logs directly from a local SQLite data warehouse (`drone_fleet.db`), parsing highly coupled engineering features:
- **Motor RPM:** Mechanical propulsion speed output.
- **Payload Weight (kg):** Static mass configuration allocation.
- **Average Wind Speed (km/h):** Dynamic environmental turbulence impact.

## 📦 Tech Stack
- **Frontend/UI:** Streamlit Cloud Framework
- **Machine Learning Engine:** Scikit-Learn (Random Forest Ensemble)
- **Data Engineering:** Pandas, NumPy, SQLite3
- **Language:** Python 3.14