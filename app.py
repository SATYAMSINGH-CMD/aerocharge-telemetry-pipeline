import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Configure the Streamlit page layout
st.set_page_config(page_title="AeroCharge Analytics", page_icon="🛸", layout="centered")

# ==========================================
# PHASE 1: SYSTEM PATHS & DATA INGESTION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"

@st.cache_resource
def load_data_and_train():
    """Connects to SQLite database, extracts logs, and trains the model once."""
    print("🤖 SYSTEM START: Loading data from warehouse...")
    connection = sqlite3.connect(DB_PATH)

    # Your original optimized query remains unchanged
    query = """
    SELECT 
        telemetry_logs.motor_rpm,
        flights.package_weight_kg,
        flights.avg_wind_speed,
        telemetry_logs.voltage_drop_rate
    FROM telemetry_logs
    INNER JOIN flights ON telemetry_logs.flight_id = flights.flight_id;
    """
    df = pd.read_sql_query(query, connection)
    connection.close()

    # ==========================================
    # PHASE 2: BACKGROUND MODEL TRAINING
    # ==========================================
    print("🧠 SYSTEM TRAINING: Optimizing predictive model...")
    X = df[['motor_rpm', 'package_weight_kg', 'avg_wind_speed']]
    y = df['voltage_drop_rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy_score = model.score(X_test, y_test) * 100
    return model, accuracy_score

# Load and train the model using Streamlit's caching so it stays fast
with st.spinner("⏳ Booting analytical engine and training model..."):
    model, accuracy = load_data_and_train()

# ==========================================
# PHASE 3: INTERACTIVE USER INTERFACE (UI)
# ==========================================
st.title("🛸 AeroCharge Telemetry Pipeline Dashboard")
st.markdown("### Real-time Drone Battery Degradation Predictor")
st.caption(f"🌲 Powered by Random Forest Regressor | Model Accuracy: **{accuracy:.2f}%**")
st.markdown("---")

st.subheader("⚙️ Live Flight Parameters")
st.markdown("Modify the flight metrics below using the sliders to see how they impact battery health:")

# Interactive sliders for user input
input_rpm = st.slider("Motor Speed (RPM)", min_value=8000, max_value=18000, value=14000, step=100)
input_weight = st.slider("Package Weight (kg)", min_value=0.5, max_value=8.0, value=2.5, step=0.1)
input_wind = st.slider("Average Wind Speed (km/h)", min_value=0.0, max_value=45.0, value=15.0, step=0.5)

st.markdown("---")

# ==========================================
# PHASE 4: ON-THE-FLY INFERENCE
# ==========================================
# Collect slider values and run them through the trained model
input_features = [[input_rpm, input_weight, input_wind]]
prediction = model.predict(input_features)[0]

st.subheader("🔮 Battery Degradation Forecast")

# Render a clean, non-technical visual breakdown cards display
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Voltage Drop Rate", value=f"{prediction:.4f} V/s")

with col2:
    if prediction > 0.06:
        st.error("⚠️ **CRITICAL DEGRADATION:** Heavy load or high wind stress detected!")
    elif prediction > 0.04:
        st.warning("⚡ **MODERATE STRESS:** Normal operating consumption under active load.")
    else:
        st.success("🟢 **OPTIMAL FLIGHT:** High efficiency trajectory profiles.")