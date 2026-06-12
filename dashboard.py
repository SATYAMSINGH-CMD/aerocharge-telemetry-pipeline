import streamlit as st
import requests

# ==========================================
# PHASE 1: UI CANVAS CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AeroCharge: Telemetry Dashboard",
    page_icon="🛸",
    layout="centered"
)

st.title("🛸 AeroCharge Mission Control")
st.subheader("Real-Time Predictive Flight Telemetry Interface")
st.write(
    "Adjust the environmental and mechanical flight factors below. Your inputs are transmitted "
    "to our background Machine Learning service to calculate real-time battery voltage degradation."
)

st.markdown("---")

# ==========================================
# PHASE 2: INTERACTIVE PILOT CONTROL SLIDERS
# ==========================================
st.sidebar.header("🛠️ Flight Environmental Controls")

# Sliders calibrated to your data warehouse boundaries
motor_rpm = st.sidebar.slider("Motor Operational Velocity (RPM)", min_value=3000, max_value=7000, value=4500, step=50)
package_weight = st.sidebar.slider("Package Payload Weight (kg)", min_value=1.0, max_value=10.0, value=3.5, step=0.1)
avg_wind_speed = st.sidebar.slider("Average Cross-Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)

# ==========================================
# PHASE 3: BACKEND API PACKET BRIDGE
# ==========================================
# Compile the slider values into a structured JSON payload match our data contract
payload = {
    "motor_rpm": int(motor_rpm),
    "package_weight_kg": float(package_weight),
    "avg_wind_speed": float(avg_wind_speed)
}

# The target URL where our FastAPI app server is listening
API_URL = "http://127.0.0.1:8000/predict"

try:
    # Ship the data across our local loop link via an HTTP POST request
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        predicted_drain = result["predicted_voltage_drop_rate"]
        
        # ==========================================
        # PHASE 4: VISUAL GRAPHIC OUTCOMES
        # ==========================================
        st.header("📊 Real-Time AI Predictions")
        
        # Display the result inside a beautiful metric card component
        st.metric(
            label="⚡ Predicted Battery Voltage Drop Rate",
            value=f"{predicted_drain} Volts / sec"
        )
        
        # Add dynamic safety alert logic based on structural thresholds
        if predicted_drain > 0.15:
            st.error("🚨 WARNING: High battery degradation detected! Reduce cruising velocity or payload weight immediately to avoid catastrophic voltage sag.")
        elif predicted_drain > 0.09:
            st.warning("⚠️ CAUTION: Increased power draw. Total mission flight range will be significantly compromised.")
        else:
            st.success("🟢 SAFE OPERATIONAL WINDOW: Battery depletion rate is within nominal cruising specs.")
            
except requests.exceptions.ConnectionError:
    st.error("❌ BACKEND CONNECTION ERROR: Could not talk to the FastAPI data layer. Make sure you open a separate terminal pane and keep 'python app.py' running!")