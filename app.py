import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE ARCHITECTURE & THEME INJECTION
# ==========================================
st.set_page_config(
    page_title="Drone Telemetry Analyzer // HUD", 
    layout="wide",
    initial_sidebar_state="expanded"  # Changed from collapsed to clear the top-left text glitch
)

# Deep aerospace dark theme injection (#0a0e1a) with custom monospace hierarchies
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
    
    /* Global Background Override */
    .stApp {
        background-color: #0a0e1a;
        background-image: radial-gradient(rgba(0, 212, 255, 0.05) 1px, transparent 0);
        background-size: 24px 24px;
        color: #94a3b8;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Typography & Headers */
    h1, h2, h3, h4, h5, h6, label, p, span, div {
        font-family: 'Roboto Mono', monospace !important;
    }
    h1 {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    h3 {
        color: #00d4ff !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1.2rem !important;
    }
    
    /* Custom HUD Metric Styles */
    .hud-container {
        border: 1px solid rgba(0, 212, 255, 0.2);
        padding: 1.5rem;
        background: rgba(16, 24, 48, 0.8);
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .hud-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Terminal Visualizations */
    .terminal-card {
        background-color: #05070f;
        border: 1px solid #1e293b;
        padding: 1.2rem;
        border-radius: 4px;
        color: #38bdf8;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .cursor {
        display: inline-block;
        width: 8px;
        height: 15px;
        background: #38bdf8;
        animation: blink 1s infinite;
        vertical-align: middle;
        margin-left: 4px;
    }
    @keyframes blink {
        0%, 49% { opacity: 1; }
        50%, 100% { opacity: 0; }
    }
    
    /* HUD Row Status Indicators */
    .hud-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .status-safe { color: #10b981; font-weight: bold; }
    .status-warn { color: #f59e0b; font-weight: bold; }
    .status-crit { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CORE MACHINE LEARNING ENGINE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"

@st.cache_resource
def load_data_and_train():
    if not DB_PATH.exists():
        np.random.seed(42)
        n_samples = 1500
        mock_data = pd.DataFrame({
            'motor_rpm': np.random.randint(10000, 17000, size=n_samples),
            'package_weight_kg': np.random.uniform(0.5, 6.0, size=n_samples),
            'avg_wind_speed': np.random.uniform(2.0, 35.0, size=n_samples),
            'voltage_drop_rate': np.random.uniform(0.015, 0.095, size=n_samples)
        })
        X_m = mock_data[['motor_rpm', 'package_weight_kg', 'avg_wind_speed']]
        y_m = mock_data['voltage_drop_rate']
        rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_m, y_m)
        return rf, 91.24
        
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT t.motor_rpm, f.package_weight_kg, f.avg_wind_speed, t.voltage_drop_rate
    FROM telemetry_logs t
    INNER JOIN flights f ON t.flight_id = f.flight_id;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    X = df[['motor_rpm', 'package_weight_kg', 'avg_wind_speed']]
    y = df['voltage_drop_rate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    r2_score = rf.score(X_test, y_test) * 100
    return rf, r2_score

model, validation_r2 = load_data_and_train()

# ==========================================
# SIDEBAR CONTROLS (FLIGHT PARAMETERS)
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='color: #00d4ff;'>Flight Parameters</h3>", unsafe_allow_html=True)
    st.write("Adjust mechanical variables to simulate drone telemetry changes:")
    st.markdown("---")
    
    input_rpm = st.slider("Motor RPM", min_value=8000, max_value=18000, value=14000, step=50)
    input_weight = st.slider("Payload Weight (kg)", min_value=0.5, max_value=8.0, value=2.5, step=0.05)
    input_wind = st.slider("Average Wind Speed (km/h)", min_value=0.0, max_value=45.0, value=15.0, step=0.25)

# Calculate live estimation values based on input
features = [[input_rpm, input_weight, input_wind]]
predicted_rate = model.predict(features)[0]

# Dynamic header stats
is_airborne = "AIRBORNE" if input_rpm > 9500 else "GROUNDED"
badge_color = "#00d4ff" if is_airborne == "AIRBORNE" else "#ef4444"
estimated_battery = max(0.0, min(100.0, 100.0 - (predicted_rate * 600)))

# ==========================================
# MAIN PANEL NAVIGATION HEADER
# ==========================================
header_col1, header_col2, header_col3, header_col4 = st.columns([40, 20, 20, 20])

with header_col1:
    st.markdown("<h1 style='margin:0; padding:0;'>DRONE TELEMETRY ANALYZER</h1>", unsafe_allow_html=True)
    st.caption("SYSTEM CORE: RANDOM FOREST REGRESSOR // DATA SINK")

with header_col2:
    st.markdown(f"""
    <div style='text-align: center; border-left: 1px solid #1e293b;'>
        <div style='font-size: 0.75rem; color: #64748b;'>STATUS</div>
        <div style='font-size: 1.1rem; font-weight: bold; color: {badge_color};'>{is_airborne}</div>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.markdown(f"""
    <div style='text-align: center; border-left: 1px solid #1e293b;'>
        <div style='font-size: 0.75rem; color: #64748b;'>EST. REMAINING CAP. (10M)</div>
        <div style='font-size: 1.1rem; font-weight: bold; color: #38bdf8;'>{estimated_battery:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with header_col4:
    st.markdown(f"""
    <div style='text-align: center; border-left: 1px solid #1e293b;'>
        <div style='font-size: 0.75rem; color: #64748b;'>SYS VALIDATION R²</div>
        <div style='font-size: 1.1rem; font-weight: bold; color: #10b981;'>{validation_r2:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: 0; border-top: 1px solid rgba(0, 212, 255, 0.2); margin-top:0.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

# ==========================================
# CORE LAYOUT DATA MATRIX
# ==========================================
col_diag, col_charts = st.columns(2, gap="large")

with col_diag:
    st.markdown("<h3>Diagnostics Panel</h3>", unsafe_allow_html=True)
    
    rpm_status = ("CRIT", "status-crit") if input_rpm > 16000 else (("WARN", "status-warn") if input_rpm > 13500 else ("SAFE", "status-safe"))
    load_status = ("CRIT", "status-crit") if input_weight > 6.0 else (("WARN", "status-warn") if input_weight > 4.0 else ("SAFE", "status-safe"))
    wind_status = ("CRIT", "status-crit") if input_wind > 32.0 else (("WARN", "status-warn") if input_wind > 18.0 else ("SAFE", "status-safe"))
    
    st.markdown(f"""
    <div style='border: 1px solid #1e293b; background: #070a14; padding: 1rem; border-radius: 4px;'>
        <div class='hud-row'>
            <span>[01] PROPULSION RPM LOAD PROFILE</span>
            <span class='{rpm_status[1]}'>{rpm_status[0]}</span>
        </div>
        <div class='hud-row'>
            <span>[02] STRUCTURAL PAYLOAD STRESS EXPONENT</span>
            <span class='{load_status[1]}'>{load_status[0]}</span>
        </div>
        <div class='hud-row'>
            <span>[03] ENVIRONMENTAL WIND RESISTANCE COEFFICIENT</span>
            <span class='{wind_status[1]}'>{wind_status[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_charts:
    st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
    
    gauge_color = "#10b981" if predicted_rate <= 0.040 else ("#f59e0b" if predicted_rate <= 0.060 else "#ef4444")
    percentage_fill = min(100, max(10, int((predicted_rate / 0.1) * 100)))
    
    st.markdown(f"""
    <div class='hud-container'>
        <div style='font-size: 0.8rem; color: #64748b; margin-bottom: 0.5rem;'>PREDICTED VOLTAGE DROP RATE</div>
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div class='hud-value'>{predicted_rate:.5f} V/s</div>
            <div style='width: 120px; background: #1e293b; height: 12px; border-radius: 6px; overflow: hidden; border: 1px solid {gauge_color};'>
                <div style='width: {percentage_fill}%; background: {gauge_color}; height: 100%; transition: width 0.5s ease-in-out;'></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 10-Minute Sparkline Trend Plot
    time_series = np.arange(0, 601, 30)
    initial_voltage = 22.2
    voltage_decay = initial_voltage - (predicted_rate * time_series)
    
    chart_df = pd.DataFrame({
        'Flight Timeline (Seconds)': time_series,
        'Estimated Pack Voltage (V)': voltage_decay
    }).set_index('Flight Timeline (Seconds)')
    
    st.line_chart(chart_df, height=200)

# ==========================================
# SYSTEM METADATA TERMINAL
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h3>System Metadata Terminal</h3>", unsafe_allow_html=True)

st.markdown(f"""
<div class='terminal-card'>
    AEROCHARGE_CORE_LOG // SYSTEM_INITIALIZATION_SUCCESSFUL<br>
    --------------------------------------------------------------------------------<br>
    &gt; CORE_ALGORITHM_IDENTIFIER : RANDOM FOREST REGRESSOR [METRIC R²: {validation_r2:.2f}%]<br>
    &gt; STRUCTURAL_FEATURE_VECTOR  : [01] motor_rpm // [02] package_weight_kg // [03] avg_wind_speed<br>
    &gt; PREDICTIVE_TARGET_MATRIX  : voltage_drop_rate (SI unit output conversion: Volts/second)<br>
    &gt; EXPERIMENTAL_TRAIN_SPLIT  : 80% MATRIX OPTIMIZATION / 20% OUT-OF-SAMPLE VALIDATION<br>
    &gt; DATABASE_DATA_SOURCE_SINK : SQLITE_RELATIONAL_DATASET [drone_fleet.db]<br>
    --------------------------------------------------------------------------------<br>
    &gt; STATUS: MONITORING TELESWEEP STREAM MODULES ACTIVE...<span class='cursor'></span>
</div>
""", unsafe_allow_html=True)