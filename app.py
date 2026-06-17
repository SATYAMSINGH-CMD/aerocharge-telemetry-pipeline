import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page configuration optimized for a clean, professional dashboard view
st.set_page_config(page_title="Drone Telemetry Analyzer", layout="wide")

# ==========================================
# STYLING SHEET: CLEAN LIGHT THEME
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
    
    /* Remove standard Streamlit structural distractions */
    span[data-testid="stSidebarCollapseButton"], 
    div[data-testid="collapsedControl"],
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #2d2d2d;
        font-family: 'Inter', sans-serif;
    }
    
    /* Focus typography strictly on industry-standard slates */
    label, [data-testid="stWidgetLabel"] p {
        color: #1a2238 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    h1, h2, h3, h4, .mono-text {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Top Header Banner Component */
    .hero-header {
        background-color: #1a2238;
        color: #ffffff;
        padding: 3.5rem 2rem;
        text-align: center;
        border-bottom: 4px solid #c8a84b;
        margin: -6rem -4rem 2rem -4rem;
    }
    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #a0aabf;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Metric Display Cards */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: #1a2238;
    }
    
    /* Plain horizontal metrics progress bars */
    .bar-wrapper {
        width: 100%;
        height: 6px;
        background-color: #e9ecef;
        margin-top: 0.5rem;
        border-radius: 3px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background-color: #c8a84b;
        transition: width 0.4s ease;
    }
    
    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #e9ecef;
        font-size: 0.85rem;
        color: #2d2d2d;
    }
    .text-safe { color: #4a7c59; font-weight: 600; }
    .text-warn { color: #a07c3a; font-weight: 600; }
    .text-crit { color: #8f3f3f; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & MODEL TRAINING
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"

@st.cache_resource
def train_telemetry_models():
    """Loads historical logs and trains estimators for independent subsystems."""
    np.random.seed(42)
    n_samples = 1000
    
    # Check if real historical data rows exist inside the SQLite data layer
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT t.motor_rpm, f.package_weight_kg, f.avg_wind_speed, t.voltage_drop_rate 
            FROM telemetry_logs t 
            INNER JOIN flights f ON t.flight_id = f.flight_id;
        """, conn)
        conn.close()
        rpm = df['motor_rpm'].values
        weight = df['package_weight_kg'].values
        wind = df['avg_wind_speed'].values
        v_drop = df['voltage_drop_rate'].values
    else:
        # Fallback simulator using physics-approximated bounds
        rpm = np.random.randint(10000, 17000, size=n_samples)
        weight = np.random.uniform(0.5, 6.0, size=n_samples)
        wind = np.random.uniform(2.0, 35.0, size=n_samples)
        v_drop = np.random.uniform(0.001, 0.012, size=n_samples)

    # Simplified engineering relationships for secondary mechanical factors
    temp = 25.0 + (rpm * 0.003) + (weight * 4.5) + np.random.normal(0, 2, size=len(rpm))
    vib = 0.2 + (wind * 0.04) + (rpm * 0.00005) + np.random.normal(0, 0.05, size=len(rpm))
    drag = (wind ** 2) * 0.008 * (1.0 + (weight * 0.05)) + np.random.normal(0, 0.1, size=len(rpm))

    data_payload = pd.DataFrame({
        'rpm': rpm, 'weight': weight, 'wind': wind,
        'v_drop': v_drop, 'temp': temp, 'vib': vib, 'drag': drag
    })
    
    features = ['rpm', 'weight', 'wind']
    models_dict = {}
    
    # Train independent sub-models
    for target in ['v_drop', 'temp', 'vib', 'drag']:
        X = data_payload[features]
        y = data_payload[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models_dict[target] = rf
        
    return models_dict

models = train_telemetry_models()

# ==========================================
# DASHBOARD HEADER BANNER
# ==========================================
st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>DRONE TELEMETRY ANALYZER</div>
        <div class='hero-subtitle'>Predictive flight analytics engine evaluating multi-subsystem airframe performance curves.</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# THREE COLUMN MAIN LAYOUT GRID
# ==========================================
col_input, col_metrics, col_hud = st.columns([25, 45, 30], gap="large")

# --- PANEL 1: INPUTS (LEFT) ---
with col_input:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Inputs</h4>", unsafe_allow_html=True)
    input_rpm = st.slider("Motor RPM", 8000, 18000, 14000, 100)
    input_weight = st.slider("Payload Weight (kg)", 0.5, 8.0, 2.5, 0.05)
    input_wind = st.slider("Average Wind Speed (km/h)", 0.0, 45.0, 15.0, 0.25)

# Calculate simultaneous inference outputs
eval_vector = [[input_rpm, input_weight, input_wind]]
raw_v_pred = models['v_drop'].predict(eval_vector)[0]

# Standardize output scale to real millivolts/sec bounds if data spans are over-expanded
pred_v = raw_v_pred * 0.05 if raw_v_pred > 0.02 else raw_v_pred
pred_t = models['temp'].predict(eval_vector)[0]
pred_vi = models['vib'].predict(eval_vector)[0]
pred_d = models['drag'].predict(eval_vector)[0]

# --- PANEL 2: SUBSYSTEM DIAGNOSTICS (CENTER) ---
with col_metrics:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>Subsystem Diagnostics</h4>", unsafe_allow_html=True)
    
    # Block 1: Battery Drop Rate
    fill_v = min(100, max(5, int((pred_v / 0.015) * 100)))
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Battery Voltage Drop Rate</div>
            <div class='metric-value'>{pred_v:.5f} <span style='font-size: 14px; color:#6c757d;'>V/s</span></div>
            <div class='bar-wrapper'><div class='bar-fill' style='width: {fill_v}%;'></div></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Block 2: Motor Temperature
    fill_t = min(100, max(5, int((pred_t / 100) * 100)))
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Motor Temperature</div>
            <div class='metric-value'>{pred_t:.1f} <span style='font-size: 14px; color:#6c757d;'>°C</span></div>
            <div class='bar-wrapper'><div class='bar-fill' style='width: {fill_t}%;'></div></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Block 3: Vibration Level
    fill_vi = min(100, max(5, int((pred_vi / 3.0) * 100)))
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Vibration Amplitude</div>
            <div class='metric-value'>{pred_vi:.2f} <span style='font-size: 14px; color:#6c757d;'>g-force</span></div>
            <div class='bar-wrapper'><div class='bar-fill' style='width: {fill_vi}%;'></div></div>
        </div>
    """, unsafe_allow_html=True)

# --- PANEL 3: STATUS & FORECASTS (RIGHT) ---
with col_hud:
    st.markdown("<h4 style='color: #1a2238; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem;'>System Status</h4>", unsafe_allow_html=True)
    
    t_status = ("Critical", "text-crit") if pred_t > 80 else (("Caution", "text-warn") if pred_t > 65 else ("Nominal", "text-safe"))
    vi_status = ("Critical", "text-crit") if pred_vi > 2.0 else (("Caution", "text-warn") if pred_vi > 1.2 else ("Nominal", "text-safe"))
    d_status = ("High", "text-crit") if pred_d > 10.0 else (("Moderate", "text-warn") if pred_d > 4.5 else ("Low", "text-safe"))
    
    st.markdown(f"""
        <div style='background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 1rem; border-radius: 4px;'>
            <div class='status-row'><span>Thermal Integrity</span><span class='{t_status[1]}'>{t_status[0]}</span></div>
            <div class='status-row'><span>Structural Vibration</span><span class='{vi_status[1]}'>{vi_status[0]}</span></div>
            <div class='status-row'><span>Aerodynamic Drag ({pred_d:.1f} N)</span><span class='{d_status[1]}'>{d_status[0]}</span></div>
        </div>
    """, unsafe_allow_html=True)
    
    # 10-Minute Trend Line Plot capped at absolute discharge floors
    st.markdown("<br><div style='font-size:0.75rem; font-weight:600; color:#6c757d; text-transform:uppercase;'>10-Min Voltage Decay Forecast</div>", unsafe_allow_html=True)
    time_steps = np.arange(0, 601, 30)
    
    # Clip values cleanly at floor safety boundaries (18.0V for a 6S Pack)
    voltage_line = np.maximum(18.0, 22.20 - (pred_v * time_steps))
    
    chart_data = pd.DataFrame({
        'Timeline (s)': time_steps, 
        'Pack Voltage (V)': voltage_line
    }).set_index('Timeline (s)')
    st.line_chart(chart_data, height=160)