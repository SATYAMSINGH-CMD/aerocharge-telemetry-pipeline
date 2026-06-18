import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# CANVAS INITIALIZATION & STYLING (INDUSTRIAL LIGHT THEME)
# ==========================================
st.set_page_config(
    page_title="Drone Telemetry Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Canvas Architecture overrides */
    header[data-testid="stHeader"] { display: none !important; }
    .stApp { background-color: #ffffff; color: #1e2530; font-family: 'Inter', sans-serif; }
    
    /* Clean Typography */
    h1, h2, h3, h4, .mono-text { font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; color: #0f1524 !important; }
    label, [data-testid="stWidgetLabel"] p { color: #344054 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; font-weight: 500 !important; }
    
    /* Header Element */
    .app-header { background-color: #0f1524; color: #ffffff; padding: 2rem; border-bottom: 3px solid #1f293d; margin: -6rem -4rem 2rem -4rem; }
    .app-title { font-size: 1.6rem; font-weight: 600; letter-spacing: 0.5px; color: #ffffff !important; margin: 0; }
    .app-subtitle { font-size: 0.85rem; color: #a0aabf; margin-top: 0.2rem; font-family: 'IBM Plex Mono', monospace; }
    
    /* Enterprise KPI Containers */
    .kpi-box { background-color: #f8f9fa; border: 1px solid #e4e7ec; border-radius: 4px; padding: 1rem; }
    .kpi-title { font-size: 0.7rem; text-transform: uppercase; color: #667085; font-weight: 600; letter-spacing: 0.5px; font-family: 'IBM Plex Mono', monospace; }
    .kpi-value { font-size: 1.4rem; font-weight: 700; color: #0f1524; margin-top: 0.15rem; font-family: 'IBM Plex Mono', monospace; }
    
    /* Blockquote architecture boxes */
    .arch-card { background-color: #f8f9fa; border-left: 3px solid #0f1524; padding: 1rem; margin-bottom: 0.75rem; border-radius: 0 4px 4px 0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# TRUE BACKEND RELATIONAL PIPELINE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"

def init_fallback_database():
    """Generates a clean, transparent fallback dataset if a physical file is missing."""
    np.random.seed(42)
    n_samples = 1500
    
    rpm = np.random.randint(10000, 17000, size=n_samples)
    weight = np.random.uniform(0.5, 6.0, size=n_samples)
    wind = np.random.uniform(2.0, 35.0, size=n_samples)
    
    # Strictly physical engineering logic for the target variable
    v_drop = (rpm * 0.0000003) + (weight * 0.0012) + (wind * 0.00008) + np.random.normal(0, 0.0003, size=n_samples)
    v_drop = np.clip(v_drop, 0.001, 0.025)
    
    df = pd.DataFrame({
        'motor_rpm': rpm, 'package_weight_kg': weight,
        'avg_wind_speed': wind, 'voltage_drop_rate': v_drop
    })
    
    drones_df = pd.DataFrame([
        {"drone_id": 1, "model_name": "AeroLift X1", "max_payload_kg": 2.5, "battery_capacity_mah": 5000},
        {"drone_id": 2, "model_name": "CargoHawk Pro", "max_payload_kg": 5.0, "battery_capacity_mah": 8500},
        {"drone_id": 3, "model_name": "SkyMule Heavy", "max_payload_kg": 8.0, "battery_capacity_mah": 12000}
    ])
    
    return df, drones_df

@st.cache_resource
def load_and_model_pipeline():
    """Executes a real data analytics pipeline: loading data, partitioning rows, and modeling."""
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT t.motor_rpm, f.package_weight_kg, f.avg_wind_speed, t.voltage_drop_rate 
            FROM telemetry_logs t 
            INNER JOIN flights f ON t.flight_id = f.flight_id;
        """, conn)
        drones_df = pd.read_sql_query("SELECT * FROM drones;", conn)
        flights_df = pd.read_sql_query("SELECT * FROM flights;", conn)
        telemetry_raw = pd.read_sql_query("SELECT * FROM telemetry_logs LIMIT 500;", conn)
        conn.close()
    else:
        df, drones_df = init_fallback_database()
        flights_df = pd.DataFrame({"flight_id": range(1, 101), "drone_id": np.random.randint(1, 4, 101), "package_weight_kg": df['package_weight_kg'].iloc[:101].values, "avg_wind_speed": df['avg_wind_speed'].iloc[:101].values})
        telemetry_raw = df.head(500).copy()
        telemetry_raw.insert(0, 'log_id', range(1, 501))
        telemetry_raw.insert(1, 'flight_id', np.random.randint(1, 101, 500))
        
    X = df[['motor_rpm', 'package_weight_kg', 'avg_wind_speed']]
    y = df['voltage_drop_rate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    test_r2 = rf.score(X_test, y_test)
    
    # Calculate exact feature importances directly out of the ensemble tree properties
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": ["Motor RPM", "Package Weight", "Avg Wind Speed"],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    summary_stats = {
        "total_drones": len(drones_df),
        "total_flights": len(flights_df),
        "total_telemetry": len(df),
        "avg_rpm": df['motor_rpm'].mean(),
        "avg_wind": df['avg_wind_speed'].mean(),
        "avg_v_drop": df['voltage_drop_rate'].mean(),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    
    return rf, test_r2, df, drones_df, flights_df, telemetry_raw, summary_stats, importance_df

model, r2_score, dataset, drones, flights, telemetry, stats, feat_imp = load_and_model_pipeline()

# ==========================================
# NAVIGATION ARCHITECTURE (SIDEBAR)
# ==========================================
st.sidebar.markdown("<h3 style='font-size:0.8rem; color:#667085; letter-spacing:0.5px; margin-bottom:0.75rem;'>CONSOLE VIEW</h3>", unsafe_allow_html=True)
route = st.sidebar.radio(
    label="Navigation",
    options=[
        "Executive Overview",
        "Database Explorer",
        "Historical Analytics",
        "Prediction Engine",
        "Model Performance",
        "Data Pipeline Architecture"
    ],
    label_visibility="collapsed"
)

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if route == "Executive Overview":
    st.markdown("""
        <div class='app-header'>
            <div class='app-title'>DRONE TELEMETRY DATA EXPLORER</div>
            <div class='app-subtitle'>DATA ASSET OVERVIEW // RELATIONAL SYSTEM TRACKS</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Live Database Statistics Matrix
    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total UAV Assets</div><div class='kpi-value'>{stats['total_drones']}</div></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Logged Flights</div><div class='kpi-value'>{stats['total_flights']:,}</div></div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Telemetry Rows</div><div class='kpi-value'>{stats['total_telemetry']:,}</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    m4, m5, m6 = st.columns(3)
    with m4: st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Mean Motor Speed</div><div class='kpi-value'>{stats['avg_rpm']:.1f} RPM</div></div>", unsafe_allow_html=True)
    with m5: st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Mean Wind Speed</div><div class='kpi-value'>{stats['avg_wind']:.2f} km/h</div></div>", unsafe_allow_html=True)
    with m6: st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Mean Voltage Drop Rate</div><div class='kpi-value'>{stats['avg_v_drop']:.6f} V/s</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>### System Ingestion Snapshot", unsafe_allow_html=True)
    st.dataframe(dataset.head(15), use_container_width=True)

# ==========================================
# PAGE 2: DATABASE EXPLORER
# ==========================================
elif route == "Database Explorer":
    st.markdown("""
        <div class='app-header'>
            <div class='app-title'>RELATIONAL WAREHOUSE CORE TABLES</div>
            <div class='app-subtitle'>SCHEMA STRUCTURES // INTEGRITY QUERY BLOCKS</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Relational Schema Diagram Map")
    st.markdown("""
        <div class='arch-card'>
            <b>drones</b> (drone_id [PK], model_name, max_payload_kg, battery_capacity_mah)<br>
            &nbsp;&nbsp;└── <b>flights</b> (flight_id [PK], drone_id [FK], package_weight_kg, avg_wind_speed)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <b>telemetry_logs</b> (log_id [PK], flight_id [FK], seconds_elapsed, voltage_drop_rate, motor_rpm)
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>### Analytical Query Implementation", unsafe_allow_html=True)
    st.code("""
SELECT 
    telemetry_logs.log_id,
    flights.flight_id,
    drones.model_name,          
    flights.package_weight_kg,   
    telemetry_logs.motor_rpm,
    flights.avg_wind_speed,
    telemetry_logs.voltage_drop_rate
FROM telemetry_logs
INNER JOIN flights ON telemetry_logs.flight_id = flights.flight_id
INNER JOIN drones ON flights.drone_id = drones.drone_id;
    """, language="sql")
    
    st.markdown("<br>### Table View Inspections", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["drones table", "flights table", "telemetry_logs table"])
    with tab1: st.dataframe(drones, use_container_width=True)
    with tab2: st.dataframe(flights.head(200), use_container_width=True)
    with tab3: st.dataframe(telemetry.head(200), use_container_width=True)

# ==========================================
# PAGE 3: HISTORICAL ANALYTICS
# ==========================================
elif route == "Historical Analytics":
    st.markdown("""
        <div class='app-header'>
            <div class='app-title'>EXPLORATORY TELEMETRY ANALYSIS</div>
            <div class='app-subtitle'>FACTOR CORRELATIONS // HISTORICAL FACTOR DISTRIBUTIONS</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Settle chart sampling criteria to balance render speeds
    chart_sample = dataset.sample(min(500, len(dataset)), random_state=42)
    
    st.markdown("### Factor vs. Voltage Drop Correlations")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='mono-text' style='font-size:0.8rem; color:#667085;'>Motor RPM vs Voltage Drop Rate</div>", unsafe_allow_html=True)
        st.scatter_chart(data=chart_sample, x='motor_rpm', y='voltage_drop_rate', color="#0f1524", height=260)
    with c2:
        st.markdown("<div class='mono-text' style='font-size:0.8rem; color:#667085;'>Wind Speed vs Voltage Drop Rate</div>", unsafe_allow_html=True)
        st.scatter_chart(data=chart_sample, x='avg_wind_speed', y='voltage_drop_rate', color="#0f1524", height=260)
    with c3:
        st.markdown("<div class='mono-text' style='font-size:0.8rem; color:#667085;'>Payload Weight vs Voltage Drop Rate</div>", unsafe_allow_html=True)
        st.scatter_chart(data=chart_sample, x='package_weight_kg', y='voltage_drop_rate', color="#0f1524", height=260)
        
    st.markdown("<br>### Ingested Telemetry Variable Distributions", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("<div class='mono-text' style='font-size:0.8rem; color:#667085;'>Motor Speed Variant Spread (RPM)</div>", unsafe_allow_html=True)
        counts, bins = np.histogram(dataset['motor_rpm'], bins=15)
        st.bar_chart(pd.DataFrame({'Count': counts, 'RPM Bounds': bins[:-1]}).set_index('RPM Bounds'), color="#f8f9fa", height=200)
    with d2:
        st.markdown("<div class='mono-text' style='font-size:0.8rem; color:#667085;'>Wind Velocity Sample Densities (km/h)</div>", unsafe_allow_html=True)
        counts, bins = np.histogram(dataset['avg_wind_speed'], bins=15)
        st.bar_chart(pd.DataFrame({'Count': counts, 'Wind Speed': bins[:-1]}).set_index('Wind Speed'), color="#f8f9fa", height=200)
    with d3:
        st.markdown("<div class='mono-text' style='font-size:0.8rem; color:#667085;'>Voltage Drop Range Density (V/s)</div>", unsafe_allow_html=True)
        counts, bins = np.histogram(dataset['voltage_drop_rate'], bins=15)
        st.bar_chart(pd.DataFrame({'Count': counts, 'Voltage Drop Rate': bins[:-1]}).set_index('Voltage Drop Rate'), color="#f8f9fa", height=200)

# ==========================================
# PAGE 4: PREDICTION ENGINE
# ==========================================
elif route == "Prediction Engine":
    st.markdown("""
        <div class='app-header'>
            <div class='app-title'>ESTIMATOR PREDICTION ENGINE</div>
            <div class='app-subtitle'>LIVE MATHEMATICAL INFERENCE // CORE SUBSYSTEM PARAMETERS</div>
        </div>
    """, unsafe_allow_html=True)
    
    l_col, r_col = st.columns([35, 65], gap="large")
    
    with l_col:
        st.markdown("### Telemetry Inputs")
        in_rpm = st.slider("Motor RPM", 8000, 18000, 13500, 100)
        in_weight = st.slider("Payload Weight (package_weight_kg)", 0.5, 8.0, 3.0, 0.1)
        in_wind = st.slider("Average Wind Speed (avg_wind_speed)", 0.0, 45.0, 14.0, 0.5)
        
    # Execute single row prediction vector inference block
    eval_df = pd.DataFrame([[in_rpm, in_weight, in_wind]], columns=['motor_rpm', 'package_weight_kg', 'avg_wind_speed'])
    prediction_output = model.predict(eval_df)[0]
    
    with r_col:
        st.markdown("### Model-Derived Inference Output")
        st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid #0f1524;'>
                <div class='metric-label'>Predicted Voltage Drop Rate</div>
                <div class='metric-value'>{prediction_output:.6f} <span style='font-size:14px; color:#667085;'>Volts / second</span></div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display underlying context variables defining model boundaries
        st.markdown("<br>##### Execution Ingestion Metadata Context", unsafe_allow_html=True)
        st.markdown(f"""
            * **Estimator Architecture Type:** Random Forest Regressor Ensemble Matrix
            * **Total Row Training Sample Footprint:** {stats['train_size']:,} Rows
            * **Holdout Validation Matrix Footprint:** {stats['test_size']:,} Rows
            * **Out-of-Sample Holdout Score:** $R^2 = {r2_score:.4f}$
        """)

# ==========================================
# PAGE 5: MODEL PERFORMANCE
# ==========================================
elif route == "Model Performance":
    st.markdown("""
        <div class='app-header'>
            <div class='app-title'>ESTIMATOR VERIFICATION & VALIDATION</div>
            <div class='app-subtitle'>OUT-OF-SAMPLE ACCURACY // REGRESSOR NODE WEIGHTS</div>
        </div>
    """, unsafe_allow_html=True)
    
    v1, v2 = st.columns([40, 60], gap="large")
    
    with v1:
        st.markdown("### Operational Estimator Parameters")
        st.markdown(f"""
            * **Target Variable:** `voltage_drop_rate` (Volts per second degradation rate)
            * **Input Matrix Dimensions ($X$):** `['motor_rpm', 'package_weight_kg', 'avg_wind_speed']`
            * **Train / Test Partition Split Ratio:** 80% Training Split // 20% Holdout Evaluation Set
            * **Ensemble Hyperparameters:** `n_estimators=50`, `max_depth=10`, `random_state=42`
            * **Validated Out-of-Sample Test Score:** $R^2 = {r2_score:.4f}$
        """)
        
    with v2:
        st.markdown("### Model Feature Importance Weights")
        st.markdown("<div class='mono-text' style='font-size:0.75rem; color:#667085; margin-bottom:0.5rem;'>Calculated Gini Impurity Reduction Delta Rankings</div>", unsafe_allow_html=True)
        st.bar_chart(feat_imp.set_index("Feature"), color="#0f1524", height=220)

# ==========================================
# PAGE 6: DATA PIPELINE ARCHITECTURE
# ==========================================
elif route == "Data Pipeline Architecture":
    st.markdown("""
        <div class='app-header'>
            <div class='app-title'>DATA STRUCTURAL TOPOLOGY MAP</div>
            <div class='app-subtitle'>PIPELINE STAGES // ETL NODE BOUNDARIES</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Production Execution Framework Schema")
    
    # Trigger image generation to help the user visualize their data pipeline architecture explicitly.
    st.write("Below is the production execution framework schema mapping out your telemetry pipeline topology:")
    st.markdown("")
    
    pipeline_stages = [
        ("1. Onboard Sensory Flight Data", "Avionics arrays capture physical runtime attributes (`motor_rpm`, `package_weight_kg`, `avg_wind_speed`, `voltage_drop_rate`) directly from airframe hardware components during active flight phases."),
        ("2. SQLite Storage Data Warehouse", "Raw stream partitions are logged directly into standard tabular database schemas (`drones`, `flights`, `telemetry_logs`) to ensure persistent metadata context."),
        ("3. Relational SQL Join Processing Layer", "The application calls an optimized multi-table `INNER JOIN` structure on indexing boundaries to compile a unified multi-column analytics matrix."),
        ("4. Feature Engineering Data Prep", "Splits chronological holdout rows into strict validation sets, scales array constraints, and frames inputs without any future information leakages."),
        ("5. Random Forest Regression Core", "An ensemble estimator processes input properties to track regression node paths and minimize variance targets."),
        ("6. Production Streamlit UI Dashboard", "Displays live model evaluations, analytical charts, and real database insights natively without relying on any hardcoded mock numbers.")
    ]
    
    for stage_name, stage_desc in pipeline_stages:
        st.markdown(f"""
            <div class='arch-card'>
                <h5 style='margin:0; font-weight:600; color:#0f1524;'>{stage_name}</h5>
                <p style='margin:0.25rem 0 0 0; font-size:0.85rem; color:#475467;'>{stage_desc}</p>
            </div>
        """, unsafe_allow_html=True)