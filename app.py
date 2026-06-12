import sqlite3
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import uvicorn

# ==========================================
# PHASE 1: SYSTEM PATHS & DATA INGESTION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"

print("🤖 SYSTEM START: Loading data from warehouse...")
connection = sqlite3.connect(DB_PATH)

# Fetching our complete, optimized analytical dataset
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

# Fast training setup using maximum multi-core CPU power
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"✅ MODEL LIVE: R² Accuracy Score: {model.score(X_test, y_test)*100:.2f}%")

# ==========================================
# PHASE 3: FASTAPI ARCHITECTURE SETUPS
# ==========================================
app = FastAPI(
    title="AeroCharge Analytics: Predictive Telemetry Core",
    description="Production-grade ML API forecasting real-time drone battery degradation."
)

# Define the exact format of the incoming JSON data packet (The Request Contract)
class TelemetryDataInput(BaseModel):
    motor_rpm: int
    package_weight_kg: float
    avg_wind_speed: float

@app.post("/predict")
def predict_battery_drain(data: TelemetryDataInput):
    """
    Accepts real-time flight metrics and streams back an absolute battery degradation forecast.
    """
    # Extract values from the incoming web request packet
    input_features = [[data.motor_rpm, data.package_weight_kg, data.avg_wind_speed]]
    
    # Run the values through our active memory model
    prediction = model.predict(input_features)[0]
    
    # Return a structured JSON response packet back over the internet
    return {
        "status": "success",
        "predicted_voltage_drop_rate": round(float(prediction), 4),
        "engineering_unit": "Volts/second"
    }

if __name__ == "__main__":
    # Fire up the live local web server on Port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)