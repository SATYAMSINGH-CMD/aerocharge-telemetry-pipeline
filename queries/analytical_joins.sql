-- ========================================================
-- PHASE 1: WAREHOUSE STRUCTURAL BLUEPRINTS (Leave these here!)
-- ========================================================

CREATE TABLE IF NOT EXISTS drones (
    drone_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    max_payload_kg REAL NOT NULL,
    battery_capacity_mah INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS flights (
    flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    drone_id INTEGER,
    package_weight_kg REAL NOT NULL,
    avg_wind_speed REAL NOT NULL,
    FOREIGN KEY (drone_id) REFERENCES drones (drone_id)
);

CREATE TABLE IF NOT EXISTS telemetry_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_id INTEGER,
    seconds_elapsed INTEGER NOT NULL,
    voltage_drop_rate REAL NOT NULL,
    motor_rpm INTEGER NOT NULL,
    FOREIGN KEY (flight_id) REFERENCES flights (flight_id)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_flight ON telemetry_logs(flight_id);

-- ========================================================
-- ANALYTICAL JOIN QUERY
-- ========================================================

SELECT 
    telemetry_logs.log_id,
    telemetry_logs.flight_id,
    drones.model_name,          
    flights.package_weight_kg,   
    telemetry_logs.motor_rpm,
    flights.avg_wind_speed,
    telemetry_logs.voltage_drop_rate
FROM telemetry_logs
INNER JOIN flights ON telemetry_logs.flight_id = flights.flight_id
INNER JOIN drones ON flights.drone_id = drones.drone_id;
