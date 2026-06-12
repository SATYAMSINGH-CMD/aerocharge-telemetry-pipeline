import sqlite3
from pathlib import Path
import random


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"
SQL_PATH = BASE_DIR / "queries" / "analytical_joins.sql"

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

with open(SQL_PATH, "r") as sql_file:
    sql_script = sql_file.read()

connection.executescript(sql_script)
connection.commit()

drones = [
    {
        "model_name": "AeroLift X1",
        "max_payload_kg": 2.5,
        "battery_capacity_mah": 5000,
    },
    {
        "model_name": "CargoHawk Pro",
        "max_payload_kg": 5.0,
        "battery_capacity_mah": 8500,
    },
    {
        "model_name": "SkyMule Heavy",
        "max_payload_kg": 8.0,
        "battery_capacity_mah": 12000,
    },
]

cursor.executemany("""
    INSERT INTO drones (model_name, max_payload_kg, battery_capacity_mah)
    VALUES (:model_name, :max_payload_kg, :battery_capacity_mah)
""", drones)

# ==========================================
# PHASE 2: FLIGHT & TELEMETRY SIMULATION ENGINE
# ==========================================

print("Manufacturing 1,000 flights and 500,000+ physics-aligned logs...")

# We will temporarily store flights and telemetry in lists for high-speed bulk insertion
flight_records = []
telemetry_records = []

# Global counter to keep track of total telemetry row IDs
telemetry_id_counter = 1

for flight_id in range(1, 1001):
    # 1. Randomly assign this flight to one of our 3 drones
    drone_id = random.randint(1, 3)
    
    # Define max payload based on the drone chosen (1=AeroLift: 2.5kg, 2=CargoHawk: 5kg, 3=SkyMule: 8kg)
    if drone_id == 1:
        max_payload = 2.5
    elif drone_id == 2:
        max_payload = 5.0
    else:
        max_payload = 8.0
        
    # 2. Generate physics variables for this flight
    # Package weight is a random decimal up to the drone's capacity
    package_weight_kg = round(random.uniform(0.5, max_payload), 2)
    # Wind speed ranges from dead calm (0 km/h) to a harsh gale (45 km/h)
    avg_wind_speed = round(random.uniform(0.0, 45.0), 2)
    
    # Store this flight record to insert into the database later
# Store this flight record to insert into the database later
    flight_records.append((
        flight_id,
        drone_id,
        package_weight_kg,
        avg_wind_speed
    ))
    
    # 3. SECOND-BY-SECOND TELEMETRY GENERATION
    seconds_elapsed = 0
    battery_level = 100.0
    
    # Calculate the physics stress factors for this flight
    base_rpm = 3000
    weight_stress = package_weight_kg * 250
    wind_stress = avg_wind_speed * 40
    stable_rpm = base_rpm + weight_stress + wind_stress
    
    # Calculate the curved battery depletion rate based on motor stress
    voltage_drop_stress = (stable_rpm / 3000) ** 2 * 0.05
    
    while battery_level > 0:
        # Add slight variations to the motor speed every second
        current_rpm = int(stable_rpm + random.uniform(-50, 50))
        
        # Drain power from the battery tank
        battery_level -= voltage_drop_stress
        if battery_level < 0:
            battery_level = 0.0
            
        # Save this second of telemetry snapshot to our list bucket
        telemetry_records.append((
            telemetry_id_counter,
            flight_id,
            seconds_elapsed,
            round(voltage_drop_stress, 4),
            current_rpm
        ))
        
        # Advance our clocks and line counters forward
        telemetry_id_counter += 1
        seconds_elapsed += 1
        
        # ==========================================
# PHASE 3: BULK INSERTION & FILE COMMIT
# ==========================================
print("Pushing data arrays into SQLite tables...")

# 1. Pushing the 1,000 flights into the flights table
cursor.executemany("""
INSERT INTO flights (flight_id, drone_id, package_weight_kg, avg_wind_speed)
VALUES (?, ?, ?, ?)
""", flight_records)

# 2. Pushing the 500,000+ logs into the telemetry table
cursor.executemany("""
INSERT INTO telemetry_logs (log_id, flight_id, seconds_elapsed, voltage_drop_rate, motor_rpm)
VALUES (?, ?, ?, ?, ?)
""", telemetry_records)

print("Finalizing records and saving changes...")
connection.commit()
connection.close()
