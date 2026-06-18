import random
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"
SQL_PATH = BASE_DIR / "queries" / "analytical_joins.sql"

RANDOM_SEED = 42
FLIGHT_COUNT = 1000

DRONES = [
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


def load_schema_sql() -> str:
    sql_script = SQL_PATH.read_text(encoding="utf-8")
    return sql_script.split("-- ANALYTICAL JOIN QUERY", 1)[0]


def reset_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS telemetry_logs;
        DROP TABLE IF EXISTS flights;
        DROP TABLE IF EXISTS drones;
        """
    )
    connection.executescript(load_schema_sql())


def build_simulated_records() -> tuple[list[tuple], list[tuple]]:
    flight_records = []
    telemetry_records = []
    telemetry_id_counter = 1

    for flight_id in range(1, FLIGHT_COUNT + 1):
        drone_id = random.randint(1, len(DRONES))
        max_payload = DRONES[drone_id - 1]["max_payload_kg"]

        package_weight_kg = round(random.uniform(0.5, max_payload), 2)
        avg_wind_speed = round(random.uniform(0.0, 45.0), 2)

        flight_records.append(
            (
                flight_id,
                drone_id,
                package_weight_kg,
                avg_wind_speed,
            )
        )

        base_rpm = 3000
        weight_stress = package_weight_kg * 250
        wind_stress = avg_wind_speed * 40
        stable_rpm = base_rpm + weight_stress + wind_stress

        # This is a transparent synthetic target for a learning project.
        voltage_drop_rate = (stable_rpm / 3000) ** 2 * 0.05

        seconds_elapsed = 0
        battery_level = 100.0

        while battery_level > 0:
            current_rpm = int(stable_rpm + random.uniform(-50, 50))
            battery_level = max(0.0, battery_level - voltage_drop_rate)

            telemetry_records.append(
                (
                    telemetry_id_counter,
                    flight_id,
                    seconds_elapsed,
                    round(voltage_drop_rate, 4),
                    current_rpm,
                )
            )

            telemetry_id_counter += 1
            seconds_elapsed += 1

    return flight_records, telemetry_records


def main() -> None:
    random.seed(RANDOM_SEED)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Creating fresh SQLite warehouse...")
    with sqlite3.connect(DB_PATH) as connection:
        reset_database(connection)
        cursor = connection.cursor()

        cursor.executemany(
            """
            INSERT INTO drones (model_name, max_payload_kg, battery_capacity_mah)
            VALUES (:model_name, :max_payload_kg, :battery_capacity_mah)
            """,
            DRONES,
        )

        print(f"Simulating {FLIGHT_COUNT:,} flights and second-by-second telemetry logs...")
        flight_records, telemetry_records = build_simulated_records()

        cursor.executemany(
            """
            INSERT INTO flights (flight_id, drone_id, package_weight_kg, avg_wind_speed)
            VALUES (?, ?, ?, ?)
            """,
            flight_records,
        )

        cursor.executemany(
            """
            INSERT INTO telemetry_logs (log_id, flight_id, seconds_elapsed, voltage_drop_rate, motor_rpm)
            VALUES (?, ?, ?, ?, ?)
            """,
            telemetry_records,
        )

        connection.commit()

    print(f"Inserted {len(DRONES):,} drones.")
    print(f"Inserted {len(flight_records):,} flights.")
    print(f"Inserted {len(telemetry_records):,} telemetry logs.")
    print(f"Database saved to {DB_PATH}.")


if __name__ == "__main__":
    main()
