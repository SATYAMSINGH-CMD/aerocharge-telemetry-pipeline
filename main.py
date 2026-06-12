import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Setup our folder paths just like we did in our generator script
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "drone_fleet.db"
# 1. Open and read your SQL analytical query file
SQL_PATH = BASE_DIR / "queries" / "analytical_joins.sql"
with open(SQL_PATH, "r") as sql_file:
    full_sql_script = sql_file.read()

# Split the file text to extract ONLY the final SELECT query at the bottom
# (We separate it from the CREATE TABLE codes up top so Pandas doesn't get confused)
analytical_query = full_sql_script.split("-- PHASE 2: THE ANALYTICAL JOIN QUERY (Our New Code!)")[-1]

# 2. Connect to the data warehouse and pull the data into a Pandas DataFrame
print("Connecting to the data warehouse...")
connection = sqlite3.connect(DB_PATH)

print("Running 3-table INNER JOIN and loading data into a DataFrame...")
df = pd.read_sql_query(analytical_query, connection)

# Close the database link since the data is now safely inside our RAM spreadsheet
connection.close()

# 3. Print a quick snapshot of our data to make sure it worked
print(f"\nPipeline Success! Loaded a spreadsheet matrix with {df.shape[0]} rows and {df.shape[1]} columns.")
print("\nFirst 5 rows of your unified engineering dataset:")
# ==========================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\nGenerating engineering stress plots...")

# Set up the visual look of our graph canvas
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))

# Since plotting 900,000 points will freeze your screen, we take a random 2,000 row sample just for the graph
graph_sample = df.sample(2000, random_state=42)

# Create a scatter plot comparing wind speed to motor RPM, color-coded by drone model
sns.scatterplot(
    data=graph_sample,
    x="avg_wind_speed",
    y="motor_rpm",
    hue="model_name",
    palette="viridis",
    alpha=0.8
)

# Add our engineering titles and axis descriptors
plt.title("Impact of Wind Speed on Drone Motor RPM Dynamics", fontsize=14, fontweight='bold')
plt.xlabel("Average Wind Speed (km/h)", fontsize=12)
plt.ylabel("Motor Operational Speed (RPM)", fontsize=12)

# Save the visualization to our workspace workspace
plot_path = BASE_DIR / "wind_vs_rpm_analysis.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to your project folder as: {plot_path.name}")

# ==========================================
# PHASE 5: BATTERY DEPLETION ANALYSIS
# ==========================================
print("Generating battery degradation regression plot...")

# Clear the canvas from the previous drawing so the two plots don't mix together
plt.clf()
plt.figure(figsize=(10, 6))

# Create a regression plot comparing motor speed to battery voltage drop rate
sns.regplot(
    data=graph_sample,
    x="motor_rpm",
    y="voltage_drop_rate",
    scatter_kws={'alpha': 0.3, 'color': '#2a9d8f'},
    line_kws={'color': '#e76f51', 'linewidth': 3}
)

# Add our labels and engineering context
plt.title("Correlation: Motor Operational Velocity vs. Battery Voltage Depletion Rate", fontsize=14, fontweight='bold')
plt.xlabel("Motor Operational Speed (RPM)", fontsize=12)
plt.ylabel("Voltage Drop Rate (Volts/sec)", fontsize=12)

# Save our second insight plot
battery_plot_path = BASE_DIR / "rpm_vs_battery_depletion.png"
plt.savefig(battery_plot_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to your project folder as: {battery_plot_path.name}")

# ==========================================
# PHASE 6: MACHINE LEARNING PREDICTIVE MODEL
# ==========================================
print("\nInitiating Machine Learning Layer...")

# 1. Separate our physical inputs (X) from our target output (y)
X = df[['motor_rpm', 'package_weight_kg', 'avg_wind_speed']]
y = df['voltage_drop_rate']

# 2. Split the dataset into 80% Training textbook and 20% Unseen Final Exam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data successfully split! Training set: {X_train.shape[0]} rows. Testing set: {X_test.shape[0]} rows.")

# 3. Instantiate the Random Forest algorithm and train it
print("Training the Random Forest Regressor model (this may take a few seconds)...")
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete!")

# 4. Evaluate the model accuracy on the unseen final exam data
r2_score = model.score(X_test, y_test)
print(f"\n==========================================")
print(f"🎉 MACHINE LEARNING MODEL RESULTS 🎉")
print(f"Model R² Accuracy Score: {r2_score * 100:.2f}%")
print(f"==========================================")