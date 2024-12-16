import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic dataset
data = {
    "Vessel_Type": np.random.choice(["Container", "Tanker", "Bulk Carrier", "RoRo"], size=200),
    "Route_Distance": np.random.randint(100, 4000, size=200),  # Distance in nautical miles
    "Cargo_Weight": np.random.randint(5000, 100000, size=200),  # Weight in tons
    "Departure_Time": pd.date_range(start="2024-01-01", periods=200, freq="h").strftime("%Y-%m-%d %H:%M:%S"),  # 'h' for hourly frequency
    "Weather_Conditions": np.random.choice(["Clear", "Stormy", "Windy", "Foggy"], size=200),
    "Delay": np.random.choice([0, 1], size=200, p=[0.7, 0.3])  # 70% on-time, 30% delayed
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("maritime_data.csv", index=False)

# Print a preview of the dataset
print("Dataset saved as 'maritime_data.csv'")
print(df.head())
