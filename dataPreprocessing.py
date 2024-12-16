import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("maritime_data.csv")

# Convert categorical features to numeric using one-hot encoding
categorical_features = ["Vessel_Type", "Weather_Conditions"]
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features])

# Create a new DataFrame for encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Combine encoded features with the rest of the dataset
df = pd.concat([df, encoded_df], axis=1)

# Drop original categorical columns
df = df.drop(columns=categorical_features + ["Departure_Time"])

# Scale numerical features
scaler = StandardScaler()
df[["Route_Distance", "Cargo_Weight"]] = scaler.fit_transform(df[["Route_Distance", "Cargo_Weight"]])

# Split into features and labels
X = df.drop(columns=["Delay"])  # Features
y = df["Delay"]  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessed data saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
