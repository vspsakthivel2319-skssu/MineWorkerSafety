import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("source_dataset.csv")

print("Dataset loaded successfully!")
print("Rows:", len(data))

# Raw sensor features
features = [
    "Temperature (°C)",
    "Humidity (%)",
    "CO (ppm)",
    "CO2 (ppm)",
    "H2S (ppm)",
    "NH3 (ppm)",
    "Heart_Rate (bpm)",
    "PR_Interval (ms)",
    "QT_Interval (ms)",
    "ST_Interval (ms)",
    "SpO2 (%)",
    "Motion (0/1)"
]

target = "Overall_Status"

# Select input and target
X = data[features]
y = data[target]

# Save feature names
with open("feature_columns.pkl", "wb") as file:
    pickle.dump(features, file)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Random Forest training completed!")
print("Accuracy:", accuracy)

# Save trained model
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
print("File: random_forest_model.pkl")