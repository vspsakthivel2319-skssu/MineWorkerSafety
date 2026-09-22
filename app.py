import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature names
with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# Page title
st.set_page_config(
    page_title="Mine Worker Safety Monitor",
    page_icon="⛑️"
)

st.title("⛑️ Mine Worker Safety Monitoring System")
st.write("Random Forest Machine Learning Prediction")

st.header("Enter Worker Sensor Data")

# Input fields
temperature = st.number_input(
    "Temperature (°C)",
    value=30.0
)

humidity = st.number_input(
    "Humidity (%)",
    value=60.0
)

co = st.number_input(
    "CO (ppm)",
    value=10.0
)

co2 = st.number_input(
    "CO2 (ppm)",
    value=400.0
)

h2s = st.number_input(
    "H2S (ppm)",
    value=2.0
)

nh3 = st.number_input(
    "NH3 (ppm)",
    value=10.0
)

heart_rate = st.number_input(
    "Heart Rate (bpm)",
    value=80.0
)

pr_interval = st.number_input(
    "PR Interval (ms)",
    value=160.0
)

qt_interval = st.number_input(
    "QT Interval (ms)",
    value=400.0
)

st_interval = st.number_input(
    "ST Interval (ms)",
    value=100.0
)

spo2 = st.number_input(
    "SpO2 (%)",
    value=98.0
)

motion = st.selectbox(
    "Motion",
    [0, 1]
)

gas_status = st.selectbox(
    "Gas Status",
    ["Normal", "Warning", "Critical"]
)

heart_status = st.selectbox(
    "Heart Status",
    ["Normal", "Warning", "Critical"]
)

# Prediction button
if st.button("🔍 Predict Safety Status"):

    # Create input data
    input_data = pd.DataFrame({
        "Temperature (°C)": [temperature],
        "Humidity (%)": [humidity],
        "CO (ppm)": [co],
        "CO2 (ppm)": [co2],
        "H2S (ppm)": [h2s],
        "NH3 (ppm)": [nh3],
        "Heart_Rate (bpm)": [heart_rate],
        "PR_Interval (ms)": [pr_interval],
        "QT_Interval (ms)": [qt_interval],
        "ST_Interval (ms)": [st_interval],
        "SpO2 (%)": [spo2],
        "Motion (0/1)": [motion],
        "Gas_Status": [gas_status],
        "Heart_Status": [heart_status]
    })

    # Convert text columns to numbers
    input_data = pd.get_dummies(input_data)

    # Make sure input has exactly the same columns as training data
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Safe":
        st.success(f"🟢 Worker Status: {prediction}")

    elif prediction == "Warning":
        st.warning(f"🟡 Worker Status: {prediction}")

    else:
        st.error(f"🔴 Worker Status: {prediction}")

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    probability_table = pd.DataFrame({
        "Status": model.classes_,
        "Probability": probabilities
    })

    st.write("Prediction probabilities:")
    st.dataframe(probability_table)