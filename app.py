# app.py - Real-Time IoT Sensor Anomaly Detection

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Real-Time IoT Anomaly Detection", layout="wide")
st.title("📡 Real-Time IoT Temperature Sensor - Anomaly Detection")
st.markdown("Simulates live sensor readings and detects anomalies using ML in real time.")

# -----------------------------
# Initialize or load data
# -----------------------------
if "df" not in st.session_state:
    # Start with empty DataFrame
    st.session_state.df = pd.DataFrame(columns=["Temperature", "Temp_Roll", "Anomaly", "Anomaly_Color"])
    st.session_state.window = 10  # rolling window for average

# Sidebar settings
st.sidebar.subheader("Simulation Settings")
contamination = st.sidebar.slider("Anomaly Contamination Rate", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
pause_time = st.sidebar.slider("Update Interval (seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# -----------------------------
# Simulate new sensor reading
# -----------------------------
def generate_reading():
    base = 25
    noise = np.random.normal(0, 2)
    reading = base + noise
    # Random anomaly spike
    if np.random.rand() < 0.01:
        reading += np.random.uniform(8, 15)
    return reading

# -----------------------------
# Main loop - simulate updates
# -----------------------------
st.subheader("Live Sensor Readings")
plot_area = st.empty()
table_area = st.empty()
metrics_area = st.empty()

for i in range(2000):  # simulate 2000 time steps
    new_temp = generate_reading()
    
    # Append to DataFrame
    st.session_state.df.loc[len(st.session_state.df)] = [new_temp, 0, 0, "blue"]
    
    # Calculate rolling average
    st.session_state.df["Temp_Roll"] = st.session_state.df["Temperature"].rolling(st.session_state.window, min_periods=1).mean()
    
    # Anomaly detection
    model = IsolationForest(contamination=contamination, random_state=42)
    st.session_state.df["Anomaly"] = model.fit_predict(st.session_state.df[["Temp_Roll"]])
    st.session_state.df["Anomaly_Color"] = st.session_state.df["Anomaly"].map({1: "blue", -1: "red"})
    
    # -----------------------------
    # Update plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(st.session_state.df.index, st.session_state.df["Temp_Roll"], label="Rolling Avg", color="green", linewidth=2)
    ax.scatter(st.session_state.df.index, st.session_state.df["Temperature"], c=st.session_state.df["Anomaly_Color"])
    ax.set_title("Live IoT Temperature Sensor Readings")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    
    plot_area.pyplot(fig)
    
    # -----------------------------
    # Update anomaly table
    # -----------------------------
    table_area.subheader("Recent Detected Anomalies")
    table_area.dataframe(st.session_state.df[st.session_state.df["Anomaly"]==-1].tail(10)[["Temperature", "Temp_Roll"]])
    
    # -----------------------------
    # Update summary metrics
    # -----------------------------
    total_points = len(st.session_state.df)
    anomaly_points = len(st.session_state.df[st.session_state.df["Anomaly"]==-1])
    metrics_area.markdown(f"**Total readings:** {total_points}  |  **Anomalies detected:** {anomaly_points}  |  **% Anomalies:** {anomaly_points/total_points*100:.2f}%")
    
    time.sleep(pause_time)
# -----------------------------
# Download anomalies as CSV
# -----------------------------
csv = st.session_state.df[st.session_state.df["Anomaly"]==-1][["Temperature", "Temp_Roll"]].to_csv(index=False)
st.download_button(
    label="📥 Download Anomalies CSV",
    data=csv,
    file_name='anomalies.csv',
    mime='text/csv'
)
