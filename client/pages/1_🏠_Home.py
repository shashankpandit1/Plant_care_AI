import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from fuzzywuzzy import process

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Home", page_icon="🏠", layout="centered")
st.title("🏡 Real-Time Plant Monitoring & Recommendations")
st.markdown("View your live sensor readings and get AI-based plant recommendations in one place.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_plant_data():
    return pd.read_csv("plant_environment_data.csv")

plant_df = load_plant_data()

# -------------------------------
# Fetch Sensor Data
# -------------------------------
try:
    response = requests.get("http://localhost:8000/sensor-data/")
    response.raise_for_status()
    data = response.json()
except Exception as e:
    st.error(f"🚨 Error fetching sensor data: {e}")
    st.stop()

# -------------------------------
# Gauge Chart Function
# -------------------------------
def gauge(title, value, min_val, max_val, unit, color):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"<b>{title}</b>", "font": {"size": 20}},
        number={"suffix": f" {unit}"},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [min_val, max_val * 0.5], "color": "#f2f2f2"},
                {"range": [max_val * 0.5, max_val * 0.8], "color": "#d9f2d9"},
                {"range": [max_val * 0.8, max_val], "color": "#ffd6cc"},
            ],
        }
    ))

# -------------------------------
# Display Gauges
# -------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(gauge("🌡️ Temperature", data["temperature"], 0, 50, "°C", "tomato"), use_container_width=True)
with col2:
    st.plotly_chart(gauge("💧 Humidity", data["humidity"], 0, 100, "%", "skyblue"), use_container_width=True)
with col3:
    st.plotly_chart(gauge("🔆 Light", data["light"], 0, 1000, "lux", "gold"), use_container_width=True)

st.markdown("---")
st.success("✅ All sensors are active and data is up to date.")

# -------------------------------
# AI Recommender
# -------------------------------
st.subheader("🌱 AI-Based Plant Recommendations")

try:
    rec = requests.post("http://localhost:8000/recommend-plants/", json=data)
    rec.raise_for_status()
    result = rec.json()
    recommended_plants = result.get("recommended_plants", [])

    if recommended_plants:
        st.success("✅ Recommended Plants:")
        
        # Fuzzy matching function
        def find_best_match(plant_name, choices, threshold=70):
            match, score = process.extractOne(plant_name, choices)
            return match if score >= threshold else None

        for plant in recommended_plants:
            with st.expander(f"🌿 {plant}"):
                match_name = find_best_match(plant, plant_df["Plant Name"].tolist())
                if match_name:
                    row = plant_df[plant_df["Plant Name"] == match_name].iloc[0]
                    st.markdown(f"**🪴 Common Name (India):** {row['Common Name (India)']}")
                    st.markdown(f"**🌡️ Temperature Range:** {row['Min Temp']}°C - {row['Max Temp']}°C")
                    st.markdown(f"**💧 Humidity Range:** {row['Min Humidity']}% - {row['Max Humidity']}%")
                    st.markdown(f"**🔆 Light Range:** {row['Min Light']} lux - {row['Max Light']} lux")
                    st.markdown(f"**🌱 Soil Moisture Range:** {row['Min Soil Moisture']}% - {row['Max Soil Moisture']}%")
                else:
                    st.warning("⚠️ No matching data found for this plant.")
    else:
        st.warning("🤷‍♂️ No plant recommendations were returned.")

except Exception as e:
    st.error(f"🚨 Error while requesting recommendations: {e}")
