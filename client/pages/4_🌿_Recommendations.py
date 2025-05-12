import streamlit as st
import requests

st.set_page_config(page_title="AI Plant Recommender", page_icon="🤖")

st.title("🤖 Smart Plant Recommender")
st.markdown("Get intelligent plant recommendations based on current sensor data.")

# Get sensor data
res = requests.get("http://localhost:8000/sensor-data/")
if res.status_code == 200:
    data = res.json()
    st.write(f"### 📡 Current Readings:")
    st.write(f"- 🌡️ Temperature: {data['temperature']} °C")
    st.write(f"- 💧 Humidity: {data['humidity']} %")
    st.write(f"- 🔆 Light: {data['light']} lux")
else:
    st.error("Failed to fetch sensor data")
    st.stop()

# Recommend button
if st.button("🎯 Recommend Plants"):
    rec = requests.post("http://localhost:8000/recommend-plants/", json=data)
    if rec.status_code == 200:
        result = rec.json()
        st.success("✅ Recommended Plants:")
        for plant in result["recommended_plants"]:
            st.markdown(f"- 🌱 **{plant}**")
    else:
        st.error("❌ Failed to get recommendations.")
