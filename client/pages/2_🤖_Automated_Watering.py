import streamlit as st
import requests

st.set_page_config(page_title="Watering Threshold Setup", page_icon="🌿")

st.title("🌱 Automated Watering Threshold Setup")

# Step 1: Fetch plant list
response = requests.get("http://localhost:8000/plants")
# plant_list = [plant['Plant Name'] for plant in response.json()]
try:
    data = response.json()
    if isinstance(data, list):
        plant_list = [plant['Plant Name'] for plant in data]
    else:
        st.error("Unexpected response format from /plants endpoint.")
        st.stop()
except Exception as e:
    st.error(f"Failed to parse response: {e}")
    st.stop()

st.divider()
st.subheader("🔍 Select 3 Plants")

# selected_plants = st.multiselect("Select 3 Plants", plant_list, max_selections=3)
col1, col2, col3 = st.columns(3)

with col1:
    plant_1 = st.selectbox("Select Plant 1", plant_list, key="plant1")

with col2:
    remaining_1 = [p for p in plant_list if p != plant_1]
    plant_2 = st.selectbox("Select Plant 2", remaining_1, key="plant2")

with col3:
    remaining_2 = [p for p in remaining_1 if p != plant_2]
    plant_3 = st.selectbox("Select Plant 3", remaining_2, key="plant3")

selected_plants = [plant_1, plant_2, plant_3]


threshold_data = []

# if selected_plants and len(selected_plants) == 3:
#     for plant in selected_plants:
#         resp = requests.get("http://localhost:8000/plant_threshold", params={"name": plant})
#         data = resp.json()
#         st.write(f"**{plant}** → Min: {data['min_soil_moisture']}%, Max: {data['max_soil_moisture']}%")
#         threshold_data.append(data)
for plant in selected_plants:
    resp = requests.get("http://localhost:8000/plant_threshold", params={"name": plant})
    if resp.status_code == 200:
        data = resp.json()
        st.markdown(f"**{plant}** → 🟢 Min: `{data['min_soil_moisture']}%` &nbsp;&nbsp;&nbsp; 🔴 Max: `{data['max_soil_moisture']}%`")
        threshold_data.append(data)
    else:
        st.error(f"Failed to fetch thresholds for {plant}")

st.divider()
st.subheader("💾 Submit Thresholds")

if st.button("✅ ADD Plants for Watering"):
    with st.spinner("Uploading thresholds..."):
        res = requests.post("http://localhost:8000/watering-thresholds", json=threshold_data)
        if res.status_code == 200:
            st.success("✅ Plants successfully Added!")
        else:
            st.error("❌ An error occurred while adding.. Please try again.")
