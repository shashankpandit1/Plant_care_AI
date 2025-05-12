import streamlit as st
import requests

st.set_page_config(page_title="Identify Plant 🌿", page_icon="🌿")

st.title("🌿 Identify Medicinal Plants from Image")
st.markdown("Upload a plant image and the system will predict its species.")

uploaded_file = st.file_uploader("📸 Upload Plant Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Identifying the plant..."):
        response = requests.post(
            "http://localhost:8000/identify-plant/",  # adjust if hosted elsewhere
            files={"file": uploaded_file.getvalue()}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Identified Plant: **{result['plant_name']}**")
        else:
            st.error("❌ Failed to identify the plant. Try again.")
