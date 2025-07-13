import streamlit as st
import pandas as pd
import joblib
import time
from streamlit_lottie import st_lottie
import json

# Function to load Lottie JSON
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animation
heart_animation = load_lottie_file("heartbeat.json")  # download from lottiefiles.com

# Load model and utilities
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered", page_icon="❤️")

st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        h1, h3 { color: #2c3e50; }
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Animation
st.title("❤️ Heart Disease Prediction App")
st_lottie(heart_animation, height=200, key="heart")

st.markdown("#### Fill in the details below to assess your heart disease risk:")

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("👤 Age", 0, 120, 30)
    sex = st.selectbox("⚧️ Sex", ["M", "F"])
    chest_pain = st.selectbox("💓 Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("💉 Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    cholesterol = st.number_input("🥩 Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fasting_blood_sugar = st.selectbox("🧪 Fasting Blood Sugar > 120 mg/dl", ["0", "1"])

with col2:
    restting_ecg = st.selectbox("📊 Resting ECG", ["Normal", "ST", "LVH"])
    max_heart_rate = st.number_input("🏃‍♂️ Max Heart Rate Achieved", min_value=0, max_value=300, value=150)
    exercise_angina = st.selectbox("🏋️‍♀️ Exercise Induced Angina", ["Y", "N"])
    st_depression = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("📈 ST Segment Slope", ["Up", "Flat", "Down"])

# Prediction
if st.button("🔍 Predict"):
    raw_data = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_ecg": restting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_angina": exercise_angina,
        "st_depression": st_depression,
        "st_slope": st_slope
    }

    input_data = pd.DataFrame([raw_data])
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)
    scaled_data = scaler.transform(input_data)

    # Simulate loading animation
    with st.spinner("Analyzing data and predicting..."):
        time.sleep(2)
        prediction = model.predict(scaled_data)

    # Result
    st.markdown("### 🩺 Prediction Result:")
    if prediction[0] == 1:
        st.error("🚨 High risk of heart disease detected. Please consult a doctor.")
        
    else:
        st.success("✅ Low risk of heart disease. Keep maintaining a healthy lifestyle!")
       

