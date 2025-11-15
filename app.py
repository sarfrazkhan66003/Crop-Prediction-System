import streamlit as st
import numpy as np
import joblib

# -------------------- Load Files --------------------
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# -------------------- Page Config --------------------
st.set_page_config(page_title="🌱 Crop Recommendation", page_icon="🌾", layout="centered")

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
        /* Green gradient background */
        .main {
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9, #a5d6a7, #81c784);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
        }
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        /* Title */
        .title {
            font-size: 2.6rem;
            font-weight: bold;
            text-align: center;
            color: #1b5e20; /* Dark green for visibility */
            padding: 10px;
            text-shadow: 1px 1px 4px rgba(255,255,255,0.8);
        }
        /* Card style */
        .card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            text-align: center;
        }
        /* Prediction result */
        .prediction {
            font-size: 2.4rem;
            font-weight: bold;
            color: #2e7d32; /* Dark green text */
            margin-top: 10px;
        }
        /* Note under result */
        .note {
            font-size: 0.95rem;
            color: #1b5e20;
            margin-top: 8px;
            font-style: italic;
        }
        /* Button style */
        div.stButton > button {
            background: linear-gradient(90deg, #66bb6a, #43a047);
            color: white;
            border: none;
            padding: 0.6rem 1.4rem;
            font-size: 1.15rem;
            border-radius: 30px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #2e7d32, #66bb6a);
            transform: scale(1.07);
            box-shadow: 0px 4px 20px rgba(56, 142, 60, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Title --------------------
st.markdown("<div class='title'>🌾 Smart Crop Prediction AI 🌱</div>", unsafe_allow_html=True)
st.write("<center>Enter your soil & climate details to find the most suitable crop for your farm.</center>", unsafe_allow_html=True)

# -------------------- Input Fields --------------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌿 Nitrogen", min_value=0.0, step=1.0)
    P = st.number_input("🌿 Phosphorus", min_value=0.0, step=1.0)
    K = st.number_input("🌿 Potassium", min_value=0.0, step=1.0)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, step=0.1)

with col2:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, step=0.1)
    ph = st.number_input("⚗️ pH", min_value=0.0, step=0.1)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=0.1)

# -------------------- Prediction --------------------
if st.button("🔍 Predict Best Crop"):
    try:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_data = scaler.transform(input_data)
        prediction_numeric = model.predict(scaled_data)[0]
        prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

        st.markdown(f"""
            <div class='card'>
                ✅ Recommended Crop:
                <div class='prediction'>{prediction_label}</div>
                <div class='note'>
                    📌 This recommendation is based on your input values: 
                    Temperature = <b>{temperature}°C</b> and pH = <b>{ph}</b>.
                </div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
