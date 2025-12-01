import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient information to estimate the likelihood of heart disease.")

# Load ONLY the model (because it already contains the preprocessor)
model = joblib.load("model.pkl")

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("Patient Information")

age = st.slider("Age", 20, 100, 50)
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 250)
thalch = st.slider("Maximum Heart Rate (thalch)", 70, 220, 150)
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

sex_label = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_label == "Male" else 0

cp_label = st.selectbox("Chest Pain Type (cp)", [
    "typical angina", "atypical angina", "non-anginal", "asymptomatic"
])
cp_map = {"typical angina":0, "atypical angina":1, "non-anginal":2, "asymptomatic":3}
cp = cp_map[cp_label]

restecg_map = {"normal":0, "st-t abnormality":1, "lv hypertrophy":2}
restecg_label = st.selectbox("Resting ECG", list(restecg_map.keys()))
restecg = restecg_map[restecg_label]

slope_map = {"upsloping":0, "flat":1, "downsloping":2}
slope_label = st.selectbox("Slope", list(slope_map.keys()))
slope = slope_map[slope_label]

thal_map = {"normal":0, "fixed defect":1, "reversable defect":2}
thal_label = st.selectbox("Thal", list(thal_map.keys()))
thal = thal_map[thal_label]

ca = st.selectbox("Number of Major Vessels (ca)", [0,1,2,3])
fbs = st.selectbox("Fasting Blood Sugar >120 (fbs)", [0,1])
exang = st.selectbox("Exercise-induced Angina (exang)", [0,1])

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Heart Disease Risk"):

    input_data = pd.DataFrame([{
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "restecg": restecg,
        "exang": exang,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # DO NOT PREPROCESS — MODEL DOES IT
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("🩺 Prediction Result")
    st.write(f"**Estimated Probability: {probability*100:.2f}%**")

    if prediction == 1:
        st.error("⚠️ High Risk — Patient likely has heart disease.")
    else:
        st.success("✅ Low Risk — Patient unlikely to have heart disease.")

st.write("---")
st.caption("Machine Learning model based on UCI Heart Disease dataset.")
