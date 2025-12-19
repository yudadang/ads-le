import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Disease Risk Dashboard",
    layout="wide"
)

st.title("❤️ Cardiovascular Disease Risk Analysis & Prediction")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("model.pkl")

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🩺 Risk Prediction", "📖 Project Narrative"]
)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------
if page == "📊 Dashboard":

    st.header("📊 Dataset Overview (Kaggle Cardiovascular Dataset)")

    st.markdown(
        """
        **Dataset Source:** Kaggle Cardiovascular Disease Dataset  
        **Size:** ~70,000 patients  
        **Goal:** Identify risk factors associated with cardiovascular disease
        """
    )

    st.subheader("🔢 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Size", "70,000+")
    col2.metric("Target Variable", "Cardio (0 / 1)")
    col3.metric("Model Type", "Random Forest")
    col4.metric("Class Handling", "Balanced Weights")

    st.divider()

    st.subheader("📌 Risk Factors Used in the Model")
    st.write(
        """
        - Age (years)
        - Gender
        - Systolic & Diastolic Blood Pressure
        - Cholesterol & Glucose Levels
        - BMI
        - Smoking, Alcohol Intake, Physical Activity
        """
    )

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif page == "🩺 Risk Prediction":

    st.header("🩺 Individual Cardiovascular Risk Prediction")

    st.write("Enter patient health information below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age (years)", 30, 80, 55)
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender = 1 if gender == "Male" else 0

    with col2:
        ap_hi = st.slider("Systolic BP", 90, 200, 130)
        ap_lo = st.slider("Diastolic BP", 60, 120, 85)

    with col3:
        cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
        gluc = st.selectbox("Glucose Level", [1, 2, 3])

    col4, col5, col6 = st.columns(3)

    with col4:
        height = st.slider("Height (cm)", 140, 200, 170)
        weight = st.slider("Weight (kg)", 40, 150, 75)

    with col5:
        smoke = st.selectbox("Smoker", [0, 1])
        alco = st.selectbox("Alcohol Intake", [0, 1])

    with col6:
        active = st.selectbox("Physically Active", [0, 1])

    # Feature Engineering (must match training)
    bmi = weight / ((height / 100) ** 2)

    input_df = pd.DataFrame([{
        "gender": gender,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "age_years": age,
        "bmi": bmi
    }])

    if st.button("🔍 Predict Risk"):

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.subheader("📈 Prediction Result")

        st.progress(int(probability * 100))
        st.write(f"**Estimated Risk Probability:** `{probability*100:.2f}%`")

        if prediction == 1:
            st.error("⚠️ High Risk of Cardiovascular Disease")
        else:
            st.success("✅ Low Risk of Cardiovascular Disease")

        # ---------------- SHAP EXPLANATION ----------------
        st.subheader("🧠 Risk Explanation (SHAP)")

        explainer = shap.TreeExplainer(model.named_steps["rf"])
        processed_input = model.named_steps["preprocess"].transform(input_df)
        shap_values = explainer.shap_values(processed_input)

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[1][0],
                base_values=explainer.expected_value[1],
                feature_names=input_df.columns
            ),
            show=False
        )
        st.pyplot(fig)

# --------------------------------------------------
# PROJECT NARRATIVE PAGE
# --------------------------------------------------
elif page == "📖 Project Narrative":

    st.header("📖 Project Narrative")

    st.markdown(
        """
        ### Problem Definition
        Cardiovascular disease remains one of the leading causes of death worldwide.
        This project aims to predict cardiovascular disease risk using patient health indicators.

        ### Data Collection
        The Kaggle Cardiovascular Disease Dataset was used, containing approximately
        70,000 patient records with demographic, lifestyle, and clinical attributes.

        ### Data Preprocessing
        - Converted age from days to years
        - Engineered BMI from height and weight
        - Encoded categorical variables
        - Standardized numerical features

        ### Modeling
        A Random Forest classifier with class-balanced weights was trained to handle
        class imbalance while maintaining model stability.

        ### Evaluation
        Model performance was assessed using:
        - ROC-AUC
        - Precision
        - Recall
        - F1-score

        ### Insights & Recommendations
        - Blood pressure, age, BMI, and cholesterol are strong predictors
        - Lifestyle factors (smoking, physical inactivity) significantly increase risk
        - The model can assist in early screening and preventive healthcare planning
        """
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Applied Data Science | Cardiovascular Risk Prediction | Streamlit App")
