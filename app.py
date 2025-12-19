import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(
    page_title="PulseGuard – Heart Disease Risk Dashboard",
    layout="wide"
)

st.title("❤️ PulseGuard")
st.caption("An Applied Data Science Dashboard for Heart Disease Risk Prediction")

# -------------------------------------------------
# Load & Prepare Data
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cardio_train.csv", sep=";")  # <-- IMPORTANT FIX

    # Convert age from days to years
    df["age"] = (df["age"] / 365.25).astype(int)

    # Rename target
    df = df.rename(columns={"cardio": "target"})

    # Gender mapping (1 = Female, 2 = Male)
    df["gender_label"] = df["gender"].map({1: "Female", 2: "Male"})

    return df



df = load_data()

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🩺 Prediction", "ℹ️ Feature Guide"]
)

# -------------------------------------------------
# 📊 DASHBOARD
# -------------------------------------------------
if page == "📊 Dashboard":

    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col2.metric("Average Age", int(df["age"].mean()))
    col3.metric(
        "Heart Disease Rate",
        f"{round(df['target'].mean()*100, 1)}%"
    )

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("❤️ Heart Disease Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["target"], ax=ax)
    ax.set_xticklabels(["No Disease", "Disease"])
    st.pyplot(fig)

    st.subheader("📌 Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------
# 🩺 PREDICTION
# -------------------------------------------------
elif page == "🩺 Prediction":

    st.header("🩺 Heart Disease Risk Prediction")

    # -----------------------------
    # Model Training (ON LOAD)
    # -----------------------------
    features = [
        "age", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol",
        "gluc", "smoke", "alco", "active"
    ]

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # -----------------------------
    # User Inputs
    # -----------------------------
    st.subheader("👤 Patient Information")

    age = st.slider("Age", 18, 100, 50)

    gender_label = st.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender_label == "Female" else 2

    height = st.slider("Height (cm)", 140, 210, 170)
    weight = st.slider("Weight (kg)", 40, 150, 70)

    ap_hi = st.slider("Systolic BP", 90, 200, 120)
    ap_lo = st.slider("Diastolic BP", 60, 140, 80)

    cholesterol = st.selectbox(
        "Cholesterol Level",
        [1, 2, 3],
        format_func=lambda x: ["Normal", "Above Normal", "High"][x-1]
    )

    gluc = st.selectbox(
        "Glucose Level",
        [1, 2, 3],
        format_func=lambda x: ["Normal", "Above Normal", "High"][x-1]
    )

    smoke = st.selectbox("Smoker?", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    alco = st.selectbox("Alcohol Intake?", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    active = st.selectbox("Physically Active?", [0, 1], format_func=lambda x: ["No", "Yes"][x])

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict Risk"):

        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active
        }])

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.subheader("🔬 Result")
        st.write(f"**Estimated Heart Disease Risk: {probability*100:.2f}%**")
        st.progress(int(probability * 100))

        if prediction == 1:
            st.error("⚠️ HIGH RISK — Medical consultation advised.")
        else:
            st.success("✅ LOW RISK — Maintain healthy lifestyle.")

        # -----------------------------
        # Model Metrics
        # -----------------------------
        st.subheader("📈 Model Performance")
        y_pred = model.predict(X_test)

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
        col2.metric("ROC AUC", round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 3))

        with st.expander("Classification Report"):
            st.text(classification_report(y_test, y_pred))

# -------------------------------------------------
# ℹ️ FEATURE GUIDE
# -------------------------------------------------
else:

    st.header("ℹ️ Feature Explanation Guide")

    st.markdown("""
    **PulseGuard** uses medical indicators commonly associated with cardiovascular risk:

    • **Age** – Risk increases with age  
    • **Blood Pressure (ap_hi / ap_lo)** – Hypertension indicator  
    • **Cholesterol & Glucose** – Metabolic risk markers  
    • **Smoking & Alcohol** – Lifestyle risk factors  
    • **Physical Activity** – Protective factor  

    ⚠️ This tool is **not a medical diagnosis**.  
    It is a **data-driven risk estimation** for educational purposes.
    """)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.write("---")
st.caption("PulseGuard • Applied Data Science Learning Evidence • 2025")

