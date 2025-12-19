import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="PulseGuard",
    layout="wide"
)

st.title("🫀 PulseGuard: Cardiovascular Risk Intelligence")
st.caption("Applied Data Science Learning Evidence – Health Analytics")

# -----------------------------
# Load & Prepare Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cardio_train.csv")

    # Age handling (dataset stores days)
    if "age" in df.columns:
        df["age_years"] = (df["age"] / 365).astype(int)
    elif "age_days" in df.columns:
        df["age_years"] = (df["age_days"] / 365).astype(int)
    else:
        st.error("Age column not found.")
        st.stop()

    # Target
    df.rename(columns={"cardio": "target"}, inplace=True)

    return df

df = load_data()

# -----------------------------
# Feature Selection
# -----------------------------
FEATURES = [
    "age_years", "gender", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

X = df[FEATURES]
y = df["target"]

# -----------------------------
# Train Model (on load)
# -----------------------------
@st.cache_resource
def train_model():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)

    metrics = {
        "accuracy": accuracy_score(y_test, pipeline.predict(X_test)),
        "precision": precision_score(y_test, pipeline.predict(X_test)),
        "recall": recall_score(y_test, pipeline.predict(X_test)),
        "f1": f1_score(y_test, pipeline.predict(X_test)),
        "roc_auc": roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    }

    return pipeline, metrics

model, metrics = train_model()

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["📊 Dashboard", "🩺 Prediction", "ℹ️ Feature Guide"]
)

# ============================================================
# 📊 DASHBOARD
# ============================================================
if page == "📊 Dashboard":

    st.header("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(df))
    c2.metric("Avg Age", int(df["age_years"].mean()))
    c3.metric("Heart Disease %", f"{df['target'].mean()*100:.1f}%")
    c4.metric("Model ROC-AUC", f"{metrics['roc_auc']:.2f}")

    st.subheader("📈 Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["age_years"], bins=30)
    st.pyplot(fig)

    st.subheader("❤️ Disease Distribution")
    fig, ax = plt.subplots()
    df["target"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["No Disease", "Disease"], rotation=0)
    st.pyplot(fig)

    st.subheader("📌 Model Performance")
    st.json(metrics)

# ============================================================
# 🩺 PREDICTION
# ============================================================
elif page == "🩺 Prediction":

    st.header("🩺 Individual Risk Assessment")

    age = st.slider("Age (years)", 20, 80, 50)

    gender_label = st.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender_label == "Female" else 2

    ap_hi = st.slider("Systolic BP", 90, 200, 120)
    ap_lo = st.slider("Diastolic BP", 60, 120, 80)

    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
    gluc = st.selectbox("Glucose Level", [1, 2, 3])

    smoke = st.selectbox("Smoker", [0, 1])
    alco = st.selectbox("Alcohol Intake", [0, 1])
    active = st.selectbox("Physically Active", [0, 1])

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([{
            "age_years": age,
            "gender": gender,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active
        }])

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.subheader("🔬 Prediction Result")
        st.progress(int(prob * 100))
        st.write(f"**Estimated Risk: {prob*100:.2f}%**")

        if pred == 1:
            st.error("⚠️ HIGH RISK — Cardiovascular disease likely.")
        else:
            st.success("✅ LOW RISK — No cardiovascular disease detected.")

# ============================================================
# ℹ️ FEATURE GUIDE
# ============================================================
else:

    st.header("ℹ️ Feature Guide & Interpretation")

    st.markdown("""
    **PulseGuard** estimates cardiovascular risk using clinical indicators
    derived from real patient data.

    ### Feature Explanations
    • **Age** – Risk increases with age  
    • **Gender** – Female (1), Male (2)  
    • **Blood Pressure** – Key hypertension indicator  
    • **Cholesterol & Glucose** – Metabolic risk factors  
    • **Lifestyle (Smoking, Alcohol, Activity)** – Behavioral risk modifiers  

    ### Model Explanation
    A Random Forest model was trained on thousands of records to capture
    nonlinear relationships between health indicators.

    ### Recommendation
    This tool is **educational** and should not replace professional diagnosis.
    """)

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("PulseGuard • Applied Data Science • 2025")
