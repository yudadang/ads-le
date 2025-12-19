import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"  # 🔒 UI locked
)

st.title("❤️ Heart Disease Risk Prediction Dashboard")
st.caption("Applied Data Science – Full End-to-End Deployment")

# -----------------------------
# LOAD & PREPARE DATA (LOCKED)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/heart_disease_kaggle.csv")
    df["target"] = (df["target"] > 0).astype(int)
    return df

df = load_data()

FEATURES = [
    "age", "trestbps", "chol", "thalach",
    "oldpeak", "sex", "cp", "fbs",
    "restecg", "exang", "slope", "ca", "thal"
]

X = df[FEATURES]
y = df["target"]

# -----------------------------
# TRAIN MODEL (ON LOAD)
# -----------------------------
@st.cache_resource
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

    return model, scaler, metrics

model, scaler, metrics = train_model(X, y)

# -----------------------------
# KPI SECTION (LOCKED)
# -----------------------------
st.subheader("📊 Model Performance (Locked)")

col1, col2, col3 = st.columns(3)
col1.metric("Dataset Size", f"{len(df):,}")
col2.metric("Accuracy", f"{metrics['accuracy']:.2f}")
col3.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")

# -----------------------------
# EDA VISUALS (READ-ONLY)
# -----------------------------
st.subheader("📈 Dataset Insights")

fig, ax = plt.subplots()
sns.countplot(x=df["target"], ax=ax)
ax.set_xticklabels(["No Disease", "Disease"])
st.pyplot(fig)

fig, ax = plt.subplots()
sns.histplot(df["age"], kde=True, ax=ax)
st.pyplot(fig)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🩺 Patient Risk Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 55)
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 130)
        chol = st.slider("Cholesterol", 100, 600, 250)
        thalach = st.slider("Max Heart Rate", 70, 220, 150)
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.5)

    with col2:
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        exang = st.selectbox("Exercise-Induced Angina", [0, 1])
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [0, 1, 2])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_df = pd.DataFrame([[ 
        age, trestbps, chol, thalach, oldpeak,
        sex, cp, fbs, restecg, exang, slope, ca, thal
    ]], columns=FEATURES)

    input_scaled = scaler.transform(input_df)
    risk = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔬 Prediction Result")
    st.progress(int(risk * 100))
    st.write(f"**Estimated Heart Disease Risk: {risk*100:.2f}%**")

    if risk > 0.6:
        st.error("⚠️ High risk detected. Medical evaluation recommended.")
    elif risk > 0.3:
        st.warning("🟠 Moderate risk. Lifestyle changes advised.")
    else:
        st.success("✅ Low risk detected.")

# -----------------------------
# FOOTER (LOCKED)
# -----------------------------
st.write("---")
st.caption(
    "This application demonstrates the full Applied Data Science lifecycle: "
    "problem definition, data processing, modeling, evaluation, and deployment."
)
