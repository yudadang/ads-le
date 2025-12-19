import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PulseGuard Analytics",
    layout="wide"
)

# -------------------------------------------------
# HERO HEADER
# -------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 2rem;
    border-radius: 14px;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
}
</style>

<div class="hero">
    <h1>❤️ PulseGuard Analytics</h1>
    <h4>AI-Powered Cardiovascular Risk Assessment Dashboard</h4>
    <p>Applied Data Science • End-to-End Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("🔒 Dashboard is read-only. Model retrains automatically on load.")

# -------------------------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cardio_train.csv")

    # Convert age to years
    df["age"] = (df["age"] / 365).astype(int)

    # Feature engineering
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    return df

df = load_data()

# -------------------------------------------------
# MODEL TRAINING (ON LOAD)
# -------------------------------------------------
@st.cache_resource
def train_model(data):
    features = [
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
    ]

    X = data[features]
    y = data["cardio"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "precision": precision_score(y_test, model.predict(X_test)),
        "recall": recall_score(y_test, model.predict(X_test)),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    }

    return model, scaler, metrics, features

model, scaler, metrics, FEATURES = train_model(df)

# -------------------------------------------------
# NAVIGATION
# -------------------------------------------------
page = st.sidebar.selectbox("Navigation", ["📊 Dashboard", "🩺 Risk Prediction"])

# =================================================
# 📊 DASHBOARD PAGE
# =================================================
if page == "📊 Dashboard":

    st.header("📊 Population Health Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Avg Age", f"{df['age'].mean():.1f}")
    col3.metric("Avg BMI", f"{df['bmi'].mean():.1f}")
    col4.metric("Heart Disease Rate", f"{df['cardio'].mean()*100:.1f}%")

    st.subheader("📈 Age Distribution by Heart Disease")

    fig, ax = plt.subplots()
    sns.histplot(data=df, x="age", hue="cardio", bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Blood Pressure vs Disease")

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="cardio", y="ap_hi", ax=ax)
    ax.set_xticklabels(["No Disease", "Disease"])
    st.pyplot(fig)

    st.subheader("🔥 Top Feature Importances")

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Model Performance Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    m2.metric("Precision", f"{metrics['precision']:.2f}")
    m3.metric("Recall", f"{metrics['recall']:.2f}")
    m4.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

    st.subheader("📌 Interpretation")
    st.info(
        "Blood pressure, BMI, and age are the strongest predictors of cardiovascular risk. "
        "This dashboard is intended for **screening and educational purposes only**, not diagnosis."
    )

# =================================================
# 🩺 RISK PREDICTION PAGE
# =================================================
else:

    st.header("🩺 Individual Heart Disease Risk Prediction")

    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 18, 100, 50)
    height = c2.slider("Height (cm)", 140, 210, 170)
    weight = c3.slider("Weight (kg)", 40, 160, 75)

    bmi = weight / ((height/100)**2)

    ap_hi = st.slider("Systolic BP", 90, 200, 120)
    ap_lo = st.slider("Diastolic BP", 60, 120, 80)

    cholesterol = st.selectbox("Cholesterol", [1,2,3])
    gluc = st.selectbox("Glucose", [1,2,3])
    smoke = st.selectbox("Smoker", [0,1])
    alco = st.selectbox("Alcohol Intake", [0,1])
    active = st.selectbox("Physically Active", [0,1])
    gender = st.selectbox("Gender", [1,2])

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([[
            age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active, bmi
        ]], columns=FEATURES)

        input_scaled = scaler.transform(input_df)
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = model.predict(input_scaled)[0]

        st.subheader("🔬 Prediction Result")

        st.metric("Estimated Risk", f"{probability*100:.1f}%")
        st.progress(int(probability*100))

        if prediction == 1:
            st.error("⚠️ HIGH RISK — Elevated likelihood of cardiovascular disease.")
            st.warning(
                "Recommendation: Lifestyle modification and medical consultation are advised."
            )
        else:
            st.success("✅ LOW RISK — No immediate cardiovascular concerns detected.")
            st.info(
                "Recommendation: Maintain healthy habits and routine health monitoring."
            )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.write("---")
st.caption("PulseGuard Analytics • Applied Data Science Learning Evidence • 2025")
