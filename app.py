import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="Heart Disease Risk Dashboard",
    layout="wide"
)

st.title("❤️ Heart Disease Risk Analysis & Prediction")
st.write(
    "This application demonstrates the **full data science lifecycle**: "
    "data loading, preprocessing, modeling, evaluation, and deployment."
)

# ======================================================
# LOAD DATA (CACHED)
# ======================================================
@st.cache_data
def load_data():
    df = pd.read_csv("cardio_train.csv", sep=";")
    return df

df = load_data()

# ======================================================
# DATA CLEANING
# ======================================================
df["age"] = (df["age"] / 365).astype(int)  # convert days → years
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# Target
target = "cardio"

features = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol",
    "gluc", "smoke", "alco", "active", "bmi"
]

X = df[features]
y = df[target]

# ======================================================
# TRAIN MODEL (TRAIN-ON-LOAD)
# ======================================================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    metrics = {
        "Accuracy": accuracy_score(y_test, pipeline.predict(X_test)),
        "Precision": precision_score(y_test, pipeline.predict(X_test)),
        "Recall": recall_score(y_test, pipeline.predict(X_test)),
        "ROC-AUC": roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    }

    return pipeline, metrics

model, metrics = train_model(X, y)

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🩺 Risk Prediction"]
)

# ======================================================
# DASHBOARD PAGE
# ======================================================
if page == "📊 Dashboard":

    st.header("📊 Dataset Overview & KPIs")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(df))
    col2.metric("Avg Age", round(df["age"].mean(), 1))
    col3.metric("Heart Disease Rate", f"{df['cardio'].mean()*100:.1f}%")
    col4.metric("Avg BMI", round(df["bmi"].mean(), 1))

    st.subheader("📈 Model Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    m2.metric("Precision", f"{metrics['Precision']:.2f}")
    m3.metric("Recall", f"{metrics['Recall']:.2f}")
    m4.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")

    st.subheader("📌 Age Distribution by Heart Disease")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="age", hue="cardio", bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("🔥 Feature Importance")
    importances = model.named_steps["rf"].feature_importances_
    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

    st.subheader("📖 Interpretation")
    st.write(
        """
        - Blood pressure, age, and BMI are strong predictors of heart disease.
        - Lifestyle factors such as smoking and physical activity also contribute.
        - The model demonstrates strong recall, making it suitable for risk screening.
        """
    )

# ======================================================
# PREDICTION PAGE
# ======================================================
else:
    st.header("🩺 Heart Disease Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 30, 80, 50)
        height = st.slider("Height (cm)", 140, 200, 170)
        weight = st.slider("Weight (kg)", 40, 150, 70)
        ap_hi = st.slider("Systolic BP", 90, 200, 120)
        ap_lo = st.slider("Diastolic BP", 60, 130, 80)
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
        gluc = st.selectbox("Glucose", [1, 2, 3])

    with col2:
        gender = st.selectbox("Gender", [1, 2])
        smoke = st.selectbox("Smoker", [0, 1])
        alco = st.selectbox("Alcohol Intake", [0, 1])
        active = st.selectbox("Physically Active", [0, 1])

    bmi = weight / ((height / 100) ** 2)

    if st.button("🔍 Predict Risk"):
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
            "active": active,
            "bmi": bmi
        }])

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.subheader("📌 Prediction Result")
        st.progress(int(probability * 100))
        st.write(f"**Estimated Risk: {probability*100:.2f}%**")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.caption(
            "This prediction is based on statistical patterns in population health data and is not a medical diagnosis."
        )

# ======================================================
# FOOTER
# ======================================================
st.write("---")
st.caption(
    "Applied Data Science Learning Evidence | End-to-End ML Dashboard | Streamlit"
)
