import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Exam Score Predictor", page_icon="📘")

# Title
st.title("📘 Exam Score Prediction App")
st.write("Enter study details to predict exam score")

# Load model
model = joblib.load("linear_regression_model.pkl")

# Input from user
hours_studied = st.number_input(
    "Hours Studied",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

# Predict button
if st.button("Predict Exam Score"):
    X = np.array([[hours_studied]])   # 2D input
    prediction = model.predict(X)

    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")
