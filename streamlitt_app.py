import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("hybrid_model.pkl")

# Sidebar for branding and instructions
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Emblem_of_the_International_Red_Cross.svg/1200px-Emblem_of_the_International_Red_Cross.svg.png", width=100)
st.sidebar.title("Hospital Diabetes Screening")
st.sidebar.markdown("""
Welcome to the Diabetes Prediction System.  
Please enter patient details below.  
All data is confidential.
""")

st.title("🩺 Diabetes Prediction App")
st.markdown("#### Enter Patient Information:")

# Layout input fields in columns
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    age = st.number_input("Age", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
with col2:
    bmi = st.number_input("BMI", min_value=0.0)
    insulin = st.number_input("Insulin Level", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

if st.button("Predict", use_container_width=True):
    features = [pregnancies, age, glucose, blood_pressure, bmi, insulin, skin_thickness, dpf]
    data = np.array([features])
    pred = model.predict(data)[0]
    if pred == 1:
        st.error("Prediction: Diabetic", icon="🚨")
        st.markdown("**Please consult a healthcare professional for further evaluation.**")
    else:
        st.success("Prediction: Not Diabetic", icon="✅")
        st.markdown("**No immediate risk detected. Maintain healthy lifestyle.**")

st.markdown("---")
st.caption("© 2025 Hospital Diabetes Screening System | For medical use only.")
