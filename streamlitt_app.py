import streamlit as st
import joblib
import numpy as np

model = joblib.load("hybrid_model.pkl")

st.title("Diabetes Prediction App")

# Input fields for each feature
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
age = st.number_input("Age", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

if st.button("Predict"):
    features = [pregnancies, age, glucose, blood_pressure, bmi, insulin, skin_thickness, dpf]
    data = np.array([features])
    pred = model.predict(data)[0]
    st.write(f"Prediction: {'Diabetic' if pred == 1 else 'Not Diabetic'}")