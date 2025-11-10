import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/best_diabetes_model.joblib")

st.title("Predicción de Diabetes")
st.write("Ingrese los datos del paciente para predecir si tiene diabetes.")

# Inputs
pregnancies = st.number_input("Embarazos", min_value=0, max_value=20, value=1)
glucose = st.number_input("Nivel de Glucosa", min_value=0.0, max_value=300.0, value=125.0)
bloodpressure = st.number_input("Presión Arterial", min_value=0.0, max_value=140.0, value=70.0)
skinthickness = st.number_input("Grosor de la piel", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulina", min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input("IMC", min_value=0.0, max_value=80.0, value=28.0)
diabetes_pedigree = st.number_input("DiabetesPedigreeFunction (porcentaje de herencia familiar)", min_value=0.0, max_value=5.0, value=0.5)
age = st.number_input("Edad", min_value=0, max_value=120, value=33)

if st.button("Predecir"):
    X_new = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetes_pedigree, age]])
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0,1]
    if pred == 1:
        st.error(f"Predicción: POSITIVO (probabilidad {proba:.2f})")
    else:
        st.success(f"Predicción: NEGATIVO (probabilidad {proba:.2f})")
