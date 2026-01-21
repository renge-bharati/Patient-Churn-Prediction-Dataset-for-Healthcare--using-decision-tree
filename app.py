import streamlit as st
import pandas as pd
import joblib

st.title("Patient Churn Prediction (Decision Tree)")

# Load model
model = joblib.load("decision_tree_churn_model.pkl")

st.subheader("Enter Patient Details")

age = st.number_input("Age", 0, 100)
tenure = st.number_input("Tenure (months)", 0, 120)
visits = st.number_input("Hospital Visits", 0, 50)

if st.button("Predict Churn"):
    input_df = pd.DataFrame([[age, tenure, visits]],
                            columns=["Age", "Tenure", "HospitalVisits"])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Patient is likely to churn ❌")
    else:
        st.success("Patient is not likely to churn ✅")
