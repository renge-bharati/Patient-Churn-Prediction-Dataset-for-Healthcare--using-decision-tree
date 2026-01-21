import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Patient Churn Prediction", layout="centered")

st.title("🩺 Patient Churn Prediction (Decision Tree)")

# Load trained model
with open("decision_tree_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

uploaded_file = st.file_uploader("Upload Patient Churn CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    le = LabelEncoder()
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    if st.button("Predict Churn"):
        predictions = model.predict(df_encoded)
        df["Churn_Prediction"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(df)

        st.success("Churn prediction completed successfully!")
else:
    st.info("Please upload a CSV file to start prediction.")
