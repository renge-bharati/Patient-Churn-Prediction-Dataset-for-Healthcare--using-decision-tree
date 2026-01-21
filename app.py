import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Patient Churn Prediction", layout="centered")
st.title("🩺 Patient Churn Prediction (Decision Tree)")

# Load trained model
model_path = "decision_tree_churn_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file '{model_path}' not found! Upload it to GitHub.")
    st.stop()

# Upload dataset for predictions
uploaded_file = st.file_uploader("Upload Patient Churn CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    # Predict churn
    if st.button("Predict Churn"):
        try:
            predictions = model.predict(df_encoded)
            df["Churn_Prediction"] = predictions
            st.subheader("Prediction Results")
            st.dataframe(df)
            st.success("Churn prediction completed successfully!")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
else:
    st.info("Please upload a CSV file to start predictions.")
