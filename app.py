import streamlit as st
import pandas as pd
import joblib

logistic_model = joblib.load("abc_churn_model.pkl")

st.title("ABC Ltd. Customer Churn Prediction")
st.subheader("AI-Based Managerial Decision Support Tool")

gender = st.selectbox("Gender",["Male", "Female"])

senior = st.selectbox("Senior Citizen",[0, 1])

partner = st.selectbox("Partner",["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input( "Tenure (months)", min_value=0,max_value=100,value=12)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

total_charges = st.number_input("Total Charges",min_value=0.0,value=500.0)

if st.button("Predict Churn"):

    st.info(
        "Complete the remaining customer fields "
        "before making a production prediction."
    )
