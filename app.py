import streamlit as st
import pandas as pd
import joblib

logistic_model = joblib.load(
    "abc_churn_model.pkl"
)

st.title("ABC Ltd. Customer Churn Prediction")
st.subheader("AI-Based Managerial Decision Support Tool")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

if st.button("Predict Churn"):
    # 1. Package the inputs into a single row DataFrame
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        # 2. Predict churn probability and class
        prediction = logistic_model.predict(input_data)[0]
        
        # If the model supports probability estimation:
        if hasattr(logistic_model, "predict_proba"):
            churn_prob = logistic_model.predict_proba(input_data)[0][1]
            prob_text = f" (Churn Probability: {churn_prob:.1%})"
        else:
            prob_text = ""

        # 3. Show result
        if prediction == 1 or prediction == "Yes":
            st.error(f"⚠️ High Risk: Customer is likely to churn!{prob_text}")
        else:
            st.success(f"✅ Low Risk: Customer is likely to stay.{prob_text}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
