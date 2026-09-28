import streamlit as st
import pandas as pd
import joblib

logistic_model = joblib.load("abc_churn_model.pkl")

st.title("ABC Ltd. Customer Churn Prediction")
st.subheader("AI-Based Managerial Decision Support Tool")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Demographic & Account")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

with col2:
    st.markdown("### Services Subscribed")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

st.markdown("---")

if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        prediction = logistic_model.predict(input_data)[0]

        if hasattr(logistic_model, "predict_proba"):
            churn_prob = logistic_model.predict_proba(input_data)[0][1]
            prob_text = f" (Churn Probability: {churn_prob:.1%})"
        else:
            prob_text = ""

        if prediction == 1 or prediction == "Yes":
            st.error(f"⚠️ High Risk: Customer is likely to churn!{prob_text}")
        else:
            st.success(f"✅ Low Risk: Customer is likely to stay.{prob_text}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
