import streamlit as st
import pandas as pd
import joblib

# Set application layout
st.set_page_config(
    page_title="Customer Churn & Monthly Charges Predictor",
    layout="wide"
)

# ---------------------------------------------------------
# 1. Load Trained Pipeline Models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    # Ensure both .pkl files are placed in the same folder as this script
    churn = joblib.load("telco_churn_model.pkl")
    revenue = joblib.load("telco_revenue_model.pkl")
    return churn, revenue

try:
    churn_model, revenue_model = load_models()
except Exception as e:
    st.error(
        f"Error loading model files: {e}. "
        "Please ensure 'telco_churn_model.pkl' and 'telco_revenue_model.pkl' "
        "are uploaded to your repository."
    )
    st.stop()

# ---------------------------------------------------------
# 2. User Interface Header
# ---------------------------------------------------------
st.title("Telco Customer Churn & Revenue Estimator")
st.markdown(
    "Provide the customer details below to estimate their **Monthly Charges** "
    "and determine their **Risk of Churn**."
)

# ---------------------------------------------------------
# 3. Input Form for Customer Characteristics
# ---------------------------------------------------------
with st.form("customer_input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.subheader("Tenure & Financials")
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, value=12, step=1)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0, step=10.0)

    with col2:
        st.subheader("Phone & Internet")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"] if phone_service == "No" else ["No", "Yes"]
        )
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("Streaming & Account")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    submit_button = st.form_submit_button("Calculate & Predict")

# ---------------------------------------------------------
# 4. Prediction Execution
# ---------------------------------------------------------
if submit_button:
    # 4A. Build record for revenue prediction (MonthlyCharges excluded)
    revenue_input = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
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
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "TotalCharges": total_charges,
    }

    df_revenue = pd.DataFrame([revenue_input])

    # 4B. Predict Monthly Charges using linear regression model
    predicted_monthly_charges = float(revenue_model.predict(df_revenue)[0])

    # 4C. Insert estimated charges for churn classification
    df_churn = df_revenue.copy()
    df_churn["MonthlyCharges"] = predicted_monthly_charges

    # 4D. Predict Churn outcome and likelihood using logistic regression model
    churn_class = int(churn_model.predict(df_churn)[0])
    churn_probability = float(churn_model.predict_proba(df_churn)[0][1])

    # ---------------------------------------------------------
    # 5. Display Outputs
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Model Predictions")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            label="Estimated Monthly Charges",
            value=f"${predicted_monthly_charges:.2f}"
        )

    with metric_col2:
        st.metric(
            label="Churn Probability",
            value=f"{churn_probability * 100:.1f}%"
        )

    if churn_class == 1:
        st.error("High Risk: This customer is predicted to churn.")
    else:
        st.success("Low Risk: This customer is likely to remain with the service.")
