import streamlit as st
import numpy as np

import joblib

model = joblib.load("models/best_model.joblib")

# -----------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor", page_icon="🔮", layout="centered"
)

st.title("Customer Churn Prediction Project")
st.write("Enter the customer details below to predict the likelihood of churn.")
st.markdown("---")

## 1. Demographics & Account Details
st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.checkbox("Is the customer a Senior Citizen?", value=False)
    senior_val = 1 if senior_citizen else 0

    # Family Score Inputs
    partner = st.checkbox("Does the customer have a partner?", value=False)
    dependents = st.checkbox("Does the customer have dependents?", value=False)
    family_score = int(partner) + int(dependents)

with col2:
    tenure = st.number_input(
        "Tenure (in months)", min_value=0, max_value=120, value=3, step=1
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=300.0, value=99.0, step=0.5
    )

# Contract Map
contract_display = ("Month-to-month", "One year", "Two year")
contract_options = list(range(len(contract_display)))
contract_selected = st.selectbox(
    "Contract Type", contract_options, format_func=lambda x: contract_display[x]
)

# Billing & Payment
col3, col4 = st.columns(2)
with col3:
    payment_options = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    payment_method = st.selectbox("Payment Method", options=payment_options, index=0)

    # Returns 1 if 'Electronic check' is selected, 0 otherwise
    electronic_check_val = 1 if payment_method == "Electronic check" else 0
    paperless = st.checkbox("Paperless Billing", value=True)
    paperless_val = 1 if paperless else 0

st.markdown("---")

## 2. Internet Service Logic
st.subheader(" Internet Services")

internet_map = {"No": 0, "DSL": 1, "Fiber Optic": 2}
internet_choice = st.selectbox(
    "Internet Service Type", options=list(internet_map.keys()), index=2
)
internet_service_val = internet_map[internet_choice]

# Dynamic inputs based on Internet Service selection
internet_service_score = -1

if internet_choice in ["DSL", "Fiber Optic"]:
    st.info("ℹSelect the additional add-on services the customer has opted for:")

    col_a, col_b = st.columns(2)
    with col_a:
        tech_support = st.checkbox("Tech Support")
        device_protection = st.checkbox("Device Protection")
    with col_b:
        online_backup = st.checkbox("Online Backup")
        online_security = st.checkbox("Online Security")

    # Calculate score: sum of all selected services (0 to 4)
    internet_service_score = (
        int(tech_support)
        + int(device_protection)
        + int(online_backup)
        + int(online_security)
    )
else:
    st.warning(
        "⚠️ Additional internet services are disabled because the customer has no Internet Service."
    )

st.markdown("---")

## 3. Prediction Execution
if st.button("Predict Churn Status", type="primary"):
    # Constructing the exact feature vector based on your dictionary structure
    # Order matters! Ensure this matches your model's expected training feature order.
    features = [
        senior_val,
        tenure,
        internet_service_val,
        contract_selected,
        paperless_val,
        monthly_charges,
        family_score,
        internet_service_score,
        electronic_check_val,
    ]

    # Reshape for model.predict() -> [[f1, f2, f3, ...]]
    features_array = np.array([features])

    # Run prediction
    prediction = model.predict(features_array)[0]

    probabilities = model.predict_proba(features_array)[0]

    # Extract confidence based on the prediction
    prob_not_churn = probabilities[0]
    prob_churn = probabilities[1]

    # Display Results
    st.subheader("Prediction Result")

    if prediction == 1:
        confidence = prob_churn * 100
        st.error(" This customer is predicted to **CHURN**.")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
    else:
        confidence = prob_not_churn * 100
        st.success("This customer is predicted to **NOT CHURN**.")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

    # Optional: Show a breakdown of both sides
    with st.expander("Detailed Probability Breakdown"):
        st.write(f"📉 **Probability of Churning:** {prob_churn * 100:.2f}%")
        st.write(f"📈 **Probability of Staying:** {prob_not_churn * 100:.2f}%")

    # Optional Debug section to check the exact sent dictionary
    with st.expander("See raw features payload sent to model"):
        payload = {
            "SeniorCitizen": senior_val,
            "tenure": tenure,
            "InternetService": internet_service_val,
            "Contract": contract_selected,
            "PaperlessBilling": paperless_val,
            "MonthlyCharges": monthly_charges,
            "family_score": family_score,
            "Internet_service_score": internet_service_score,
            "Payment_Electronic_check": electronic_check_val,
        }
        st.json(payload)
