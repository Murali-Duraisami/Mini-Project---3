import streamlit as st
import pandas as pd
import pickle

import sys
import sklearn

print(sys.executable)
print(sklearn.__version__)


# Load model
with open("artifacts/final_model.pkl","rb") as f:
    model=pickle.load(f)

# Load preprocessor
with open("artifacts/preprocessor.pkl","rb") as f:
    preprocessor=pickle.load(f)


# Title
st.title("💰 SmartPremium Insurance Predictor")

st.write("Predict Insurance Premium Amount")


# =========================
# User Inputs
# =========================

age=st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender=st.selectbox(
    "Gender",
    ["Male","Female"]
)

annual_income=st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

marital_status=st.selectbox(
    "Marital Status",
    ["Single","Married","Divorced"]
)

dependents=st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=1
)

education=st.selectbox(
    "Education Level",
    ["High School","Bachelor's","Master's","PhD"]
)

occupation=st.selectbox(
    "Occupation",
    ["Employed","Self-Employed","Unemployed"]
)

health_score=st.number_input(
    "Health Score",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

location=st.selectbox(
    "Location",
    ["Urban","Suburban","Rural"]
)

policy_type=st.selectbox(
    "Policy Type",
    ["Basic","Comprehensive","Premium"]
)

previous_claims=st.number_input(
    "Previous Claims",
    min_value=0,
    value=0
)

vehicle_age=st.number_input(
    "Vehicle Age",
    min_value=0.0,
    value=5.0
)

credit_score=st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

insurance_duration=st.number_input(
    "Insurance Duration",
    min_value=1,
    value=5
)

customer_feedback=st.selectbox(
    "Customer Feedback",
    ["Poor","Average","Good"]
)

smoking_status=st.selectbox(
    "Smoking Status",
    ["Yes","No"]
)

exercise_frequency=st.selectbox(
    "Exercise Frequency",
    ["Daily","Weekly","Monthly"]
)

property_type=st.selectbox(
    "Property Type",
    ["House","Apartment","Condo"]
)


# =========================
# Predict Button
# =========================

if st.button("Predict Premium"):


    input_data=pd.DataFrame({

        'Age':[age],
        'Gender':[gender],
        'Annual Income':[annual_income],
        'Marital Status':[marital_status],
        'Number of Dependents':[dependents],
        'Education Level':[education],
        'Occupation':[occupation],
        'Health Score':[health_score],
        'Location':[location],
        'Policy Type':[policy_type],
        'Previous Claims':[previous_claims],
        'Vehicle Age':[vehicle_age],
        'Credit Score':[credit_score],
        'Insurance Duration':[insurance_duration],
        'Customer Feedback':[customer_feedback],
        'Smoking Status':[smoking_status],
        'Exercise Frequency':[exercise_frequency],
        'Property Type':[property_type],
        'policy_year':[2020],
        'policy_month':[1],
        'policy_day':[1]

    })


    processed_data=preprocessor.transform(
        input_data
    )


    prediction=model.predict(
        processed_data
    )


    st.success(
        f"Predicted Insurance Premium: ₹ {prediction[0]:,.2f}"
    )