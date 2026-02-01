import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn risk.")

tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_dict = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly,
    "Total Charges": total,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "Tech Support_Yes": 1 if tech_support == "Yes" else 0,
    "Internet Service_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "Internet Service_No": 1 if internet == "No" else 0
}

input_df = pd.DataFrame([input_dict])

for col in model.feature_names_in_:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[model.feature_names_in_]

input_df[input_df.columns[:3]] = scaler.transform(input_df[input_df.columns[:3]])
input_final = imputer.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_final)[0]
    prob = model.predict_proba(input_final)[0][1]

    if prediction == 1:
        st.error(f"High Churn Risk ({prob*100:.2f}%)")
    else:
        st.success(f"Low Churn Risk ({prob*100:.2f}%)")
