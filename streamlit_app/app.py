import streamlit as st
import requests

# add title
st.set_page_config(page_title="Predict Loan Approval", page_icon=":bank:")
st.title('Predict Loan Approval')
st.write('An app to predict whether the applicants loan request, will be rejected or accepted')

# ensure: the order is match between train, test, prediction:
# person_age,person_income,loan_amnt,loan_int_rate,loan_percent_income,credit_score,previous_loan_defaults_on_file

# CREATE UI
# create columns 1, 2
col1, col2 = st.columns(2)
with col1:
    person_age_col = st.number_input("Applicant's Age",
                                    placeholder="Type applicant's age...",
                                    min_value=18, max_value=75, value=None, step=1)

with col2:
    person_income_col = st.number_input("Applicant's Income",
                                    placeholder="Type applicant's annual income...",
                                    min_value=8000, max_value=3000000, value=None, step=1000)
    

# create columns 3, 4
col3, col4 = st.columns(2)

with col3:
    loan_amount_col = st.number_input("Applicant's Loan Amount",
                                    placeholder="Type applicant's loan amount requested...",
                                    min_value=400, max_value=100000, value=None)

with col4:
    loan_percent_income_col = st.number_input("Applicant's Loan Percent Income",
                                            placeholder="Type applicant's loan amount as a percentage of annual income...",
                                            min_value=0.0001, max_value=0.7, value=None)
    

# create columns 5, 6
col5, col6 = st.columns(2)

with col5:
    loan_int_rate_col = st.number_input("Applicant's Loan Interest Rate",
                                        placeholder="Type applicant's loan interest rate...",
                                        min_value=3, max_value=22, value=None, step=1)
    
with col6:
    credit_score_col = st.number_input("Applicant's Credit Score",
                                        placeholder="Type applicant's credit score...",
                                        min_value=300, max_value=850, value=None, step=1)
    

# last column
previous_loan_default_on_file_col = st.selectbox(
            "Does the applicant has previous loan default on file?",
            ("Yes", "No"),
            index=None,
            placeholder="Select applicant's indicator of previous loan defaults...")

# save input in JSON format, to be sent to Flask API
user_input = {
    "person_age": person_age_col,
    "person_income": person_income_col,
    "loan_amnt": loan_amount_col,
    "loan_int_rate": loan_int_rate_col,
    "loan_percent_income": loan_percent_income_col,
    "credit_score": credit_score_col,
    "previous_loan_defaults_on_file": previous_loan_default_on_file_col
}

# decode label from API result and show the prediction result
def show_prediction(prediction):
    if prediction == 0:
        return st.info(f"Loan Request is: Rejected", icon="🚨")
    else:
        return st.error(f"Loan Request get Accepted", icon="🔥")

# show the result
if st.button("Predict Loan Approval"):
    try:
        response = requests.post("http://localhost:8000/predict", json=user_input)
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            show_prediction(prediction)
        else:
            st.error("Prediction failed. Server error or invalid response.")
    except Exception as e:
        st.error(f"❌ Failed to contact Flask API: {e}")
