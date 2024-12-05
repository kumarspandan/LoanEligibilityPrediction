import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu

# Setting the page configuration
st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide", page_icon="🏦")

# Loading my trained SVM model
mymodel_path = "C:/Users/LENOVO/Desktop/LoanEligibilityPrediction/savedModels/svmloan.sav"
svm_model = pickle.load(open(mymodel_path, 'rb'))

# Sidebar for navigation (only keep SVM option)
with st.sidebar:
    selected = option_menu('Loan Eligibility Prediction System',
                           ['SVM Prediction'],
                           menu_icon='bank',
                           icons=['diagram-3-fill'],
                           default_index=0)

# Create input fields for loan prediction
def get_user_input():
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender', ('Male', 'Female'))
    with col2:
        Married = st.selectbox('Married', ('Yes', 'No'))
    with col3:
        Dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))

    with col1:
        Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
    with col2:
        Self_Employed = st.selectbox('Self Employed', ('Yes', 'No'))
    with col3:
        ApplicantIncome = st.text_input('Applicant Income')

    with col1:
        CoapplicantIncome = st.text_input('Coapplicant Income')
    with col2:
        LoanAmount = st.text_input('Loan Amount')
    with col3:
        Loan_Amount_Term = st.text_input('Loan Amount Term')

    with col1:
        Credit_History = st.selectbox('Credit History', ('1', '0'))
    with col2:
        Property_Area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

    user_data = [Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
                 CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]

    return user_data

# Function to prepare input for prediction
def prepare_input(user_input):
    # Convert categorical inputs to numerical
    Gender = 1 if user_input[0] == 'Male' else 0
    Married = 1 if user_input[1] == 'Yes' else 0
    Dependents = int(user_input[2]) if user_input[2] != '3+' else 3
    Education = 1 if user_input[3] == 'Graduate' else 0
    Self_Employed = 1 if user_input[4] == 'Yes' else 0
    ApplicantIncome = float(user_input[5])
    CoapplicantIncome = float(user_input[6])
    LoanAmount = float(user_input[7])
    Loan_Amount_Term = float(user_input[8])
    Credit_History = int(user_input[9])
    Property_Area = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[user_input[10]]

    input_data = [[Gender, Married, Dependents, Education, Self_Employed,
                   ApplicantIncome, CoapplicantIncome, LoanAmount,
                   Loan_Amount_Term, Credit_History, Property_Area]]

    # input_data = scaler.transform(input_data)  # Uncomment if using scaling

    return np.array(input_data)

# SVM Prediction tab
if selected == 'SVM Prediction':
    st.title('Loan Prediction using SVM Model')
    user_input = get_user_input()

    if st.button('Predict using SVM'):
        input_data = prepare_input(user_input)

        # Make prediction using the SVM model
        try:
            prediction = svm_model.predict(input_data)
            st.success("Congratulations! Loan Approved" if prediction[0] == 1 else "Sorry! Loan Not Approved")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Display mappings at the bottom of the screen in a nice format
st.markdown("""
    **Mappings used for prediction:**

    - **Married:** Yes → 1, No → 0
    - **Gender:** Male → 1, Female → 0
    - **Self Employed:** Yes → 1, No → 0
    - **Property Area:** Urban → 2, Semiurban → 1, Rural → 0
    - **Education:** Graduate → 1, Not Graduate → 0
""")