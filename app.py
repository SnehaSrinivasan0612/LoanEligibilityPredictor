import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Set page config
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

# Title
st.title("Loan Eligibility Predictor")
st.write("This app predicts whether a loan application will be approved based on various factors.")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("LoanApprovalPrediction.csv")
    data.drop(['Loan_ID'], axis=1, inplace=True)
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            # Categorical column: fill with mode (most frequent value)
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # Numeric column: fill with mean
            data[col] = data[col].fillna(data[col].mean())
    
    # Encode categorical variables
    label_encoder = preprocessing.LabelEncoder()
    obj = (data.dtypes == 'object')
    for col in list(obj[obj].index):
        data[col] = label_encoder.fit_transform(data[col])
    
    return data

# Load data
data = load_data()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Loan Application Details")
    
    # Create form for user input
    with st.form("loan_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income (in thousands)", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income (in thousands)", min_value=0)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
        loan_term = st.selectbox("Loan Term (in months)", ["12", "36", "60", "84", "120", "180", "240", "300", "360", "480"])
        credit_history = st.selectbox("Credit History", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        submitted = st.form_submit_button("Predict Loan Eligibility")

with col2:
    st.subheader("Data Analysis")
    
    # Show correlation heatmap
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True, ax=ax)
    st.pyplot(fig)

# Train model
@st.cache_resource
def train_model():
    X = data.drop(['Loan_Status'], axis=1)
    Y = data['Loan_Status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    
    rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
    rfc.fit(X_train, Y_train)
    return rfc, X.columns

# Get model and feature names
model, feature_names = train_model()

# Process prediction when form is submitted
if submitted:
    # Prepare input data
    input_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Married': [1 if married == "Yes" else 0],
        'Dependents': [int(dependents.replace("+", ""))],
        'Education': [1 if education == "Graduate" else 0],
        'Self_Employed': [1 if self_employed == "Yes" else 0],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [int(loan_term)],
        'Credit_History': [1 if credit_history == "Yes" else 0],
        'Property_Area': [0 if property_area == "Urban" else (1 if property_area == "Semiurban" else 2)]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    # Display results
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.success("Loan Application is likely to be APPROVED!")
    else:
        st.error("Loan Application is likely to be REJECTED!")
    
    st.write(f"Confidence: {probability[0][prediction[0]]*100:.2f}%")
    
    # Show feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
    st.pyplot(fig)