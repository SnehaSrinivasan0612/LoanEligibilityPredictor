# Loan Eligibility Predictor

This Streamlit application predicts whether a loan application will be approved based on various factors such as income, credit history, education, and more.

## Features

- Interactive form for loan application details
- Real-time loan eligibility prediction
- Data visualization including correlation heatmap and feature importance
- Confidence score for predictions

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have the `LoanApprovalPrediction.csv` file in the same directory as `app.py`
2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## Input Fields

- Gender
- Marital Status
- Number of Dependents
- Education
- Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

## Output

The application provides:
- Loan approval prediction (Approved/Rejected)
- Confidence score
- Feature importance visualization
- Data analysis visualizations 