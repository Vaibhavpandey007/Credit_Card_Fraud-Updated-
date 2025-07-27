import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
def load_model():
    model = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# Load data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

st.title('Credit Card Fraud Detection')
st.write('Enter your Transaction ID to check if it is Legit or Fraudulent.')

# User input
txn_id = st.number_input('Enter Transaction ID (Row Number):', min_value=0, step=1)

if st.button('Check Transaction'):
    df = load_data()
    if txn_id >= len(df):
        st.error('Invalid Transaction ID!')
    else:
        row = df.iloc[[txn_id]].drop(['Class'], axis=1)
        model, scaler = load_model()
        row_scaled = scaler.transform(row)
        pred = model.predict(row_scaled)[0]
        if pred == 1:
            st.error('Fraudulent Transaction!')
        else:
            st.success('Legit Transaction!')
        st.write('Transaction Details:', df.iloc[txn_id]) 