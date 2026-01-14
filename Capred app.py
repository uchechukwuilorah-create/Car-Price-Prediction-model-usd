import streamlit as st
import pandas as pd
import pickle

st.title('Car Selling Price Prediction App')

# Load the trained model
# Ensure 'Main intern project.pkl' is in the same directory as this app.py or provide the full path
with open('Main intern project2.pkl', 'rb') as f:
    model = pickle.load(f)

st.header('Enter Car Details for Prediction')

# Input fields for features
year = st.slider('Manufacturing Year', min_value=2000, max_value=2024, value=2015)
present_price = st.number_input('Present Price (in Dollars($))', min_value=0.1, max_value=50.0, value=5.0, step=0.1)
kms_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=30000, step=1000)

# Make prediction
if st.button('Predict Selling Price'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([{
        'Year': year,
        'Present_Price_USD': present_price,
        'Kms_Driven': kms_driven
    }])

    # Predict
    prediction = model.predict(input_data)

    st.success(f'The predicted selling price is: {prediction[0]:.2f} Lakhs')

st.markdown("""
This app predicts the selling price of a car based on its manufacturing year,
present price, and kilometers driven using a pre-trained machine learning model.

""")
