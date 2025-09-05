import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# -----------------------
# Load model and encoders
# -----------------------
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)

# -----------------------
# Streamlit App
# -----------------------
st.title('Customer Churn Prediction')

# Input fields
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance', min_value=0.0, step=100.0)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=100.0)
credit_score = st.number_input('Credit Score', min_value=0, max_value=850, step=1)

# -----------------------
# Submit button
# -----------------------
if st.button('Predict'):
    # Prepare input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary],
    })

    # -----------------------
    # One-hot encode geography
    # -----------------------
    # Use sparse=False to get a dense array
    geography_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()

    # Make sure number of columns matches encoder
    geography_encoded_df = pd.DataFrame(
        geography_encoded,
        columns=one_hot_encoder_geography.get_feature_names_out(['Geography'])
    )

    # Concatenate input data
    input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

    # -----------------------
    # Ensure feature order matches training
    # -----------------------
    final_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    ] + list(one_hot_encoder_geography.get_feature_names_out(['Geography']))

    input_data = input_data[final_columns]

    # -----------------------
    # Scale input and predict
    # -----------------------
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    prediction_probability = prediction[0][0]

    st.write(f'Prediction probability: {prediction_probability:.2f}')
    if prediction_probability > 0.5:
        st.success('The customer is likely to leave the bank')
    else:
        st.info('The customer is likely to stay with the bank')
