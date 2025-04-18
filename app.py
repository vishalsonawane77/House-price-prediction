import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("house_price_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define the correct feature order (same as during training)
feature_order = [
    'area', 'bedrooms', 'bathrooms', 'stories',
    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
]

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below to predict the price:")

# Start layout with two columns
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=3000, step=100)
    bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
    bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)
    stories = st.selectbox("Stories", options=[1, 2, 3, 4], index=1)
    parking = st.slider("Parking Spaces", 0, 3, 1)

with col2:
    mainroad = st.selectbox("Main Road Access", ['yes', 'no'])
    guestroom = st.selectbox("Guest Room", ['yes', 'no'])
    basement = st.selectbox("Basement", ['yes', 'no'])
    hotwaterheating = st.selectbox("Hot Water Heating", ['yes', 'no'])
    airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])
    prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
    furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# Map input to model format with proper column order
def prepare_input():
    input_dict = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': label_encoders['mainroad'].transform([mainroad])[0],
        'guestroom': label_encoders['guestroom'].transform([guestroom])[0],
        'basement': label_encoders['basement'].transform([basement])[0],
        'hotwaterheating': label_encoders['hotwaterheating'].transform([hotwaterheating])[0],
        'airconditioning': label_encoders['airconditioning'].transform([airconditioning])[0],
        'parking': parking,
        'prefarea': label_encoders['prefarea'].transform([prefarea])[0],
        'furnishingstatus': label_encoders['furnishingstatus'].transform([furnishingstatus])[0]
    }
    df_input = pd.DataFrame([input_dict])
    return df_input[feature_order]  # Reorder columns

# Centered prediction button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("💰 Predict Price"):
    input_data = prepare_input()
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ₹ {int(predicted_price):,}")
st.markdown("</div>", unsafe_allow_html=True)
