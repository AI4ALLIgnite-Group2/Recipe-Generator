import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model_filename.pkl')

st.title('Recipe Rating Predictor')

# Create sliders for input features
ingredient1 = st.slider('Ingredient 1 Quantity', 0, 100, 25)
ingredient2 = st.slider('Ingredient 2 Quantity', 0, 100, 25)
ingredient3 = st.slider('Ingredient 3 Quantity', 0, 100, 25)

if st.button('Predict Rating'):
    # Prepare the input data
    input_data = np.array([[ingredient1, ingredient2, ingredient3]])
    prediction = model.predict(input_data)
    st.write(f'The predicted rating for your recipe is: {prediction[0]}')