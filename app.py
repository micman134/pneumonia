import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras

def display_class_probabilities(predictions):
    prob_df = pd.DataFrame({'Cancer Class Category': class_labels, 'Probability (%)': predictions[0] * 100})
    st.table(prob_df.style.format({'Probability (%)': '{:.2f}%'}))

model = keras.models.load_model('Pneumonia.h5')

# Define class labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

# Model performance analysis
if uploaded_file is not None:

    # Display the uploaded image for training
    st.image(uploaded_file, caption="Uploaded Image (Training)", use_column_width=True)

    
    predictions = model.predict(test_image)

    # Display probability scores for each class
    st.write("Class Probabilities:")
    display_class_probabilities(predictions)

    # Print the classification label with probability
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    predicted_class_probability = predictions[0][predicted_class_index] * 100
    st.success(f'Predicted Class: {predicted_class_label} with {predicted_class_probability:.2f}% probability')
