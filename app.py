import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('Pneumonia.h5')

# Define class labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Streamlit App
st.title("Pneumonia Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)

    # Display prediction result
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
