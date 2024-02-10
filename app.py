import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model('Pneumonia.h5', compile=False)

# Define the image size for model input
img_size = (224, 224)

# Function to make predictions
def predict_pneumonia(image):
    # Preprocess the image
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = loaded_model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    return class_index, confidence

# Streamlit App
st.title("Pneumonia Detection App")

# Upload Image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction on the uploaded image
    class_index, confidence = predict_pneumonia(image)

    # Get class label
    g_dict = train_gen.class_indices
    classes = list(g_dict.keys())
    class_label = classes[class_index]

    st.subheader("Prediction Result:")
    st.write(f"The model predicts that the image contains: **{class_label}**")
    st.write(f"Confidence: {confidence:.2%}")
