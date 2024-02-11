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

try:
    model = keras.models.load_model('Pneumonia.h5')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define class labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Sidebar navigation
page = st.sidebar.selectbox("Navbar", ["Prediction", "Performance Analysis", "Processed Pixels"])

if page == "Prediction":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])
    
        # Model performance analysis
    if uploaded_file is not None:
            
            # Display the uploaded image for training
            st.image(uploaded_file, caption="Uploaded Image (Training)", use_column_width=True)
    
            # Load and preprocess the test image
            st.write("Processing the image...")
    
            # Process the image and perform inference
            test_image = image.load_img(uploaded_file, target_size=(224, 224))  # Change target size to (224, 224)
            st.image(test_image, caption="Processed Image (Training)", use_column_width=True)
    
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0  # Normalize
    
            # Perform inference for prediction
            #st.write("Performing inference...")
    
            predictions = model.predict(test_image)
    
            # Display probability scores for each class
            st.write("Class Probabilities:")
            display_class_probabilities(predictions)
    
            # Print the classification label with probability
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_probability = predictions[0][predicted_class_index] * 100
            st.success(f'Predicted Class: {predicted_class_label} with {predicted_class_probability:.2f}% probability')

elif page == "Performance Analysis":
    # Perform inference for performance analysis
    st.subheader("Model Performance Analysis")
    st.text("CNN Model Classification Report")
    st.image('classificationreport.PNG', caption="Model Classification Report", use_column_width=True)
   
    st.text("Training and Validation Graph")
    st.image('training_validation.png', caption="Training and Validation", use_column_width=True)
    
    
    st.text("Performance Analysis For Normal Scan")
    st.image('NORMAL_metrics.png', caption="Normal Lungs", use_column_width=True)
    st.text("Performance Analysis For Pneumonia Scan")
    st.image('PNEUMONIA_metrics.png', caption="Pneumonia", use_column_width=True)
   
    st.subheader("Model Confusion Matrix")
    st.image('confusion_matrix.png', caption="Confusion Matrix", use_column_width=True)

