import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras

# Function to display limited rows of the pixel table with loading percentage
def display_limited_rows_with_loading(pixel_table, num_rows=50):
    st.subheader(f"Pixel Values of the Processed Image (Showing {num_rows} rows)")
    st.write(pixel_table.head(num_rows))

# Function to display class probabilities in a table
def display_class_probabilities(predictions):
    prob_df = pd.DataFrame({'Cancer Class Category': class_labels, 'Probability (%)': predictions[0] * 100})
    st.table(prob_df.style.format({'Probability (%)': '{:.2f}%'}))

def display_class_probabilities(predictions):
    prob_df = pd.DataFrame({'Class Category': class_labels, 'Probability (%)': predictions[0] * 100})
    st.table(prob_df.style.format({'Probability (%)': '{:.2f}%'}))

try:
    model = keras.models.load_model('Pneumonia.h5')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define class labels
class_labels = ['NORMAL', 'PNEUMONIA']

# Sidebar navigation
page = st.sidebar.selectbox("Navbar", ["Pneumonia Prediction", "Model Analysis", "Processed Image Pixels Table"])

if page == "Pneumonia Prediction":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])
    
        # Model performance analysis
    if uploaded_file is not None:
            
            # Display the uploaded image for training
            st.image(uploaded_file, caption="Uploaded Image For Training", use_column_width=True)
    
            # Load and preprocess the test image
            st.write("Processing the image...")
    
            # Process the image and perform inference
            test_image = image.load_img(uploaded_file, target_size=(224, 224))  # Change target size to (224, 224)
            st.image(test_image, caption="Train and Processed Image", use_column_width=True)
    
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0  # Normalize
    
            # Perform inference for prediction
            #st.write("Performing inference...")
    
            predictions = model.predict(test_image)
    
            # Display probability scores for each class
            st.write("Prediction Estimate:")
            display_class_probabilities(predictions)
    
            # Print the classification label with probability
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_probability = predictions[0][predicted_class_index] * 100
            st.success(f'Model Prediction: {predicted_class_label} with {predicted_class_probability:.2f}% probability')

elif page == "Model Analysis":
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

elif page == "Processed Image Pixels Table":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image for training
        st.image(uploaded_file, caption="Uploaded Image (Training)", use_column_width=True)

        # Load and preprocess the test image
        st.write("Processing the image...")
        
        test_image = image.load_img(uploaded_file, target_size=(150, 150))
        st.image(test_image, caption="Processed Image (Trained)", use_column_width=True)
        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Display a table showing pixel values
        pixel_table = pd.DataFrame(test_image.reshape(-1, 3), columns=['Red', 'Green', 'Blue'])
        display_limited_rows_with_loading(pixel_table)

        # Download button for CSV file
        st.download_button(
            label="Download Pixel Table as CSV",
            data=pixel_table.to_csv(index=False, encoding='utf-8'),
            file_name="pixel_table.csv",
            key="download_csv"
        )

