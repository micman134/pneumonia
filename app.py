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

    test_image = image.load_img(uploaded_file, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize

    # Perform inference for prediction
    st.write("Performing inference...")
    
    # Add processing stage: Displaying intermediate layer activations
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
    intermediate_output = intermediate_layer_model.predict(test_image)

    st.subheader("Intermediate Layer Activations")

    # Create an image with the desired colormap using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(intermediate_output[0, :, :, 0], cmap='viridis')
    ax.axis('off')
    st.pyplot(fig)

    predictions = model.predict(test_image)

    # Display probability scores for each class
    st.write("Class Probabilities:")
    display_class_probabilities(predictions)

    # Print the classification label with probability
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    predicted_class_probability = predictions[0][predicted_class_index] * 100
    st.success(f'Predicted Class: {predicted_class_label} with {predicted_class_probability:.2f}% probability')
